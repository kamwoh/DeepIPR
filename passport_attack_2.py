import json
import os
import time

import pandas as pd
import torch
import torch.nn as nn

import passport_generator
from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict
from models.alexnet_normal import AlexNetNormal
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.resnet_normal import ResNet18


class DatasetArgs():
    pass


def train(model, optimizer, criterion, trainloader, device):
    model.train()
    loss_meter = 0
    acc_meter = 0
    start_time = time.time()
    for k, (d, t) in enumerate(trainloader):
        d = d.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        pred = model(d)
        loss = criterion(pred, t)

        loss.backward()

        optimizer.step()

        acc = (pred.max(dim=1)[1] == t).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()

        print(f'Batch [{k + 1}/{len(trainloader)}]: '
              f'Loss: {loss_meter / (k + 1):.4f} '
              f'Acc: {acc_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
              end='\r')

    print()
    loss_meter /= len(trainloader)
    acc_meter /= len(trainloader)

    return {'loss': loss_meter,
            'acc': acc_meter,
            'time': start_time - time.time()}


def test(model, criterion, valloader, device):
    model.eval()
    loss_meter = 0
    acc_meter = 0
    start_time = time.time()

    with torch.no_grad():
        for k, (d, t) in enumerate(valloader):
            d = d.to(device)
            t = t.to(device)

            pred = model(d)
            loss = criterion(pred, t)

            acc = (pred.max(dim=1)[1] == t).float().mean()

            loss_meter += loss.item()
            acc_meter += acc.item()

            print(f'Batch [{k + 1}/{len(valloader)}]: '
                  f'Loss: {loss_meter / (k + 1):.4f} '
                  f'Acc: {acc_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
                  end='\r')

    print()

    loss_meter /= len(valloader)
    acc_meter /= len(valloader)

    return {'loss': loss_meter,
            'acc': acc_meter,
            'time': time.time() - start_time}


def set_intermediate_keys(passport_model, pretrained_model, x, y=None):
    with torch.no_grad():
        for pretrained_layer, passport_layer in zip(pretrained_model.features, passport_model.features):
            if isinstance(passport_layer, PassportBlock) or isinstance(passport_layer, PassportPrivateBlock):
                passport_layer.set_key(x, y)

            x = pretrained_layer(x)
            if y is not None:
                y = pretrained_layer(y)


def get_passport(passport_data, device):
    n = 20  # any number
    key_y, y_inds = passport_generator.get_key(passport_data, n)
    key_y = key_y.to(device)

    key_x, x_inds = passport_generator.get_key(passport_data, n)
    key_x = key_x.to(device)

    return key_x, key_y


def run_attack_2(rep=1, arch='alexnet', dataset='cifar10', scheme=1, loadpath='',
                 passport_config='passport_configs/alexnet_passport.json', tagnum=1):
    epochs = 100
    batch_size = 64
    nclass = {
        'cifar100': 100,
        'imagenet1000': 1000
    }.get(dataset, 10)
    inchan = 3
    lr = 0.01
    device = torch.device('cuda')

    trainloader, valloader = prepare_dataset({'transfer_learning': False,
                                              'dataset': dataset,
                                              'tl_dataset': '',
                                              'batch_size': batch_size})
    passport_kwargs, plkeys = construct_passport_kwargs_from_dict({'passport_config': json.load(open(passport_config)),
                                                                   'norm_type': 'bn',
                                                                   'sl_ratio': 0.1,
                                                                   'key_type': 'shuffle'},
                                                                  True)

    if arch == 'alexnet':
        model = AlexNetNormal(inchan, nclass, 'bn' if scheme == 1 else 'gn')
    else:
        model = ResNet18(num_classes=nclass,
                         norm_type='bn' if scheme == 1 else 'gn')

    model.to(device)

    sd = torch.load(loadpath)
    model.load_state_dict(sd, strict=False)  # need to load with strict because passport model no scale and bias

    for param in model.parameters():
        param.requires_grad_(False)

    # for fidx in [0, 2]:
    #     model.features[fidx].bn.weight.data.copy_(sd[f'features.{fidx}.scale'])
    #     model.features[fidx].bn.bias.data.copy_(sd[f'features.{fidx}.bias'])

    for fidx in plkeys:
        fidx = int(fidx)
        model.features[fidx].bn.weight.data.normal_().sign_().mul_(0.5)
        model.features[fidx].bn.bias.data.zero_()

        model.features[fidx].bn.weight.requires_grad_(True)
        model.features[fidx].bn.bias.requires_grad_(True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=lr,
                                momentum=0.9,
                                weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  [int(epochs * 0.5), int(epochs * 0.75)],
    #                                                  0.1)
    scheduler = None
    criterion = nn.CrossEntropyLoss()

    history = []

    print('Before training')
    valres = test(model, criterion, valloader, device)
    res = {}
    for key in valres: res[f'valid_{key}'] = valres[key]
    res['epoch'] = 0
    history.append(res)
    print()

    dirname = f'logs/passport_attack_2/{loadpath.split("/")[1]}/{loadpath.split("/")[2]}'
    os.makedirs(dirname, exist_ok=True)

    for ep in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()

        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        print(f'Epoch {ep:3d}:')
        print('Training')
        trainres = train(model, optimizer, criterion, trainloader, device)

        print('Testing')
        valres = test(model, criterion, valloader, device)

        print()

        res = {}

        for key in trainres: res[f'train_{key}'] = trainres[key]
        for key in valres: res[f'valid_{key}'] = valres[key]
        res['epoch'] = ep

        history.append(res)

        torch.save(model.state_dict(),
                   f'{dirname}/{arch}-{scheme}-last-{dataset}-{rep}-{tagnum}.pth')

    histdf = pd.DataFrame(history)
    histdf.to_csv(f'{dirname}/{arch}-{scheme}-history-{dataset}-{tagnum}.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='fake attack 2: reverse engineer passport scale & bias')
    parser.add_argument('--rep', default=1, type=int)
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet1000'])
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    parser.add_argument('--tagnum', default=torch.randint(100000, ()).item(), type=int,
                        help='tag number of the experiment')
    args = parser.parse_args()

    run_attack_2(args.rep,
                 args.arch,
                 args.dataset,
                 args.scheme,
                 args.loadpath,
                 args.passport_config,
                 args.tagnum)
