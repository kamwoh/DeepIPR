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
from models.alexnet_passport import AlexNetPassport
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.losses.sign_loss import SignLoss
from models.resnet_normal import ResNet18
from models.resnet_passport import ResNet18Passport
from models.resnet_passport_private import ResNet18Private


class DatasetArgs():
    pass


def train_maximize(origpassport, fakepassport, model, optimizer, criterion, trainloader, device):
    model.train()
    loss_meter = 0
    signloss_meter = 0
    maximizeloss_meter = 0
    mseloss_meter = 0
    csloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    start_time = time.time()
    mse_criterion = nn.MSELoss()
    cs_criterion = nn.CosineSimilarity()
    for k, (d, t) in enumerate(trainloader):
        d = d.to(device)
        t = t.to(device)

        optimizer.zero_grad()

        pred = model(d, ind=1)
        loss = criterion(pred, t)

        signloss = torch.tensor(0.).to(device)
        signacc = torch.tensor(0.).to(device)
        count = 0
        for m in model.modules():
            if isinstance(m, SignLoss):
                signloss += m.loss
                signacc += m.acc
                count += 1

        maximizeloss = torch.tensor(0.).to(device)
        mseloss = torch.tensor(0.).to(device)
        csloss = torch.tensor(0.).to(device)
        for l, r in zip(origpassport, fakepassport):
            mse = mse_criterion(l, r)
            cs = cs_criterion(l.view(1, -1), r.view(1, -1)).mean()
            csloss += cs
            mseloss += mse
            maximizeloss += 1 / mse

        (loss + signloss + 2 * maximizeloss).backward()

        torch.nn.utils.clip_grad_norm_(fakepassport, 2)

        optimizer.step()

        acc = (pred.max(dim=1)[1] == t).float().mean()

        loss_meter += loss.item()
        acc_meter += acc.item()
        signloss_meter += signloss.item()
        signacc_meter += signacc.item() / count
        maximizeloss_meter += maximizeloss.item()
        mseloss_meter += mseloss.item()
        csloss_meter += csloss.item()

        print(f'Batch [{k + 1}/{len(trainloader)}]: '
              f'Loss: {loss_meter / (k + 1):.4f} '
              f'Acc: {acc_meter / (k + 1):.4f} '
              f'Sign Loss: {signloss_meter / (k + 1):.4f} '
              f'Sign Acc: {signacc_meter / (k + 1):.4f} '
              f'MSE Loss: {mseloss_meter / (k + 1):.4f} '
              f'Maximize Dist: {maximizeloss_meter / (k + 1):.4f} '
              f'CS: {csloss_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
              end='\r')

    print()
    loss_meter /= len(trainloader)
    acc_meter /= len(trainloader)
    signloss_meter /= len(trainloader)
    signacc_meter /= len(trainloader)
    maximizeloss_meter /= len(trainloader)
    mseloss_meter /= len(trainloader)
    csloss_meter /= len(trainloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
            'maximizeloss': maximizeloss_meter,
            'mseloss': mseloss_meter,
            'csloss': csloss_meter,
            'time': start_time - time.time()}


def test(model, criterion, valloader, device, ind=1):
    model.eval()
    loss_meter = 0
    signloss_meter = 0
    acc_meter = 0
    signacc_meter = 0
    start_time = time.time()

    with torch.no_grad():
        for k, (d, t) in enumerate(valloader):
            d = d.to(device)
            t = t.to(device)

            if ind == 0:
                pred = model(d)
            else:
                pred = model(d, ind=ind)

            loss = criterion(pred, t)

            signloss = torch.tensor(0.).to(device)
            signacc = torch.tensor(0.).to(device)
            count = 0

            for m in model.modules():
                if isinstance(m, SignLoss):
                    signloss += m.get_loss()
                    signacc += m.get_acc()
                    count += 1

            acc = (pred.max(dim=1)[1] == t).float().mean()

            loss_meter += loss.item()
            acc_meter += acc.item()
            signloss_meter += signloss.item()
            try:
                signacc_meter += signacc.item() / count
            except:
                pass

            print(f'Batch [{k + 1}/{len(valloader)}]: '
                  f'Loss: {loss_meter / (k + 1):.4f} '
                  f'Acc: {acc_meter / (k + 1):.4f} '
                  f'Sign Loss: {signloss_meter / (k + 1):.4f} '
                  f'Sign Acc: {signacc_meter / (k + 1):.4f} ({time.time() - start_time:.2f}s)',
                  end='\r')

    print()

    loss_meter /= len(valloader)
    acc_meter /= len(valloader)
    signloss_meter /= len(valloader)
    signacc_meter /= len(valloader)

    return {'loss': loss_meter,
            'signloss': signloss_meter,
            'acc': acc_meter,
            'signacc': signacc_meter,
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


def load_pretrained(arch, nclass):
    if arch == 'alexnet':
        pretrained_model = AlexNetNormal(3,
                                         nclass,
                                         norm_type='none',
                                         pretrained=True)
    else:
        pretrained_model = ResNet18(num_classes=nclass,
                                    norm_type='bn',
                                    pretrained=True)

    return pretrained_model


def run_attack_1(attack_rep=50, arch='alexnet', dataset='cifar10', scheme=1,
                 loadpath='', passport_config='passport_configs/alexnet_passport.json'):
    batch_size = 64
    nclass = {
        'cifar100': 100,
        'imagenet1000': 1000
    }.get(dataset, 10)
    inchan = 3
    lr = 0.01
    device = torch.device('cuda')

    # baselinepath = f'logs/alexnet_{dataset}/1/models/best.pth'
    passport_kwargs = construct_passport_kwargs_from_dict({'passport_config': json.load(open(passport_config)),
                                                           'norm_type': 'bn',
                                                           'sl_ratio': 0.1,
                                                           'key_type': 'shuffle'})

    if arch == 'alexnet':
        if scheme == 1:
            model = AlexNetPassport(inchan, nclass, passport_kwargs)
        else:
            model = AlexNetPassportPrivate(inchan, nclass, passport_kwargs)
    else:
        if scheme == 1:
            model = ResNet18Passport(num_classes=nclass, passport_kwargs=passport_kwargs)
        else:
            model = ResNet18Private(num_classes=nclass, passport_kwargs=passport_kwargs)

    sd = torch.load(loadpath)
    model.load_state_dict(sd, strict=False)

    for fidx in [0, 2]:
        model.features[fidx].bn.weight.data.copy_(sd[f'features.{fidx}.scale'])
        model.features[fidx].bn.bias.data.copy_(sd[f'features.{fidx}.bias'])

    passblocks = []

    for m in model.modules():
        if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
            passblocks.append(m)

    trainloader, valloader = prepare_dataset({'transfer_learning': False,
                                              'dataset': dataset,
                                              'tl_dataset': '',
                                              'batch_size': batch_size})
    passport_data = valloader

    pretrained_model = load_pretrained(arch, nclass).to(device)

    def reset_passport():
        print('Reset passport')
        x, y = get_passport(passport_data, device)
        set_intermediate_keys(model, pretrained_model, x, y)

    def run_test():
        res = {}
        valres = test(model, criterion, valloader, device, 1 if scheme != 1 else 0)
        for key in valres: res[f'valid_{key}'] = valres[key]
        res['attack_rep'] = 0
        return res

    criterion = nn.CrossEntropyLoss()

    os.makedirs('logs/passport_attack_1', exist_ok=True)

    history = []

    print('Before training')
    res = run_test()
    history.append(res)

    for r in range(attack_rep):
        print(f'Attack count: {r}')
        reset_passport()
        res = run_test()
        history.append(res)

    histdf = pd.DataFrame(history)
    histdf.to_csv(f'logs/passport_attack_1/{arch}-{scheme}-history-{dataset}-{attack_rep}.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='fake attack 1: random passport')
    parser.add_argument('--attack-rep', default=1, type=int)
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet1000'])
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    args = parser.parse_args()

    run_attack_1(args.attack_rep,
                 args.arch,
                 args.dataset,
                 args.scheme,
                 args.loadpath,
                 args.passport_config)
