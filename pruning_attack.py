import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict
from models.alexnet_passport import AlexNetPassport
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.resnet_passport import ResNet18Passport
from models.resnet_passport_private import ResNet18Private


def detect_signature(model):
    detection = {}

    for name, m in model.named_modules():
        if isinstance(m, (PassportBlock, PassportPrivateBlock)):
            btarget = m.b
            bembed = m.get_scale(True).detach().view(-1)

            detection_rate = (btarget == bembed.sign()).sum().item() / m.conv.out_channels
            detection[name] = detection_rate

    return detection


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    '''
    all_weights = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


def pruning_resnet(model, pruning_perc):
    if pruning_perc == 0:
        return

    allweights = []
    for p in model.parameters():
        allweights += p.data.cpu().abs().numpy().flatten().tolist()

    allweights = np.array(allweights)
    threshold = np.percentile(allweights, pruning_perc)
    for p in model.parameters():
        mask = p.abs() > threshold
        p.data.mul_(mask.float())


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


def main(arch='alexnet', dataset='cifar10', scheme=1, loadpath='',
         passport_config='passport_configs/alexnet_passport.json', tagnum=1):
    batch_size = 64
    nclass = {
        'cifar100': 100,
        'imagenet1000': 1000
    }.get(dataset, 10)
    inchan = 3
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
    criterion = nn.CrossEntropyLoss()
    prunedf = []
    for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        model.load_state_dict(sd)
        pruning_resnet(model, perc)
        model = model.to(device)

        res = detect_signature(model)

        res['perc'] = perc
        res['tag'] = arch
        res['dataset'] = dataset
        res.update(test(model, criterion, valloader, device))
        prunedf.append(res)

    dirname = f'logs/pruning_attack/{loadpath.split("/")[1]}/{loadpath.split("/")[2]}'
    os.makedirs(dirname, exist_ok=True)

    histdf = pd.DataFrame(prunedf)
    histdf.to_csv(f'{dirname}/{arch}-{scheme}-history-{dataset}-{tagnum}.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='pruning attack: measure sig. det. & acc pruning')
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet1000'])
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    parser.add_argument('--tagnum', default=torch.randint(100000, ()).item(), type=int,
                        help='tag number of the experiment')

    args = parser.parse_args()

    main(args.arch,
         args.dataset,
         args.scheme,
         args.loadpath,
         args.passport_config,
         args.tagnum)
