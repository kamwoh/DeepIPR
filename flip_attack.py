import json
import os
import time

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


def flipping_alexnet(model, flipping_perc, fidxs, arch, device):
    conv_weights_to_reset = []
    total_weight_size = 0

    if arch == 'alexnet':
        sim = 0
        for fidx in fidxs:
            fidx = int(fidx)

            if model.features[fidx].scale is None:
                model.features[fidx].init_scale(True)
                model.features[fidx].init_bias(True)
                model.to(device)

            model.features[fidx].scale.data.copy_(model.features[fidx].get_scale(True).view(-1).detach())
            model.features[fidx].bias.data.copy_(model.features[fidx].get_bias(True).view(-1).detach())

            w = model.features[fidx].scale

            size = w.size(0)
            conv_weights_to_reset.append(w)
            total_weight_size += size
    else:
        raise ValueError('not support resnet')

    if flipping_perc == 0:
        return

    randidxs = torch.randperm(total_weight_size)
    idxs = randidxs[:int(total_weight_size * flipping_perc)]
    print(total_weight_size, len(idxs))
    sim = 0

    for w in conv_weights_to_reset:
        size = w.size(0)
        # wsize of first layer = 64, e.g. 0~63 - 64 = -64~-1, this is the indices within the first layer
        print(len(idxs), size)
        widxs = idxs[(idxs - size) < 0]

        # reset the weights but remains signature sign bit
        origsign = w.data.clone()
        newsign = origsign.clone()

        # reverse the sign on target bit
        newsign[widxs] *= -1

        # assign new signature
        w.data.copy_(newsign)

        sim += ((w.data.sign() == origsign.sign()).float().mean())

        # remove all indices from first layer
        idxs = idxs[(idxs - size) >= 0] - size

    print('Similarity', sim / len(conv_weights_to_reset))


def flipping(model, flipping_perc, plkeys, arch, device):
    conv_weights_to_reset = []
    total_weight_size = 0

    if arch == 'alexnet':
        sim = 0
        for fidx in plkeys:
            fidx = int(fidx)
            
            if model.features[fidx].scale is None:
                model.features[fidx].init_scale(True)
                model.features[fidx].init_bias(True)
                model.to(device)
            
            model.features[fidx].scale.data.copy_(model.features[fidx].get_scale(True).view(-1).detach())
            model.features[fidx].bias.data.copy_(model.features[fidx].get_bias(True).view(-1).detach())

            w = model.features[fidx].scale
            
            size = w.size(0)
            conv_weights_to_reset.append(w)
            total_weight_size += size
    else:
        for fidx in plkeys:
            layer_key, i, module_key = fidx.split('.')

            def get_layer(m):
                return m.__getattr__(layer_key)[int(i)].__getattr__(module_key)

            convblock = get_layer(model)
            
            if convblock.scale is None:
                convblock.init_scale(True)
                convblock.init_bias(True)
                model.to(device)

            convblock.scale.data.copy_(convblock.get_scale(True).view(-1).detach())
            convblock.bias.data.copy_(convblock.get_bias(True).view(-1).detach())
            
            w = convblock.scale
            size = w.size(0)
            conv_weights_to_reset.append(w)
            total_weight_size += size
            
    if flipping_perc == 0:
        return

    randidxs = torch.randperm(total_weight_size)
    idxs = randidxs[:int(total_weight_size * flipping_perc)]
    print(total_weight_size, len(idxs))
    sim = 0

    for w in conv_weights_to_reset:
        size = w.size(0)
        # wsize of first layer = 64, e.g. 0~63 - 64 = -64~-1, this is the indices within the first layer
        print(len(idxs), size)
        widxs = idxs[(idxs - size) < 0]

        # reset the weights but remains signature sign bit
        origsign = w.data.clone()
        newsign = origsign.clone()

        # reverse the sign on target bit
        newsign[widxs] *= -1

        # assign new signature
        w.data.copy_(newsign)

        sim += ((w.data.sign() == origsign.sign()).float().mean())

        # remove all indices from first layer
        idxs = idxs[(idxs - size) >= 0] - size
        
    print('Similarity', sim / len(conv_weights_to_reset))


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
                                                                   'norm_type': 'bn' if scheme == 1 else 'gn',
                                                                   'sl_ratio': 0.1,
                                                                   'key_type': 'shuffle'},
                                                                  True)

    fidxs = args.fidxs
    if arch == 'alexnet':
        if scheme == 1:
            model = AlexNetPassport(inchan, nclass, passport_kwargs)
        else:
            model = AlexNetPassportPrivate(inchan, nclass, passport_kwargs)
    else:
        assert fidxs == '', 'not support for resnet'
        if scheme == 1:
            model = ResNet18Passport(num_classes=nclass, passport_kwargs=passport_kwargs)
        else:
            model = ResNet18Private(num_classes=nclass, passport_kwargs=passport_kwargs)


    sd = torch.load(loadpath)
    criterion = nn.CrossEntropyLoss()
    prunedf = []
    for perc in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        model.load_state_dict(sd, strict=False)
        model = model.to(device)

        if fidxs != '':
            flipping_alexnet(model, perc / 100, fidxs.split(','), arch, device)
        else:
            flipping(model, perc / 100, plkeys, arch, device)

        res = detect_signature(model)

        res['perc'] = perc
        res['tag'] = arch
        res['dataset'] = dataset
        res['fidxs'] = fidxs
        res.update(test(model, criterion, valloader, device))
        prunedf.append(res)

    dirname = f'logs/flipping_attack/{loadpath.split("/")[1]}/{loadpath.split("/")[2]}'
    os.makedirs(dirname, exist_ok=True)

    histdf = pd.DataFrame(prunedf)
    histdf.to_csv(f'{dirname}/{arch}-{scheme}-history-{dataset}-{tagnum}.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='flipping attack: measure sig. det. & acc after flipping sign')
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet1000'])
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    parser.add_argument('--tagnum', default=torch.randint(100000, ()).item(), type=int,
                        help='tag number of the experiment')
    parser.add_argument('--fidxs', default='', help='flip index for alexnet')

    args = parser.parse_args()

    main(args.arch,
         args.dataset,
         args.scheme,
         args.loadpath,
         args.passport_config,
         args.tagnum)
