import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import prepare_dataset
from experiments.utils import construct_passport_kwargs_from_dict
from models.alexnet_passport import AlexNetPassport
from models.alexnet_passport_private import AlexNetPassportPrivate
from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.losses.sign_loss import SignLoss
from models.resnet_passport import ResNet18Passport
from models.resnet_passport_private import ResNet18Private


class DatasetArgs():
    pass


def train_maximize(origpassport, fakepassport, model, optimizer, criterion, trainloader, device, scheme):
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

        if scheme == 1:
            pred = model(d)
        else:
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

        (loss + signloss + maximizeloss).backward()

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


def test(model, criterion, valloader, device, scheme):
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

            if scheme == 1:
                pred = model(d)
            else:
                pred = model(d, ind=1)

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


def run_maximize(rep=1, flipperc=0, arch='alexnet', dataset='cifar10', scheme=1,
                 loadpath='', passport_config='', tagnum=1):
    epochs = {
        'imagenet1000': 40
    }.get(dataset, 100)
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
    model.load_state_dict(sd)

    for param in model.parameters():
        param.requires_grad_(False)

    passblocks = []
    origpassport = []
    fakepassport = []

    for m in model.modules():
        if isinstance(m, PassportBlock) or isinstance(m, PassportPrivateBlock):
            passblocks.append(m)

            if scheme == 1:
                keyname = 'key'
                skeyname = 'skey'
            else:
                keyname = 'key_private'
                skeyname = 'skey_private'

            key, skey = m.__getattr__(keyname).data.clone(), m.__getattr__(skeyname).data.clone()
            origpassport.append(key.to(device))
            origpassport.append(skey.to(device))

            m.__delattr__(keyname)
            m.__delattr__(skeyname)

            # re-initialize the key and skey, but by adding noise on it
            m.register_parameter(keyname, nn.Parameter(key.clone() + torch.randn(*key.size()) * 0.001))
            m.register_parameter(skeyname, nn.Parameter(skey.clone() + torch.randn(*skey.size()) * 0.001))
            fakepassport.append(m.__getattr__(keyname))
            fakepassport.append(m.__getattr__(skeyname))

    if flipperc != 0:
        print(f'Reverse {flipperc * 100:.2f}% of binary signature')
        for m in passblocks:
            mflip = flipperc
            if scheme == 1:
                oldb = m.sign_loss.b
            else:
                oldb = m.sign_loss_private.b
            newb = oldb.clone()

            npidx = np.arange(len(oldb))
            randsize = int(oldb.view(-1).size(0) * mflip)
            randomidx = np.random.choice(npidx, randsize, replace=False)

            newb[randomidx] = oldb[randomidx] * -1  # reverse bit
            if scheme == 1:
                m.sign_loss.set_b(newb)
            else:
                m.sign_loss_private.set_b(newb)

    model.to(device)

    optimizer = torch.optim.SGD(fakepassport,
                                lr=lr,
                                momentum=0.9,
                                weight_decay=0.0005)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                  [int(epochs * 0.5), int(epochs * 0.75)],
    #                                                  0.1)
    scheduler = None
    criterion = nn.CrossEntropyLoss()

    history = []

    dirname = f'logs/passport_attack_3/{loadpath.split("/")[1]}/{loadpath.split("/")[2]}'
    os.makedirs(dirname, exist_ok=True)

    def run_cs():
        cs = []

        for d1, d2 in zip(origpassport, fakepassport):
            d1 = d1.view(d1.size(0), -1)
            d2 = d2.view(d2.size(0), -1)

            cs.append(F.cosine_similarity(d1, d2).item())

        return cs

    print('Before training')
    res = {}
    valres = test(model, criterion, valloader, device, scheme)
    for key in valres: res[f'valid_{key}'] = valres[key]
    with torch.no_grad():
        cs = run_cs()

        mseloss = 0
        for l, r in zip(origpassport, fakepassport):
            mse = F.mse_loss(l, r)
            mseloss += mse.item()
        mseloss /= len(origpassport)

    print(f'MSE of Real and Maximize passport: {mseloss:.4f}')
    print(f'Cosine Similarity of Real and Maximize passport: {sum(cs) / len(origpassport):.4f}')
    print()

    res['epoch'] = 0
    res['cosine_similarity'] = cs
    res['flipperc'] = flipperc
    res['train_mseloss'] = mseloss

    history.append(res)

    torch.save({'origpassport': origpassport,
                'fakepassport': fakepassport,
                'state_dict': model.state_dict()},
               f'{dirname}/{arch}-{scheme}-last-{dataset}-{rep}-{tagnum}-{flipperc:.1f}-e0.pth')

    for ep in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()

        print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        print(f'Epoch {ep:3d}:')
        print('Training')
        trainres = train_maximize(origpassport, fakepassport, model, optimizer, criterion, trainloader, device, scheme)

        print('Testing')
        valres = test(model, criterion, valloader, device, scheme)

        res = {}

        for key in trainres: res[f'train_{key}'] = trainres[key]
        for key in valres: res[f'valid_{key}'] = valres[key]
        res['epoch'] = ep
        res['flipperc'] = flipperc

        with torch.no_grad():
            cs = run_cs()
            res['cosine_similarity'] = cs

        print(f'Cosine Similarity of Real and Maximize passport: '
              f'{sum(cs) / len(origpassport):.4f}')
        print()

        history.append(res)

        torch.save({'origpassport': origpassport,
                    'fakepassport': fakepassport,
                    'state_dict': model.state_dict()},
                   f'{dirname}/{arch}-{scheme}-{dataset}-{rep}-{tagnum}-{flipperc:.1f}-last.pth')

        histdf = pd.DataFrame(history)
        histdf.to_csv(f'{dirname}/{arch}-{scheme}-history-{dataset}-{rep}-{tagnum}-{flipperc:.1f}.csv')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='fake attack 3: create another passport maximized from current passport')
    parser.add_argument('--rep', default=1, type=int,
                        help='training id')
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet18'])
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet1000'])
    parser.add_argument('--flipperc', default=0, type=float,
                        help='flip percentange 0~1')
    parser.add_argument('--scheme', default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--loadpath', default='', help='path to model to be attacked')
    parser.add_argument('--passport-config', default='', help='path to passport config')
    parser.add_argument('--tagnum', default=torch.randint(100000, ()).item(), type=int,
                        help='tag number of the experiment')
    args = parser.parse_args()

    run_maximize(args.rep, args.flipperc,
                 args.arch, args.dataset,
                 args.scheme, args.loadpath,
                 args.passport_config, args.tagnum)
