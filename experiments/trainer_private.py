import time

import torch
import torch.nn.functional as F

from models.layers.passportconv2d import PassportBlock
from models.layers.passportconv2d_private import PassportPrivateBlock
from models.losses.sign_loss import SignLoss


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


class TesterPrivate(object):
    def __init__(self, model, device, verbose=True):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.device = device
        self.verbose = verbose

    def test_signature(self):
        self.model.eval()

        res = {}
        avg_private = 0
        avg_public = 0
        count_private = 0
        count_public = 0

        with torch.no_grad():
            for name, m in self.model.named_modules():
                if isinstance(m, PassportPrivateBlock):
                    signbit = m.get_scale(ind=1).view(-1).sign()
                    privatebit = m.b

                    detection = (signbit == privatebit).float().mean().item()
                    res['private_' + name] = detection
                    avg_private += detection
                    count_private += 1

                if isinstance(m, PassportBlock):
                    signbit = m.get_scale().view(-1).sign()
                    publicbit = m.b

                    detection = (signbit == publicbit).float().mean().item()
                    res['public_' + name] = detection
                    avg_public += detection
                    count_public += 1

        if count_private != 0:
            print(f'Private Sign Detection Accuracy: {avg_private / count_private * 100:6.4f}')
        if count_public != 0:
            print(f'Public Sign Detection Accuracy: {avg_public / count_public * 100:6.4f}')

        return res

    def test(self, dataloader, msg='Testing Result', ind=0):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for i, load in enumerate(dataloader):
                data, target = load[:2]
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                pred = self.model(data, ind=ind)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
                pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)
                if self.verbose:
                    print(f'{msg} [{i + 1}/{len(dataloader)}]: '
                          f'Loss: {loss_meter / runcount:6.4f} '
                          f'Acc: {100 * acc_meter / runcount:6.2f} ({time.time() - start_time:.2f}s)', end='\r')

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount

        if self.verbose:
            print(f'{msg}: '
                  f'Loss: {loss_meter:6.4f} '
                  f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
            print()

        return {'loss': loss_meter, 'acc': acc_meter, 'time': time.time() - start_time}


class TrainerPrivate(object):
    def __init__(self, model, optimizer, scheduler, device):
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.tester = TesterPrivate(model, device)

    def train(self, e, dataloader, wm_dataloader=None):
        self.model.train()
        loss_meter = 0
        sign_loss_meter = 0
        public_acc_meter = 0
        private_acc_meter = 0

        if wm_dataloader is not None:
            iter_wm_dataloader = iter(wm_dataloader)
        else:
            iter_wm_dataloader = None

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            if iter_wm_dataloader is not None:
                try:
                    wm_data, wm_target = next(iter_wm_dataloader)
                except StopIteration:
                    iter_wm_dataloader = iter(wm_dataloader)
                    wm_data, wm_target = next(iter_wm_dataloader)

                wm_data = wm_data.to(self.device, non_blocking=True)
                wm_target = wm_target.to(self.device, non_blocking=True)

                data = torch.cat([data, wm_data], dim=0)
                target = torch.cat([target, wm_target], dim=0)

            self.optimizer.zero_grad()

            # passport sign loss reset
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            loss = torch.tensor(0.).to(self.device)
            sign_loss = torch.tensor(0.).to(self.device)

            # backprop to two graph at once
            for ind in range(2):
                pred = self.model(data, ind=ind)
                loss += F.cross_entropy(pred, target)

                if ind == 0:
                    public_acc_meter += accuracy(pred, target)[0].item()
                else:
                    private_acc_meter += accuracy(pred, target)[0].item()

            # sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    sign_loss += m.loss

            (loss + sign_loss).backward()
            self.optimizer.step()

            sign_loss_meter += sign_loss.item()
            loss_meter += loss.item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'Sign Loss: {sign_loss_meter / (i + 1):6.4f} '
                  f'Priv. Acc: {private_acc_meter / (i + 1):.4f} '
                  f'Publ. Acc: {public_acc_meter / (i + 1):.4f} '
                  f'({time.time() - start_time:.2f}s)', end='\r')

        print()

        loss_meter /= len(dataloader)
        public_acc_meter /= len(dataloader)
        private_acc_meter /= len(dataloader)

        sign_acc = torch.tensor(0.).to(self.device)
        count = 0

        for m in self.model.modules():
            if isinstance(m, SignLoss):
                sign_acc += m.acc
                count += 1

        if count != 0:
            sign_acc /= count

        if self.scheduler is not None:
            self.scheduler.step()

        return {'loss': loss_meter,
                'sign_loss': sign_loss_meter,
                'sign_acc': sign_acc.item(),
                'acc_public': public_acc_meter,
                'acc_private': private_acc_meter,
                'time': time.time() - start_time}

    def test(self, dataloader, msg='Testing Result'):
        self.model.eval()

        out = {}

        for i in range(2):
            key = 'public' if i == 0 else 'private'
            loss_meter = 0
            acc_meter = 0
            runcount = 0

            start_time = time.time()
            with torch.no_grad():
                for j, load in enumerate(dataloader):
                    data, target = load[:2]
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    pred = self.model(data, ind=i)
                    loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
                    pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    acc_meter += pred.eq(target.view_as(pred)).sum().item()
                    runcount += data.size(0)
                    print(f'{msg} [{i + 1}/{len(dataloader)}]: '
                          f'Loss: {loss_meter / runcount:6.4f} '
                          f'Acc: {acc_meter / runcount:6.2f} ({time.time() - start_time:.2f}s)', end='\r')

            loss_meter /= runcount
            acc_meter = 100 * acc_meter / runcount
            print(f'{msg} {key}: '
                  f'Loss: {loss_meter:6.4f} '
                  f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
            print()

            out.update({'loss_' + key: loss_meter, 'acc_' + key: acc_meter, 'time_' + key: time.time() - start_time})

        out['total_acc'] = (out['acc_public'] + out['acc_private']) / 2
        print(f'Total acc: {out["total_acc"]:.2f}')
        print()

        sign_out = self.tester.test_signature()
        for key in sign_out:
            out['s_' + key] = sign_out[key]

        return out
