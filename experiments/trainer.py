import time

import torch
import torch.nn.functional as F

from models.losses.sign_loss import SignLoss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


class Tester(object):
    def __init__(self, model, device, verbose=True):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test(self, dataloader, msg='Testing Result', compare=[]):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()  # sum up batch loss
                pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
                compare.append((pred, target))
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount

        if self.verbose:
            print(f'{msg}: '
                  f'Loss: {loss_meter:6.4f} '
                  f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
            print()

        return {'loss': loss_meter, 'acc': acc_meter, 'time': time.time() - start_time}


class Trainer(object):
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def finetune_backdoor(self, e, wm_dataloader, backdoor_input=None):
        self.model.eval()  # freeze running mean
        loss_meter = 0
        acc_meter = 0

        start_time = time.time()
        for i, load in enumerate(wm_dataloader):
            if backdoor_input is None:
                data, target = load
                data = data.to(self.device)
                target = target.to(self.device)
            else:
                data, target, index = load
                data = data.to(self.device)
                target = target.to(self.device)
                data = backdoor_input(data, index)

            self.optimizer.zero_grad()

            pred = self.model(data)
            loss = F.cross_entropy(pred, target)
            loss.backward()

            self.optimizer.step()

            loss_meter += loss.item()
            acc_meter += accuracy(pred, target)[0].item()

            print(f'Epoch {e:3d} [{i:4d}/{len(wm_dataloader):4d}] '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'Acc: {acc_meter / (i + 1):.4f} ({time.time() - start_time:.2f}s)', end='\r')

        loss_meter /= len(wm_dataloader)
        acc_meter /= len(wm_dataloader)
        print()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}

    def train(self, e, dataloader, wm_dataloader=None):
        self.model.train()
        loss_meter = 0
        acc_meter = 0

        if self.scheduler is not None:
            self.scheduler.step()

        if wm_dataloader is not None:
            iter_wm_dataloader = iter(wm_dataloader)
        else:
            iter_wm_dataloader = None

        start_time = time.time()
        for i, (data, target) in enumerate(dataloader):
            data = data.to(self.device)
            target = target.to(self.device)

            if iter_wm_dataloader is not None:
                try:
                    wm_data, wm_target = next(iter_wm_dataloader)
                except StopIteration:
                    iter_wm_dataloader = iter(wm_dataloader)
                    wm_data, wm_target = next(iter_wm_dataloader)

                wm_data = wm_data.to(self.device)
                wm_target = wm_target.to(self.device)

                data = torch.cat([data, wm_data], dim=0)
                target = torch.cat([target, wm_target], dim=0)

            self.optimizer.zero_grad()

            # reset sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    m.reset()

            pred = self.model(data)
            loss = F.cross_entropy(pred, target)

            # add up sign loss
            for m in self.model.modules():
                if isinstance(m, SignLoss):
                    loss += m.loss

            loss.backward()
            self.optimizer.step()

            loss_meter += loss.item()
            acc_meter += accuracy(pred, target)[0].item()

            print(f'Epoch {e:3d} [{i:4d}/{len(dataloader):4d}] '
                  f'Loss: {loss_meter / (i + 1):6.4f} '
                  f'Acc: {acc_meter / (i + 1):.4f} ({time.time() - start_time:.2f}s)', end='\r')

        print()

        loss_meter /= len(dataloader)
        acc_meter /= len(dataloader)

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}

    def test(self, dataloader, msg='Testing Result'):
        self.model.eval()
        loss_meter = 0
        acc_meter = 0
        runcount = 0

        start_time = time.time()
        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]

                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0)

        loss_meter /= runcount
        acc_meter = 100 * acc_meter / runcount
        print(f'{msg}: '
              f'Loss: {loss_meter:6.4f} '
              f'Acc: {acc_meter:6.2f} ({time.time() - start_time:.2f}s)')
        print()

        return {'loss': loss_meter,
                'acc': acc_meter,
                'time': time.time() - start_time}
