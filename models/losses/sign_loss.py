import torch
import torch.nn as nn
import torch.nn.functional as F


class SignLoss(nn.Module):
    def __init__(self, alpha, b=None):
        super(SignLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer('b', b)
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

    def set_b(self, b):
        self.b.copy_(b)

    def get_acc(self):
        if self.scale_cache is not None:
            acc = (torch.sign(self.b.view(-1)) == torch.sign(self.scale_cache.view(-1))).float().mean()
            return acc
        else:
            raise Exception('scale_cache is None')

    def get_loss(self):
        if self.scale_cache is not None:
            loss = (self.alpha * F.relu(-self.b.view(-1) * self.scale_cache.view(-1) + 0.1)).sum()
            return loss
        else:
            raise Exception('scale_cache is None')

    def add(self, scale):
        self.scale_cache = scale

        # hinge loss concept
        # f(x) = max(x + 0.5, 0)*-b
        # f(x) = max(x + 0.5, 0) if b = -1
        # f(x) = max(0.5 - x, 0) if b = 1

        # case b = -1
        # - (-1) * 1 = 1 === bad
        # - (-1) * -1 = -1 -> 0 === good

        # - (-1) * 0.6 + 0.5 = 1.1 === bad
        # - (-1) * -0.6 + 0.5 = -0.1 -> 0 === good

        # case b = 1
        # - (1) * -1 = 1 -> 1 === bad
        # - (1) * 1 = -1 -> 0 === good

        # let it has minimum of 0.1
        self.loss += self.get_loss()
        self.loss += (0.00001 * scale.view(-1).pow(2).sum())  # to regularize the scale not to be so large
        self.acc += self.get_acc()

    def reset(self):
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

    # def to(self, *args, **kwargs):
    #     self.loss = self.loss.to(args[0])
    #     return super().to(*args, **kwargs)
