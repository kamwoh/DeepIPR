import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from models.losses.sign_loss import SignLoss


class PassportBlock(nn.Module):
    def __init__(self, i, o, ks=3, s=1, pd=1, passport_kwargs={}):
        super(PassportBlock, self).__init__()

        if passport_kwargs == {}:
            print('Warning, passport_kwargs is empty')

        self.conv = nn.Conv2d(i, o, ks, s, pd, bias=False)

        self.key_type = passport_kwargs.get('key_type', 'random')
        self.weight = self.conv.weight

        self.alpha = passport_kwargs.get('sign_loss', 1)

        b = passport_kwargs.get('b', torch.sign(torch.rand(o) - 0.5))  # bit information to store
        if isinstance(b, int):
            b = torch.ones(o) * b
        if isinstance(b, str):
            if len(b) * 8 > o:
                raise Exception('Too much bit information')
            bsign = torch.sign(torch.rand(o) - 0.5)
            bitstring = ''.join([format(ord(c), 'b').zfill(8) for c in b])

            for i, c in enumerate(bitstring):
                if c == '0':
                    bsign[i] = -1
                else:
                    bsign[i] = 1

            b = bsign
        self.register_buffer('b', b)

        self.requires_reset_key = False

        if self.alpha != 0:
            self.sign_loss = SignLoss(self.alpha, self.b)
        else:
            self.sign_loss = None

        self.register_buffer('key', None)
        self.register_buffer('skey', None)

        self.init_scale()
        self.init_bias()

        norm_type = passport_kwargs.get('norm_type', 'bn')
        if norm_type == 'bn':
            self.bn = nn.BatchNorm2d(o, affine=False)
        elif norm_type == 'gn':
            self.bn = nn.GroupNorm(o // 16, o, affine=False)
        elif norm_type == 'in':
            self.bn = nn.InstanceNorm2d(o, affine=False)
        else:
            self.bn = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)

        self.reset_parameters()

    def init_bias(self, force_init=False):
        if force_init:
            self.bias = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.zeros_(self.bias)
        else:
            self.bias = None

    def init_scale(self, force_init=False):
        if force_init:
            self.scale = nn.Parameter(torch.Tensor(self.conv.out_channels).to(self.weight.device))
            init.ones_(self.scale)
        else:
            self.scale = None

    def reset_parameters(self):
        init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def passport_selection(self, passport_candidates):
        b, c, h, w = passport_candidates.size()

        if c == 3:  # input channel
            randb = random.randint(0, b - 1)
            return passport_candidates[randb].unsqueeze(0)

        passport_candidates = passport_candidates.view(b * c, h, w)
        full = False
        flag = [False for _ in range(b * c)]
        channel = c
        passportcount = 0
        bcount = 0
        passport = []

        while not full:
            if bcount >= b:
                bcount = 0

            randc = bcount * channel + random.randint(0, channel - 1)
            while flag[randc]:
                randc = bcount * channel + random.randint(0, channel - 1)
            flag[randc] = True

            passport.append(passport_candidates[randc].unsqueeze(0).unsqueeze(0))

            passportcount += 1
            bcount += 1

            if passportcount >= channel:
                full = True

        passport = torch.cat(passport, dim=1)
        return passport

    def set_key(self, x, y=None):
        n = int(x.size(0))

        if n != 1:
            x = self.passport_selection(x)
            if y is not None:
                y = self.passport_selection(y)

        # assert x.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('key', x)

        # assert y is not None and y.size(0) == 1, 'only batch size of 1 for key'
        self.register_buffer('skey', y)

    def get_scale_key(self):
        return self.skey

    def get_scale(self, force_passport=False):
        if self.scale is not None and not force_passport:
            return self.scale.view(1, -1, 1, 1)
        else:
            skey = self.skey

            scalekey = self.conv(skey)
            b = scalekey.size(0)
            c = scalekey.size(1)
            scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
            scale = scale.mean(dim=0).view(1, c, 1, 1)

            if self.sign_loss is not None:
                self.sign_loss.reset()
                self.sign_loss.add(scale)

            return scale

    def get_bias_key(self):
        return self.key

    def get_bias(self, force_passport=False):
        if self.bias is not None and not force_passport:
            return self.bias.view(1, -1, 1, 1)
        else:
            key = self.key

            biaskey = self.conv(key)  # key batch always 1
            b = biaskey.size(0)
            c = biaskey.size(1)
            bias = biaskey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
            bias = bias.mean(dim=0).view(1, c, 1, 1)

            return bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        keyname = prefix + 'key'
        skeyname = prefix + 'skey'

        if keyname in state_dict:
            self.register_buffer('key', torch.randn(*state_dict[keyname].size()))
        if skeyname in state_dict:
            self.register_buffer('skey', torch.randn(*state_dict[skeyname].size()))

        scalename = prefix + 'scale'
        biasname = prefix + 'bias'
        if scalename in state_dict:
            self.scale = nn.Parameter(torch.randn(*state_dict[scalename].size()))

        if biasname in state_dict:
            self.bias = nn.Parameter(torch.randn(*state_dict[biasname].size()))

        super(PassportBlock, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                                         missing_keys, unexpected_keys, error_msgs)

    def generate_key(self, *shape):
        global key_type

        newshape = list(shape)
        newshape[0] = 1

        min = -1.0
        max = 1.0
        key = np.random.uniform(min, max, newshape)
        return key

    def forward(self, x, force_passport=False):
        if (self.key is None and self.key_type == 'random') or self.requires_reset_key:
            self.set_key(torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device),
                         torch.tensor(self.generate_key(*x.size()),
                                      dtype=x.dtype,
                                      device=x.device))

        x = self.conv(x)
        x = self.bn(x)
        x = self.get_scale(force_passport) * x + self.get_bias(force_passport)
        x = self.relu(x)
        return x
