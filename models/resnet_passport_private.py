import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d_private import PassportPrivateBlock


def get_convblock(passport_kwargs):
    def convblock_(*args, **kwargs):
        if passport_kwargs['flag']:
            return PassportPrivateBlock(*args, **kwargs, passport_kwargs=passport_kwargs)
        else:
            return ConvBlock(*args, **kwargs, bn=passport_kwargs['norm_type'])

    return convblock_


class BasicPrivateBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, passport_kwargs={}):
        super(BasicPrivateBlock, self).__init__()

        self.convbnrelu_1 = get_convblock(passport_kwargs['convbnrelu_1'])(in_planes, planes, 3, stride, 1)
        self.convbn_2 = get_convblock(passport_kwargs['convbn_2'])(planes, planes, 3, 1, 1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = get_convblock(passport_kwargs['shortcut'])(in_planes, self.expansion * planes, 1, stride, 0)

    def set_intermediate_keys(self, pretrained_block, x, y=None):
        if isinstance(self.convbnrelu_1, PassportPrivateBlock):
            self.convbnrelu_1.set_key(x, y)
        out_x = pretrained_block.convbnrelu_1(x)
        if y is not None:
            out_y = pretrained_block.convbnrelu_1(y)
        else:
            out_y = None

        if isinstance(self.convbn_2, PassportPrivateBlock):
            self.convbn_2.set_key(out_x, out_y)
        out_x = pretrained_block.convbn_2(out_x)
        if y is not None:
            out_y = pretrained_block.convbn_2(out_y)

        if not isinstance(self.shortcut, nn.Sequential):
            if isinstance(self.shortcut, PassportPrivateBlock):
                self.shortcut.set_key(x, y)

            shortcut_x = pretrained_block.shortcut(x)
            out_x = out_x + shortcut_x
            if y is not None:
                shortcut_y = pretrained_block.shortcut(y)
                out_y = out_y + shortcut_y
        else:
            out_x = out_x + x
            if y is not None:
                out_y = out_y + y

        out_x = F.relu(out_x)
        if y is not None:
            out_y = F.relu(out_y)

        return out_x, out_y

    def forward(self, x, force_passport=False, ind=0):
        if isinstance(self.convbnrelu_1, PassportPrivateBlock):
            out = self.convbnrelu_1(x, force_passport, ind)
        else:
            out = self.convbnrelu_1(x)

        if isinstance(self.convbn_2, PassportPrivateBlock):
            out = self.convbn_2(out, force_passport, ind)
        else:
            out = self.convbn_2(out)

        if not isinstance(self.shortcut, nn.Sequential):
            if isinstance(self.shortcut, PassportPrivateBlock):
                out = out + self.shortcut(x, force_passport, ind)
            else:
                out = out + self.shortcut(x)
        else:
            out = out + x
        out = F.relu(out)
        return out


class ResNetPrivate(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, passport_kwargs={}, pretrained=False, imagenet=False):
        super(ResNetPrivate, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks

        if num_classes == 1000 or imagenet:
            self.convbnrelu_1 = nn.Sequential(
                get_convblock(passport_kwargs['convbnrelu_1'])(3, 64, 7, 2, 3),  # 112
                nn.MaxPool2d(3, 2, 1),  # 56
            )
        else:
            self.convbnrelu_1 = get_convblock(passport_kwargs['convbnrelu_1'])(3, 64, 3, 1, 1)  # 32

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, passport_kwargs=passport_kwargs['layer1'])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, passport_kwargs=passport_kwargs['layer2'])
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, passport_kwargs=passport_kwargs['layer3'])
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, passport_kwargs=passport_kwargs['layer4'])
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        if num_classes == 1000 and pretrained:
            assert sum(num_blocks) == 8, 'only implemented for resnet18'
            layers = [self.convbnrelu_1[0].conv, self.convbnrelu_1[0].bn]
            for blocklayers in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for blocklayer in blocklayers:
                    b1 = blocklayer.convbnrelu_1
                    b2 = blocklayer.convbn_2
                    b3 = blocklayer.shortcut
                    layers += [b1.conv, b1.bn, b2.conv, b2.bn]
                    if not isinstance(b3, nn.Sequential):
                        layers += [b3.conv, b3.bn]
            layers += [self.linear]

            self._load_pretrained_from_torch(layers)

    def _load_pretrained_from_torch(self, layers):
        # load a pretrained alexnet from torchvision
        torchmodel = resnet18(True)
        torchlayers = [torchmodel.conv1, torchmodel.bn1]
        for torchblocklayers in [torchmodel.layer1, torchmodel.layer2, torchmodel.layer3, torchmodel.layer4]:
            for blocklayer in torchblocklayers:
                torchlayers += [blocklayer.conv1, blocklayer.bn1, blocklayer.conv2, blocklayer.bn2]
                if blocklayer.downsample is not None:
                    torchlayers += [blocklayer.downsample[0], blocklayer.downsample[1]]

        for torchlayer, layer in zip(torchlayers, layers):
            assert torchlayer.weight.size() == layer.weight.size(), 'must be same'
            layer.load_state_dict(torchlayer.state_dict())

    def _make_layer(self, block, planes, num_blocks, stride, passport_kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes, stride, passport_kwargs[str(i)]))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_intermediate_keys(self, pretrained_model, x, y=None):
        with torch.no_grad():
            if isinstance(self.convbnrelu_1, PassportPrivateBlock):
                self.convbnrelu_1.set_key(x, y)

            x = pretrained_model.convbnrelu_1(x)
            if y is not None:
                y = pretrained_model.convbnrelu_1(y)

            for self_block, pretrained_block in zip(self.layer1, pretrained_model.layer1):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)
            for self_block, pretrained_block in zip(self.layer2, pretrained_model.layer2):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)
            for self_block, pretrained_block in zip(self.layer3, pretrained_model.layer3):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)
            for self_block, pretrained_block in zip(self.layer4, pretrained_model.layer4):
                x, y = self_block.set_intermediate_keys(pretrained_block, x, y)

    def forward(self, x, force_passport=False, ind=0):
        if isinstance(self.convbnrelu_1, PassportPrivateBlock):
            out = self.convbnrelu_1(x, force_passport, ind)
        else:
            out = self.convbnrelu_1(x)

        for block in self.layer1:
            out = block(out, force_passport, ind)
        for block in self.layer2:
            out = block(out, force_passport, ind)
        for block in self.layer3:
            out = block(out, force_passport, ind)
        for block in self.layer4:
            out = block(out, force_passport, ind)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet18Private(**model_kwargs):
    return ResNetPrivate(BasicPrivateBlock, [2, 2, 2, 2], **model_kwargs)


if __name__ == '__main__':
    import json
    from pprint import pprint
    from experiments.trainer_private import TesterPrivate

    passport_settings = json.load(open('../passport_configs/resnet18_passport.json'))
    passport_kwargs = {}

    for layer_key in passport_settings:
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:
                passport_kwargs[layer_key][i] = {}
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]:
                    flag = passport_settings[layer_key][i][module_key]
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True
                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag,
                        'norm_type': 'gn',
                        'key_type': 'random',
                        'sign_loss': 1
                    }
                    if b is not None:
                        passport_kwargs[layer_key][i][module_key]['b'] = b

        else:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            passport_kwargs[layer_key] = {
                'flag': flag,
                'norm_type': 'gn',
                'key_type': 'random',
                'sign_loss': 1
            }
            if b is not None:
                passport_kwargs[layer_key][i][module_key]['b'] = b

    pprint(passport_kwargs)
    key_model = ResNet18Private(passport_kwargs=passport_kwargs)
    for name in key_model.named_modules():
        print(name[0], name[1].__class__.__name__)

    key_model.set_intermediate_keys(ResNet18Private(passport_kwargs=passport_kwargs),
                                    torch.randn(1, 3, 32, 32),
                                    torch.randn(1, 3, 32, 32))

    key_model(torch.randn(1, 3, 32, 32), ind=0)
    key_model(torch.randn(1, 3, 32, 32), ind=1)

    TesterPrivate(key_model, torch.device('cpu')).test_signature()
