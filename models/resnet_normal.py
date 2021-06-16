import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from models.layers.conv2d import ConvBlock


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(BasicBlock, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1, bn=norm_type, relu=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes,
                                      1, stride, 0, bn=norm_type, relu=True)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(Bottleneck, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 1, 1, 0, bn=norm_type, relu=True)
        self.convbnrelu_2 = ConvBlock(planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_3 = ConvBlock(planes, self.expansion * planes, 1, 1, 0, bn=norm_type, relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes, 1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbnrelu_2(out)
        out = self.convbn_3(out) + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_type='bn', pretrained=False, imagenet=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.norm_type = norm_type

        if num_classes == 1000 or imagenet:
            self.convbnrelu_1 = nn.Sequential(
                ConvBlock(3, 64, 7, 2, 3, bn=norm_type, relu=True),  # 112
                nn.MaxPool2d(3, 2, 1),  # 56
            )
        else:
            self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1, bn=norm_type, relu=True)  # 32
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 32/ 56
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16/ 28
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8/ 14
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4/ 7
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet9(**model_kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **model_kwargs)


def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)


def ResNet34(**model_kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **model_kwargs)


def ResNet50(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **model_kwargs)


def ResNet101(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **model_kwargs)


def ResNet152(**model_kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **model_kwargs)


if __name__ == '__main__':
    key_model = ResNet18(num_classes=1000, pretrained=False)
    print(key_model.convbnrelu_1(torch.randn(1, 3, 224, 224)).size())
    # for name in key_model.named_modules():
    #     print(name[0])
    #
    # model = ResNet50(num_classes=10)
    #
    # x = torch.randn(1, 3, 32, 32)
    # y = model(x)
    #
    # print(y)
