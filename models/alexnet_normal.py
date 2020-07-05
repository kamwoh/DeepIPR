import torch.nn as nn
from torchvision.models import alexnet

from models.layers.conv2d import ConvBlock


class AlexNetNormal(nn.Module):
    def __init__(self, in_channels, num_classes, norm_type='bn', pretrained=False):
        super(AlexNetNormal, self).__init__()

        params = []

        if num_classes == 1000:  # imagenet1000
            if pretrained:
                norm_type = 'none'
            self.features = nn.Sequential(
                ConvBlock(3, 64, 11, 4, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                ConvBlock(192, 384, 3, 1, 1, bn=norm_type),
                ConvBlock(384, 256, 3, 1, 1, bn=norm_type),
                ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6, 6))
            )

            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

            for layer in self.features:
                if isinstance(layer, ConvBlock):
                    params.append(layer.conv.weight)
                    params.append(layer.conv.bias)

            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    params.append(layer.weight)
                    params.append(layer.bias)

            if pretrained:
                self._load_pretrained_from_torch(params)
        else:
            self.features = nn.Sequential(
                ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
                ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
                ConvBlock(192, 384, bn=norm_type),
                ConvBlock(384, 256, bn=norm_type),
                ConvBlock(256, 256, bn=norm_type),
                nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
            )

            self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def _load_pretrained_from_torch(self, params):
        # load a pretrained alexnet from torchvision
        torchmodel = alexnet(True)
        torchparams = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchparams.append(layer.weight)
                torchparams.append(layer.bias)

        for torchparam, param in zip(torchparams, params):
            assert torchparam.size() == param.size(), 'size not match'
            param.data.copy_(torchparam.data)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # test load pretrained
    AlexNetNormal(3, 1000, 'none', True)
