import torch.nn as nn

from models.layers.conv2d import ConvBlock


class AlexNetNormal(nn.Module):

    def __init__(self, in_channels, num_classes, norm_type='bn'):
        super(AlexNetNormal, self).__init__()

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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
