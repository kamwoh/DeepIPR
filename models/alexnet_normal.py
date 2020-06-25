import torch.nn as nn

from models.layers.conv2d import ConvBlock


class AlexNetNormal(nn.Module):

    def __init__(self, in_channels, num_classes, norm_type='bn'):
        super(AlexNetNormal, self).__init__()

        if num_classes == 1000:  # imagenet1000
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
