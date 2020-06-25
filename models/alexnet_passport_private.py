import torch
import torch.nn as nn

from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d_private import PassportPrivateBlock


class AlexNetPassportPrivate(nn.Module):

    def __init__(self, in_channels, num_classes, passport_kwargs):
        super().__init__()

        maxpoolidx = [1, 3, 7]

        layers = []

        inp = in_channels
        oups = {
            0: 64,
            2: 192,
            4: 384,
            5: 256,
            6: 256
        }
        kp = {
            0: (5, 2) if num_classes != 1000 else (11, 4, 2),
            2: (5, 2),
            4: (3, 1),
            5: (3, 1),
            6: (3, 1)
        }

        for layeridx in range(8):
            if layeridx in maxpoolidx:
                ks = 2 if num_classes != 1000 else 3
                layers.append(nn.MaxPool2d(ks, 2))
            else:
                if len(kp[layeridx]) == 2:
                    k, p = kp[layeridx]
                    s = 1
                else:
                    k, s, p = kp[layeridx]
                normtype = passport_kwargs[str(layeridx)]['norm_type']
                if passport_kwargs[str(layeridx)]['flag']:
                    layers.append(PassportPrivateBlock(inp, oups[layeridx], k, s, p, passport_kwargs[str(layeridx)]))
                else:
                    layers.append(ConvBlock(inp, oups[layeridx], k, s, p, normtype))

                inp = oups[layeridx]

        if num_classes == 1000:
            layers.append(nn.AdaptiveAvgPool2d((6, 6)))

        self.features = nn.Sequential(*layers)

        if num_classes == 1000:
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
            self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def set_intermediate_keys(self, pretrained_model, x, y=None):
        with torch.no_grad():
            for pretrained_layer, self_layer in zip(pretrained_model.features, self.features):
                if isinstance(self_layer, PassportPrivateBlock):
                    self_layer.set_key(x, y)

                x = pretrained_layer(x)
                if y is not None:
                    y = pretrained_layer(y)

    def forward(self, x, force_passport=False, ind=0):
        for m in self.features:
            if isinstance(m, PassportPrivateBlock):
                x = m(x, force_passport, ind)
            else:
                x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
