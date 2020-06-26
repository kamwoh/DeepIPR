import torch
import torch.nn as nn
from torchvision.models import alexnet

from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d import PassportBlock


class AlexNetPassport(nn.Module):

    def __init__(self, in_channels, num_classes, passport_kwargs, pretrained=False):
        super(AlexNetPassport, self).__init__()

        maxpoolidx = [1, 3, 7]

        layers = []
        params = []

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
                    layers.append(PassportBlock(inp, oups[layeridx], k, s, p, passport_kwargs[str(layeridx)]))
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

        if num_classes == 1000 and pretrained:
            assert normtype == 'none', 'torchvision pretrained alexnet does not have normalization layer'
            layers = []
            for layer in self.features:
                if isinstance(layer, (ConvBlock, PassportBlock)):
                    layers.append(layer)

            for layer in self.classifier:
                if isinstance(layer, nn.Linear):
                    layers.append(layer)

            self._load_pretrained_from_torch(layers)

    def _load_pretrained_from_torch(self, layers):
        torchmodel = alexnet(True)

        torchlayers = []
        for layer in torchmodel.features:
            if isinstance(layer, nn.Conv2d):
                torchlayers.append(layer)
        for layer in torchmodel.classifier:
            if isinstance(layer, nn.Linear):
                torchlayers.append(layer)

        for torchlayer, layer in zip(torchlayers, layers):
            if isinstance(layer, ConvBlock):
                layer.conv.weight.data.copy_(torchlayer.weight.data)
                layer.conv.bias.data.copy_(torchlayer.bias.data)

            if isinstance(layer, nn.Linear):
                layer.weight.data.copy_(torchlayer.weight.data)
                layer.bias.data.copy_(torchlayer.bias.data)

    def set_intermediate_keys(self, pretrained_model, x, y=None):
        with torch.no_grad():
            for pretrained_layer, self_layer in zip(pretrained_model.features, self.features):
                if isinstance(self_layer, PassportBlock):
                    self_layer.set_key(x, y)

                x = pretrained_layer(x)
                if y is not None:
                    y = pretrained_layer(y)

    def forward(self, x, force_passport=False):
        for m in self.features:
            if isinstance(m, PassportBlock):
                x = m(x, force_passport)
            else:
                x = m(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import json
    from pprint import pprint
    from experiments.utils import construct_passport_kwargs_from_dict

    passport_kwargs = construct_passport_kwargs_from_dict({'norm_type': 'none',
                                                           'key_type': 'random',
                                                           'sl_ratio': 0.1,
                                                           'passport_config': json.load(
                                                               open('../passport_configs/alexnet_passport.json'))})
    pprint(AlexNetPassport(3, 1000, passport_kwargs, True))
