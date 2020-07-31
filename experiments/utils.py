from torch import nn

from models.layers.passportconv2d import PassportBlock


def construct_passport_kwargs(self, need_index=False):
    passport_settings = self.passport_config
    passport_kwargs = {}
    keys = []

    for layer_key in passport_settings:
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]:
                    flag = passport_settings[layer_key][i][module_key]
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True
                    if flag:
                        keys.append(f'{layer_key}.{i}.{module_key}')
                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag,
                        'norm_type': self.norm_type,
                        'key_type': self.key_type,
                        'sign_loss': self.sl_ratio
                    }
                    if b is not None:
                        passport_kwargs[layer_key][i][module_key]['b'] = b
        else:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            if flag:
                keys.append(layer_key)
            passport_kwargs[layer_key] = {
                'flag': flag,
                'norm_type': self.norm_type,
                'key_type': self.key_type,
                'sign_loss': self.sl_ratio
            }
            if b is not None:
                passport_kwargs[layer_key]['b'] = b

    if need_index:
        return passport_kwargs, keys

    return passport_kwargs


def construct_passport_kwargs_from_dict(self, need_index=False):
    passport_settings = self['passport_config']
    passport_kwargs = {}
    keys = []

    for layer_key in passport_settings:
        if isinstance(passport_settings[layer_key], dict):
            passport_kwargs[layer_key] = {}
            for i in passport_settings[layer_key]:
                passport_kwargs[layer_key][i] = {}
                for module_key in passport_settings[layer_key][i]:
                    flag = passport_settings[layer_key][i][module_key]
                    b = flag if isinstance(flag, str) else None
                    if b is not None:
                        flag = True
                    if flag:
                        keys.append(f'{layer_key}.{i}.{module_key}')
                    passport_kwargs[layer_key][i][module_key] = {
                        'flag': flag,
                        'norm_type': self['norm_type'],
                        'key_type': self['key_type'],
                        'sign_loss': self['sl_ratio']
                    }
                    if b is not None:
                        passport_kwargs[layer_key][i][module_key]['b'] = b
        else:
            flag = passport_settings[layer_key]
            b = flag if isinstance(flag, str) else None
            if b is not None:
                flag = True
            if flag:
                keys.append(layer_key)
            passport_kwargs[layer_key] = {
                'flag': flag,
                'norm_type': self['norm_type'],
                'key_type': self['key_type'],
                'sign_loss': self['sl_ratio']
            }
            if b is not None:
                passport_kwargs[layer_key]['b'] = b

    if need_index:
        return passport_kwargs, keys

    return passport_kwargs


def load_normal_model_to_passport_model(arch, plkeys, passport_model, model):
    for m in passport_model.modules():  # detect signature on scale sign
        if isinstance(m, PassportBlock):
            m.init_scale(True)
            m.init_bias(True)

    if arch == 'alexnet':
        # load features weight
        passport_model.features.load_state_dict(model.features.state_dict(), False)

        # load scale/bias
        for fidx in plkeys:
            fidx = int(fidx)

            # for private, using public scale/bias, therefore no need force_passport
            passport_model.features[fidx].scale.data.copy_(model.features[fidx].bn.weight.data)
            passport_model.features[fidx].bias.data.copy_(model.features[fidx].bn.bias.data)

        # load classifier except last one
        if isinstance(passport_model.classifier, nn.Sequential):
            for i, (passport_layer, layer) in enumerate(zip(passport_model.classifier, model.classifier)):
                if i != len(passport_model.classifier) - 1:  # do not load last one
                    passport_layer.load_state_dict(layer.state_dict())
    else:
        # for l_key in passport_settings:  # layer
        #     if isinstance(passport_settings[l_key], dict):
        #         for i in passport_settings[l_key]:  # sequential
        #             for m_key in passport_settings[l_key][i]:  # convblock
        #                 layer = model.__getattr__(l_key)[int(i)].__getattr__(m_key)
        #                 passport_layer = passport_model.__getattr__(l_key)[int(i)].__getattr__(m_key)
        #                 passport_layer.load_state_dict(layer.state_dict(), strict=False)
        #     else:
        #         layer = model.__getattr__(l_key)
        #         passport_layer = passport_model.__getattr__(l_key)
        #         passport_layer.load_state_dict(layer.state_dict(), strict=False)

        feature_pairs = [
            (model.convbnrelu_1, passport_model.convbnrelu_1),
            (model.layer1, passport_model.layer1),
            (model.layer2, passport_model.layer2),
            (model.layer3, passport_model.layer3),
            (model.layer4, passport_model.layer4)
        ]

        # load feature weights
        for layer, passport_layer in feature_pairs:
            passport_layer.load_state_dict(layer.state_dict(), strict=False)

            # load scale/bias
            for fidx in plkeys:
                layer_key, i, module_key = fidx.split('.')

                def get_layer(m):
                    return m.__getattr__(layer_key)[int(i)].__getattr__(module_key)

                convblock = get_layer(model)
                passblock = get_layer(passport_model)

                # for private, using public scale/bias, therefore no need force_passport
                passblock.scale.data.copy_(convblock.bn.weight.data)
                passblock.bias.data.copy_(convblock.bn.bias.data)

        # no need to load classifer as it has only one layer


def load_passport_model_to_normal_model(arch, plkeys, passport_model, model):
    if arch == 'alexnet':
        # load features from passport model.features
        model.features.load_state_dict(passport_model.features.state_dict(), strict=False)

        # load scale/bias
        for fidx in plkeys:
            fidx = int(fidx)

            # for private, using public scale/bias, therefore no need force_passport
            model.features[fidx].bn.weight.data.copy_(passport_model.features[fidx].get_scale().view(-1))
            model.features[fidx].bn.bias.data.copy_(passport_model.features[fidx].get_bias().view(-1))

            model.features[fidx].bn.weight.requires_grad_(True)
            model.features[fidx].bn.bias.requires_grad_(True)

        # load classifier except last one
        if isinstance(passport_model.classifier, nn.Sequential):
            for i, (passport_layer, layer) in enumerate(zip(passport_model.classifier, model.classifier)):
                if i != len(passport_model.classifier) - 1:  # do not load last one
                    layer.load_state_dict(passport_layer.state_dict())
    else:
        feature_pairs = [
            (model.convbnrelu_1, passport_model.convbnrelu_1),
            (model.layer1, passport_model.layer1),
            (model.layer2, passport_model.layer2),
            (model.layer3, passport_model.layer3),
            (model.layer4, passport_model.layer4)
        ]

        # load feature weights
        for layer, passport_layer in feature_pairs:
            layer.load_state_dict(passport_layer.state_dict(), strict=False)

        # load scale/bias
        for fidx in plkeys:
            layer_key, i, module_key = fidx.split('.')

            def get_layer(m):
                return m.__getattr__(layer_key)[int(i)].__getattr__(module_key)

            convblock = get_layer(model)
            passblock = get_layer(passport_model)

            # for private, using public scale/bias, therefore no need force_passport
            convblock.bn.weight.data.copy_(passblock.get_scale().view(-1))
            convblock.bn.bias.data.copy_(passblock.get_bias().view(-1))

        # no need to load classifer as it has only one layer
