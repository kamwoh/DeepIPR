from torch import nn


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
    try:
        passport_model.load_state_dict(model.state_dict())
    except:
        if arch == 'alexnet':
            for clone_m, self_m in zip(clone_model.features, passport_model.features):
                try:
                    self_m.load_state_dict(clone_m.state_dict())
                except:
                    self_m.load_state_dict(clone_m.state_dict(), False)

            if isinstance(passport_model.classifier, nn.Sequential):
                for passport_m, normal_m in passport_model.classifier:
                    pass  # todo: continue implement
        else:
            passport_settings = self.passport_config
            for l_key in passport_settings:  # layer
                if isinstance(passport_settings[l_key], dict):
                    for i in passport_settings[l_key]:  # sequential
                        for m_key in passport_settings[l_key][i]:  # convblock
                            clone_m = clone_model.__getattr__(l_key)[int(i)].__getattr__(m_key)
                            self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)

                            try:
                                self_m.load_state_dict(clone_m.state_dict())
                            except:
                                self_m.load_state_dict(clone_m.state_dict(), False)
                else:
                    clone_m = clone_model.__getattr__(l_key)
                    self_m = self.model.__getattr__(l_key)

                    try:
                        self_m.load_state_dict(clone_m.state_dict())
                    except:
                        self_m.load_state_dict(clone_m.state_dict(), False)


def load_passport_model_to_normal_model(arch, plkeys, passport_model, model):
    model.load_state_dict(passport_model.state_dict(), strict=False)

    if arch == 'alexnet':
        for fidx in plkeys:
            fidx = int(fidx)
            model.features[fidx].bn.weight.data.copy_(passport_model.features[fidx].get_scale().view(-1))
            model.features[fidx].bn.bias.data.copy_(passport_model.features[fidx].get_bias().view(-1))

            model.features[fidx].bn.weight.requires_grad_(True)
            model.features[fidx].bn.bias.requires_grad_(True)
    else:
        for fidx in plkeys:
            layer_key, i, module_key = fidx.split('.')

            def get_layer(m):
                return m.__getattr__(layer_key)[int(i)].__getattr__(module_key)

            convblock = get_layer(model)
            passblock = get_layer(passport_model)
            convblock.bn.weight.data.copy_(passblock.get_scale().view(-1))
            convblock.bn.bias.data.copy_(passblock.get_bias().view(-1))

            convblock.bn.weight.requires_grad_(True)
            convblock.bn.bias.requires_grad_(True)
