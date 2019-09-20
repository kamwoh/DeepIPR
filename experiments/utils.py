def construct_passport_kwargs(self):
    passport_settings = self.passport_config
    passport_kwargs = {}

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
            passport_kwargs[layer_key] = {
                'flag': flag,
                'norm_type': self.norm_type,
                'key_type': self.key_type,
                'sign_loss': self.sl_ratio
            }
            if b is not None:
                passport_kwargs[layer_key]['b'] = b

    return passport_kwargs


def construct_passport_kwargs_from_dict(self):
    passport_settings = self['passport_config']
    passport_kwargs = {}

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
            passport_kwargs[layer_key] = {
                'flag': flag,
                'norm_type': self['norm_type'],
                'key_type': self['key_type'],
                'sign_loss': self['sl_ratio']
            }
            if b is not None:
                passport_kwargs[layer_key]['b'] = b

    return passport_kwargs
