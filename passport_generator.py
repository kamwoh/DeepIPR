import random

import torch


def get_key(dataset_loader, n=32):
    dataset = dataset_loader.dataset

    indices = random.sample(range(len(dataset)), n)

    imgs = []

    for i in indices:
        img, target = dataset[i]
        imgs.append(img.unsqueeze(0))

    return torch.cat(imgs, dim=0), indices


def get_intermediate_key(input_key, intermediate_key_name, pretrained_model):
    x = input_key

    with torch.no_grad():
        for i, m in enumerate(pretrained_model.features):
            if 'features.' + str(i) == intermediate_key_name:
                return x
            x = m(x)


def set_key(pretrained_model, target_model,
            key_x, key_y, ind=None):
    print('Setting keys')
    if len(key_x.size()) == 3:
        key_x = key_x.unsqueeze(0)
    if key_y is not None and len(key_y.size()) == 3:
        key_y = key_y.unsqueeze(0)

    print('Key size', key_x.size())
    if ind is not None:
        target_model.set_intermediate_keys(pretrained_model, key_x, key_y, ind)
    else:
        target_model.set_intermediate_keys(pretrained_model, key_x, key_y)
    print('Key is set!')
