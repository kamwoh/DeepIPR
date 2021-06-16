import argparse
from pprint import pprint

import torch

from experiments.classification import ClassificationExperiment

torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='alexnet', choices=['alexnet', 'resnet', 'resnet9'],
                        help='architecture (default: alexnet)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10',
                                                                 'cifar100',
                                                                 'caltech-101',
                                                                 'caltech-256',
                                                                 'imagenet1000'],
                        help='training dataset (default: cifar10)')
    parser.add_argument('--norm-type', default='bn', choices=['bn', 'gn', 'in', 'none'],
                        help='norm type (default: bn)')

    # passport argument
    parser.add_argument('--key-type', choices=['random', 'image', 'shuffle'], default='shuffle',
                        help='passport key type (default: shuffle)')
    parser.add_argument('--sign-loss', type=float, default=0.1,
                        help='sign loss to avoid scale not trainable (default: 0.1)')
    parser.add_argument('--use-trigger-as-passport', action='store_true', default=False,
                        help='use trigger data as passport')

    parser.add_argument('--train-passport', action='store_true', default=False,
                        help='train passport')
    parser.add_argument('--train-backdoor', action='store_true', default=False,
                        help='train backdoor, adding backdoor images for blackbox detection')
    parser.add_argument('--train-private', action='store_true', default=False,
                        help='train private')

    # paths
    parser.add_argument('--pretrained-path',
                        help='load pretrained path')
    parser.add_argument('--lr-config', default='lr_configs/default.json',
                        help='lr config json file')
    parser.add_argument('--passport-config', default='passport_configs/alexnet_passport.json',
                        help='should be same json file as arch')

    # misc
    parser.add_argument('--save-interval', type=int, default=0,
                        help='save model interval')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='for evaluation')
    parser.add_argument('--exp-id', type=int, default=1,
                        help='experiment id')
    parser.add_argument('--tag',
                        help='tag')

    # transfer learning
    parser.add_argument('--transfer-learning', action='store_true', default=False,
                        help='turn on transfer learning')
    parser.add_argument('--tl-dataset', default='cifar100', choices=['cifar10',
                                                                     'cifar100',
                                                                     'caltech-101',
                                                                     'caltech-256',
                                                                     'imagenet1000'],
                        help='transfer learning dataset (default: cifar100)')
    parser.add_argument('--tl-scheme', default='rtal', choices=['rtal',
                                                                'ftal'],
                        help='transfer learning scheme (default: rtal)')

    args = parser.parse_args()

    pprint(vars(args))

    exp = ClassificationExperiment(vars(args))

    if exp.is_tl:
        exp.transfer_learning()
    else:
        exp.training()

    print('Training done at', exp.logdir)


if __name__ == '__main__':
    main()
