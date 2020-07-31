import os
from pprint import pprint

import torch
import torch.optim as optim
from torch import nn

import passport_generator
from dataset import prepare_dataset, prepare_wm
from experiments.base import Experiment
from experiments.trainer import Trainer, Tester
from experiments.trainer_private import TesterPrivate
from experiments.utils import construct_passport_kwargs, load_passport_model_to_normal_model, \
    load_normal_model_to_passport_model
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport import AlexNetPassport
from models.resnet_normal import ResNet18, ResNet9
from models.resnet_passport import ResNet18Passport, ResNet9Passport


class ClassificationExperiment(Experiment):
    def __init__(self, args):
        super().__init__(args)

        self.in_channels = 1 if self.dataset == 'mnist' else 3
        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256,
            'imagenet1000': 1000
        }[self.dataset]

        self.train_data, self.valid_data = prepare_dataset(self.args)
        self.wm_data = None

        if self.use_trigger_as_passport:
            self.passport_data = prepare_wm('data/trigger_set/pics', crop=self.imgcrop)
        else:
            self.passport_data = self.valid_data

        if self.train_backdoor:
            self.wm_data = prepare_wm('data/trigger_set/pics', crop=self.imgcrop)

        self.construct_model()

        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0001)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        self.trainer = Trainer(self.model, optimizer, scheduler, self.device)

        if self.is_tl:
            self.finetune_load()
        else:
            self.makedirs_or_load()

    def construct_model(self):
        print('Construct Model')

        def setup_keys():
            if self.key_type != 'random':
                pretrained_from_torch = self.pretrained_path is None
                if self.arch == 'alexnet':
                    norm_type = 'none' if pretrained_from_torch else self.norm_type
                    pretrained_model = AlexNetNormal(self.in_channels,
                                                     self.num_classes,
                                                     norm_type=norm_type,
                                                     pretrained=pretrained_from_torch)
                else:
                    ResNetClass = ResNet18 if self.arch == 'resnet' else ResNet9
                    norm_type = 'bn' if pretrained_from_torch else self.norm_type
                    pretrained_model = ResNetClass(num_classes=self.num_classes,
                                                   norm_type=norm_type,
                                                   pretrained=pretrained_from_torch)

                if not pretrained_from_torch:
                    print('Loading pretrained from self-trained model')
                    pretrained_model.load_state_dict(torch.load(self.pretrained_path))
                else:
                    print('Loading pretrained from torch-pretrained model')

                pretrained_model = pretrained_model.to(self.device)
                self.setup_keys(pretrained_model)

        def load_pretrained():
            if self.pretrained_path is not None:
                sd = torch.load(self.pretrained_path)
                model.load_state_dict(sd)

        if self.train_passport:
            passport_kwargs, plkeys = construct_passport_kwargs(self, True)
            self.passport_kwargs = passport_kwargs
            self.plkeys = plkeys

            print('Loading arch: ' + self.arch)
            if self.arch == 'alexnet':
                model = AlexNetPassport(self.in_channels, self.num_classes, passport_kwargs)
            else:
                ResNetPassportClass = ResNet18Passport if self.arch == 'resnet' else ResNet9Passport
                model = ResNetPassportClass(num_classes=self.num_classes,
                                            passport_kwargs=passport_kwargs)
            self.model = model.to(self.device)

            setup_keys()
        else:  # train normally or train backdoor
            print('Loading arch: ' + self.arch)
            if self.arch == 'alexnet':
                model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
            else:
                ResNetClass = ResNet18 if self.arch == 'resnet' else ResNet9
                model = ResNetClass(num_classes=self.num_classes, norm_type=self.norm_type)

            load_pretrained()
            self.model = model.to(self.device)

        pprint(self.model)

    def setup_keys(self, pretrained_model):
        if self.key_type != 'random':
            n = 1 if self.key_type == 'image' else 20  # any number will do

            key_x, x_inds = passport_generator.get_key(self.passport_data, n)
            key_x = key_x.to(self.device)
            key_y, y_inds = passport_generator.get_key(self.passport_data, n)
            key_y = key_y.to(self.device)

            passport_generator.set_key(pretrained_model, self.model,
                                       key_x, key_y)

    def transfer_learning(self):
        if not self.is_tl:
            raise Exception('Please run with --transfer-learning')

        is_imagenet = self.num_classes == 1000

        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256,
            'imagenet1000': 1000
        }[self.tl_dataset]

        ##### load clone model #####
        print('Loading clone model')
        if self.arch == 'alexnet':
            tl_model = AlexNetNormal(self.in_channels,
                                     self.num_classes,
                                     self.norm_type,
                                     imagenet=is_imagenet)
        else:
            tl_model = ResNet18(num_classes=self.num_classes,
                                norm_type=self.norm_type,
                                imagenet=is_imagenet)

        ##### load / reset weights of passport layers for clone model #####
        tl_model.to(self.device)
        load_passport_model_to_normal_model(self.arch, self.plkeys, self.model, tl_model)
        print('Loaded clone model')

        ##### dataset is created at constructor #####

        ##### tl scheme setup #####
        if self.tl_scheme == 'rtal':
            # rtal = reset last layer + train all layer
            # ftal = train all layer
            try:
                if isinstance(tl_model.classifier, nn.Sequential):
                    tl_model.classifier[-1].reset_parameters()
                else:
                    tl_model.classifier.reset_parameters()
            except:
                tl_model.linear.reset_parameters()

        ##### optimizer setup #####
        optimizer = optim.SGD(tl_model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        ##### training is on finetune model
        self.trainer = Trainer(tl_model,
                               optimizer,
                               scheduler,
                               self.device)

        ##### tester is on original model
        tester = Tester(self.model,
                        self.device)
        tester_passport = TesterPrivate(self.model,
                                        self.device)

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True
        best_acc = 0

        for ep in range(1, self.epochs + 1):
            ##### transfer learning on new tasks #####
            train_metrics = self.trainer.train(ep, self.train_data)
            valid_metrics = self.trainer.test(self.valid_data)

            ##### load transfer learning weights from clone model  #####
            load_normal_model_to_passport_model(self.arch, self.passport_kwargs, self.model, tl_model)

            tl_model.to(self.device)
            self.model.to(self.device)

            ##### check if using weight of finetuned model is still able to detect trigger set watermark #####
            wm_metrics = {}
            if self.train_backdoor:
                wm_metrics = tester.test(self.wm_data, 'WM Result')

            ##### check if using weight of finetuend model is still able to extract signature correctly #####
            if self.train_passport:
                res = tester_passport.test_signature()
                for key in res: wm_metrics['passport_' + key] = res[key]

            ##### store results #####
            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'old_wm_{key}'] = wm_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')
                self.save_model(f'tl-epoch-{ep}.pth', tl_model)

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')
                self.save_model('tl-best.pth', tl_model)

            self.save_last_model()

    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

        print('Start training')

        for ep in range(1, self.epochs + 1):
            train_metrics = self.trainer.train(ep, self.train_data, self.wm_data)
            print(f'Sign Detection Accuracy: {train_metrics["sign_acc"] * 100:6.4f}')

            valid_metrics = self.trainer.test(self.valid_data, 'Testing Result')

            wm_metrics = {}

            if self.train_backdoor:
                wm_metrics = self.trainer.test(self.wm_data, 'WM Result')

            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'wm_{key}'] = wm_metrics[key]

            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')

            self.save_last_model()

    def evaluate(self):
        self.trainer.test(self.valid_data)
