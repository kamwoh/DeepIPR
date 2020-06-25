import os
from pprint import pprint

import torch
import torch.optim as optim

import passport_generator
from dataset import prepare_dataset, prepare_wm
from experiments.base import Experiment
from experiments.trainer import Trainer, Tester
from experiments.trainer_private import TesterPrivate
from experiments.utils import construct_passport_kwargs
from models.alexnet_normal import AlexNetNormal
from models.alexnet_passport import AlexNetPassport
from models.layers.conv2d import ConvBlock
from models.layers.passportconv2d import PassportBlock
from models.resnet_normal import ResNet18
from models.resnet_passport import ResNet18Passport


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

        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.2023, 0.1994, 0.2010])

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
                              weight_decay=0.0005)

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
        def setup_keys():
            if self.key_type != 'random':
                if self.arch == 'alexnet':
                    pretrained_model = AlexNetNormal(self.in_channels, self.num_classes)
                else:
                    pretrained_model = ResNet18(num_classes=self.num_classes,
                                                norm_type=self.norm_type)

                pretrained_model.load_state_dict(torch.load(self.pretrained_path))
                pretrained_model = pretrained_model.to(self.device)
                self.setup_keys(pretrained_model)

        def load_pretrained():
            if self.pretrained_path is not None:
                sd = torch.load(self.pretrained_path)
                model.load_state_dict(sd)

        if self.train_passport:
            passport_kwargs = construct_passport_kwargs(self)
            self.passport_kwargs = passport_kwargs

            print('Loading arch: ' + self.arch)
            if self.arch == 'alexnet':
                model = AlexNetPassport(self.in_channels, self.num_classes, passport_kwargs)
            else:
                model = ResNet18Passport(num_classes=self.num_classes,
                                         passport_kwargs=passport_kwargs)
            self.model = model.to(self.device)

            setup_keys()
        else:  # train normally or train backdoor
            print('Loading arch: ' + self.arch)
            if self.arch == 'alexnet':
                model = AlexNetNormal(self.in_channels, self.num_classes, self.norm_type)
            else:
                model = ResNet18(num_classes=self.num_classes, norm_type=self.norm_type)

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

        self.num_classes = {
            'cifar10': 10,
            'cifar100': 100,
            'caltech-101': 101,
            'caltech-256': 256
        }[self.tl_dataset]

        ##### load clone model #####
        print('Loading clone model')
        if self.arch == 'alexnet':
            clone_model = AlexNetNormal(self.in_channels,
                                        self.num_classes,
                                        self.norm_type)
        else:
            clone_model = ResNet18(num_classes=self.num_classes,
                                   norm_type=self.norm_type)

        ##### load / reset weights of passport layers for clone model #####
        try:
            clone_model.load_state_dict(self.model.state_dict())
        except:
            print('Having problem to direct load state dict, loading it manually')
            if self.arch == 'alexnet':
                for clone_m, self_m in zip(clone_model.features, self.model.features):
                    try:
                        clone_m.load_state_dict(self_m.state_dict())
                    except:
                        print('Having problem to load state dict usually caused by missing keys, load by strict=False')
                        clone_m.load_state_dict(self_m.state_dict(), False)  # load conv weight, bn running mean
                        clone_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                        clone_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

            else:
                passport_settings = self.passport_config
                for l_key in passport_settings:  # layer
                    if isinstance(passport_settings[l_key], dict):
                        for i in passport_settings[l_key]:  # sequential
                            for m_key in passport_settings[l_key][i]:  # convblock
                                clone_m = clone_model.__getattr__(l_key)[int(i)].__getattr__(m_key)  # type: ConvBlock
                                self_m = self.model.__getattr__(l_key)[int(i)].__getattr__(m_key)  # type: PassportBlock

                                try:
                                    clone_m.load_state_dict(self_m.state_dict())
                                except:
                                    print(f'{l_key}.{i}.{m_key} cannot load state dict directly')
                                    clone_m.load_state_dict(self_m.state_dict(), False)
                                    clone_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                                    clone_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

                    else:
                        clone_m = clone_model.__getattr__(l_key)
                        self_m = self.model.__getattr__(l_key)

                        try:
                            clone_m.load_state_dict(self_m.state_dict())
                        except:
                            print(f'{l_key} cannot load state dict directly')
                            clone_m.load_state_dict(self_m.state_dict(), False)
                            clone_m.bn.weight.data.copy_(self_m.get_scale().detach().view(-1))
                            clone_m.bn.bias.data.copy_(self_m.get_bias().detach().view(-1))

        clone_model.to(self.device)
        print('Loaded clone model')

        ##### dataset is created at constructor #####

        ##### tl scheme setup #####
        if self.tl_scheme == 'rtal':
            # rtal = reset last layer + train all layer
            # ftal = train all layer
            try:
                clone_model.classifier.reset_parameters()
            except:
                clone_model.linear.reset_parameters()

        ##### optimizer setup #####
        optimizer = optim.SGD(clone_model.parameters(),
                              lr=self.lr,
                              momentum=0.9,
                              weight_decay=0.0005)

        if len(self.lr_config[self.lr_config['type']]) != 0:  # if no specify steps, then scheduler = None
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       self.lr_config[self.lr_config['type']],
                                                       self.lr_config['gamma'])
        else:
            scheduler = None

        self.trainer = Trainer(clone_model,
                               optimizer,
                               scheduler,
                               self.device)
        tester = Tester(self.model,
                        self.device)
        tester_passport = TesterPrivate(self.model,
                                        self.device)

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True
        best_acc = 0

        for ep in range(1, self.epochs + 1):
            train_metrics = self.trainer.train(ep, self.train_data)
            valid_metrics = self.trainer.test(self.valid_data)

            ##### load transfer learning weights from clone model  #####
            try:
                self.model.load_state_dict(clone_model.state_dict())
            except:
                if self.arch == 'alexnet':
                    for clone_m, self_m in zip(clone_model.features, self.model.features):
                        try:
                            self_m.load_state_dict(clone_m.state_dict())
                        except:
                            self_m.load_state_dict(clone_m.state_dict(), False)
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

            clone_model.to(self.device)
            self.model.to(self.device)

            wm_metrics = {}
            if self.train_backdoor:
                wm_metrics = tester.test(self.wm_data, 'WM Result')

            if self.train_passport:
                res = tester_passport.test_signature()
                for key in res: wm_metrics['passport_' + key] = res[key]

            metrics = {}
            for key in train_metrics: metrics[f'train_{key}'] = train_metrics[key]
            for key in valid_metrics: metrics[f'valid_{key}'] = valid_metrics[key]
            for key in wm_metrics: metrics[f'old_wm_{key}'] = wm_metrics[key]
            self.append_history(history_file, metrics, first)
            first = False

            if self.save_interval and ep % self.save_interval == 0:
                self.save_model(f'epoch-{ep}.pth')
                self.save_model(f'tl-epoch-{ep}.pth', clone_model)

            if best_acc < metrics['valid_acc']:
                print(f'Found best at epoch {ep}\n')
                best_acc = metrics['valid_acc']
                self.save_model('best.pth')
                self.save_model('tl-best.pth', clone_model)

            self.save_last_model()

    def training(self):
        best_acc = float('-inf')

        history_file = os.path.join(self.logdir, 'history.csv')
        first = True

        if self.save_interval > 0:
            self.save_model('epoch-0.pth')

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
