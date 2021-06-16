#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train_v1.py --arch alexnet \
                                          --batch-size 256 \
                                          --epochs 200 \
                                          --lr 0.01 \
                                          --dataset cifar10 \
                                          --norm-type bn \
                                          --key-type shuffle \
                                          --sign-loss 0.1 \
                                          --passport-config passport_configs/alexnet_passport.json \
                                          --lr-config lr_configs/default.json \
                                          --save-interval 0 \
                                          --exp-id 1 \
                                          --tag exptag