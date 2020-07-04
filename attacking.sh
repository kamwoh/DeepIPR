#!/bin/bash

GPU=\"device=${1}\" ./rundocker.sh python passport_attack_1.py --rep 10 \
                                      --arch ${3} \
                                      --dataset imagenet1000 \
                                      --scheme 1 \
                                      --loadpath ${2} \
                                      --passport-config passport_configs/${4}_passport.json