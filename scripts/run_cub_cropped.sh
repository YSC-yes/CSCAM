#!/bin/bash
# TDM+CSCAM
python3 train.py --TDM --noise --model FRN --dataset cub_cropped --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 2 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 20 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --pre --gpu_num 1
python3 train.py --TDM --noise --model FRN --dataset cub_cropped --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 400 --stage 3 --val_epoch 20 --weight_decay 5e-4 --nesterov --train_way 10 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --pre --resnet --gpu_num 1
