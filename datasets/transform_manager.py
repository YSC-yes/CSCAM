import os
import math
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from copy import deepcopy
from PIL import Image



def get_transforms(input_size, eval_crop_ratio=1., mean=[0, 0, 0], std=[1., 1., 1.], aug_cfg=None, erase_prob=0.):

    transforms_train_ls = [
        torchvision.transforms.Resize(input_size, interpolation=3),
        torchvision.transforms.RandomCrop(input_size, padding=4, padding_mode='reflect'),
        torchvision.transforms.RandomHorizontalFlip()]

    if aug_cfg is not None and aug_cfg.METHOD:
        aug_module = importlib.import_module(aug_cfg.MODULE)
        aug_method = getattr(aug_module, aug_cfg.METHOD)
        transforms_train_ls += [aug_method(**aug_cfg.ARGS)]

    transforms_train_ls += [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std),
        torchvision.transforms.RandomErasing(p=erase_prob),
    ]

    transforms_train = torchvision.transforms.Compose(transforms_train_ls)

    size = int(input_size / eval_crop_ratio)

    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size, interpolation=3),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])
    return transforms_train, transforms_test



def get_transform(is_training=None,transform_type=None,pre=None):

    if is_training and pre:
        raise Exception('is_training and pre cannot be specified as True at the same time')

    if transform_type and pre:
        raise Exception('transform_type and pre cannot be specified as True at the same time')

    mean=[0.485,0.456,0.406]
    std=[0.229,0.224,0.225]

    normalize = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean,std=std)
                                    ])

    if is_training:

        if transform_type == 0:
            size_transform = transforms.RandomResizedCrop(84)
        elif transform_type == 1:
            size_transform = transforms.RandomCrop(84,padding=8)
        else:
            raise Exception('transform_type must be specified during training!')
        
        train_transform = transforms.Compose([size_transform,
                                            transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                            transforms.RandomHorizontalFlip(),
                                            normalize
                                            ])
        return train_transform
    
    elif pre:
        return normalize
    
    else:
        
        if transform_type == 0:
            size_transform = transforms.Compose([transforms.Resize(92),
                                                transforms.CenterCrop(84)])
        elif transform_type == 1:
            size_transform = transforms.Compose([transforms.Resize([92,92]),
                                                transforms.CenterCrop(84)])
        elif transform_type == 2:
            # for tiered-imagenet and (tiered) meta-inat where val/test images are already 84x84
            return normalize

        else:
            raise Exception('transform_type must be specified during inference if not using pre!')
        
        eval_transform = transforms.Compose([size_transform,normalize])
        return eval_transform
