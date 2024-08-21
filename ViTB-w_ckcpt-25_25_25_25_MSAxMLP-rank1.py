import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="torch.utils._pytree._register_pytree_node is deprecated")

import glob
import sys
import os
import shutil
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import scipy as spy
import torchvision
import copy
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import ssl
from src.utils.autoaugment import CIFAR10Policy

from src.compression.LowRankLinear import LowRankLinear
import pickle, json
from transformers import ViTForImageClassification, ViTFeatureExtractor
import src.main as lc
import old_lc.main as olc
import torch.optim as optim
import src.compression.deltaCompress as lc_compress
from src.utils.utils import evaluate_accuracy, evaluate_accuracy_vit, evaluate_accuracy_gpu, evaluate_accuracy_vit_gpu, lazy_restore,lazy_restore_gpu, evaluate_compression
import wandb
# Connect to W&B
wandb.login(key="beb938fdf67db528128a4298e19b9997afd83dfd")

train_batch_size = 128
test_batch_size = 1024
num_work = 16

DECOMPOSED_LAYERS = [
    'vit.encoder.layer.0.attention.attention.query.weight',
    'vit.encoder.layer.0.attention.attention.key.weight',
    'vit.encoder.layer.0.attention.attention.value.weight',
    'vit.encoder.layer.0.intermediate.dense.weight',
    'vit.encoder.layer.0.output.dense.weight',


    'vit.encoder.layer.1.attention.attention.query.weight',
    'vit.encoder.layer.1.attention.attention.key.weight',
    'vit.encoder.layer.1.attention.attention.value.weight',
    'vit.encoder.layer.1.intermediate.dense.weight',
    'vit.encoder.layer.1.output.dense.weight',

    'vit.encoder.layer.2.attention.attention.query.weight',
    'vit.encoder.layer.2.attention.attention.key.weight',
    'vit.encoder.layer.2.attention.attention.value.weight',
    'vit.encoder.layer.2.intermediate.dense.weight',
    'vit.encoder.layer.2.output.dense.weight',

    'vit.encoder.layer.3.attention.attention.query.weight',
    'vit.encoder.layer.3.attention.attention.key.weight',
    'vit.encoder.layer.3.attention.attention.value.weight',
    'vit.encoder.layer.3.intermediate.dense.weight',
    'vit.encoder.layer.3.output.dense.weight',

    'vit.encoder.layer.4.attention.attention.query.weight',
    'vit.encoder.layer.4.attention.attention.key.weight',
    'vit.encoder.layer.4.attention.attention.value.weight',
    'vit.encoder.layer.4.intermediate.dense.weight',
    'vit.encoder.layer.4.output.dense.weight',

    'vit.encoder.layer.5.attention.attention.query.weight',
    'vit.encoder.layer.5.attention.attention.key.weight',
    'vit.encoder.layer.5.attention.attention.value.weight',
    'vit.encoder.layer.5.intermediate.dense.weight',
    'vit.encoder.layer.5.output.dense.weight',

    'vit.encoder.layer.6.attention.attention.query.weight',
    'vit.encoder.layer.6.attention.attention.key.weight',
    'vit.encoder.layer.6.attention.attention.value.weight',
    'vit.encoder.layer.6.intermediate.dense.weight',
    'vit.encoder.layer.6.output.dense.weight',

    'vit.encoder.layer.7.attention.attention.query.weight',
    'vit.encoder.layer.7.attention.attention.key.weight',
    'vit.encoder.layer.7.attention.attention.value.weight',
    'vit.encoder.layer.7.intermediate.dense.weight',
    'vit.encoder.layer.7.output.dense.weight',

    'vit.encoder.layer.8.attention.attention.query.weight',
    'vit.encoder.layer.8.attention.attention.key.weight',
    'vit.encoder.layer.8.attention.attention.value.weight',
    'vit.encoder.layer.8.intermediate.dense.weight',
    'vit.encoder.layer.8.output.dense.weight',

    'vit.encoder.layer.9.attention.attention.query.weight',
    'vit.encoder.layer.9.attention.attention.key.weight',
    'vit.encoder.layer.9.attention.attention.value.weight',
    'vit.encoder.layer.9.intermediate.dense.weight',
    'vit.encoder.layer.9.output.dense.weight',


    'vit.encoder.layer.10.attention.attention.query.weight',
    'vit.encoder.layer.10.attention.attention.key.weight',
    'vit.encoder.layer.10.attention.attention.value.weight',
    'vit.encoder.layer.10.intermediate.dense.weight',
    'vit.encoder.layer.10.output.dense.weight',

    'vit.encoder.layer.11.attention.attention.query.weight',
    'vit.encoder.layer.11.attention.attention.key.weight',
    'vit.encoder.layer.11.attention.attention.value.weight',
    'vit.encoder.layer.11.intermediate.dense.weight',
    'vit.encoder.layer.11.output.dense.weight'
]
RANK = 1
SCALING = -1
learning_rate = 2e-5
learning_rate_dloralc = 4e-5
NUM_EPOCHES = 2

seed = 42

# Set the seed for PyTorch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

# Ensure that all operations are deterministic on GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def stratified_split(dataset, proportions):
    class_indices = [np.where(np.array(dataset.targets) == i)[0] for i in range(10)]
    
    split_indices = []
    for proportion in proportions:
        class_split_indices = [np.split(indices, [int(proportion[0]*len(indices)), int((proportion[0]+proportion[1])*len(indices)), int((proportion[0]+proportion[1]+proportion[2])*len(indices))]) for indices in class_indices]
        split_indices.append([np.concatenate([split[i] for split in class_split_indices]) for i in range(4)])
    
    return split_indices

def data_loader():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),  # Add AutoAugment by default
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load the whole MNIST dataset
    full_trainset = torchvision.datasets.CIFAR10(root='./data_CIFAR10', train=True, download=True, transform=train_transform)
    
    # Proportions pour les splits
    proportions = [(0.25, 0.25, 0.25, 0.25)] * 10
    
    # Obtenir les indices pour chaque split
    split_indices = stratified_split(full_trainset, proportions)
    
    # Créer des Subsets
    trainset1 = Subset(full_trainset, split_indices[0][0])
    trainset2 = Subset(full_trainset, split_indices[0][1])
    trainset3 = Subset(full_trainset, split_indices[0][2])
    trainset4 = Subset(full_trainset, split_indices[0][3])
    
    # Créer des DataLoaders pour chacun des sous-ensembles
    train_loader1 = DataLoader(trainset1, batch_size=train_batch_size, shuffle=True, num_workers=16)
    train_loader2 = DataLoader(trainset2, batch_size=train_batch_size, shuffle=True, num_workers=16)
    train_loader3 = DataLoader(trainset3, batch_size=train_batch_size, shuffle=True, num_workers=16)
    train_loader4 = DataLoader(trainset4, batch_size=train_batch_size, shuffle=True, num_workers=16)

    # Charger le jeu de données de test complet
    testset = torchvision.datasets.CIFAR10(root='./data_CIFAR10', train=False, download=True, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=num_work)

    return train_loader1, train_loader2, train_loader3, train_loader4, test_loader

def getBase(model, basepath=""):
    wd = model.state_dict()
    w = [
        wd['vit.encoder.layer.0.attention.attention.query.weight'],
        wd['vit.encoder.layer.0.attention.attention.key.weight'],
        wd['vit.encoder.layer.0.attention.attention.value.weight'],
        wd['vit.encoder.layer.0.intermediate.dense.weight'],
        wd['vit.encoder.layer.0.output.dense.weight'],


        wd['vit.encoder.layer.1.attention.attention.query.weight'],
        wd['vit.encoder.layer.1.attention.attention.key.weight'],
        wd['vit.encoder.layer.1.attention.attention.value.weight'],
        wd['vit.encoder.layer.1.intermediate.dense.weight'],
        wd['vit.encoder.layer.1.output.dense.weight'],

        wd['vit.encoder.layer.2.attention.attention.query.weight'],
        wd['vit.encoder.layer.2.attention.attention.key.weight'],
        wd['vit.encoder.layer.2.attention.attention.value.weight'],
        wd['vit.encoder.layer.2.intermediate.dense.weight'],
        wd['vit.encoder.layer.2.output.dense.weight'],

        wd['vit.encoder.layer.3.attention.attention.query.weight'],
        wd['vit.encoder.layer.3.attention.attention.key.weight'],
        wd['vit.encoder.layer.3.attention.attention.value.weight'],
        wd['vit.encoder.layer.3.intermediate.dense.weight'],
        wd['vit.encoder.layer.3.output.dense.weight'],

        wd['vit.encoder.layer.4.attention.attention.query.weight'],
        wd['vit.encoder.layer.4.attention.attention.key.weight'],
        wd['vit.encoder.layer.4.attention.attention.value.weight'],
        wd['vit.encoder.layer.4.intermediate.dense.weight'],
        wd['vit.encoder.layer.4.output.dense.weight'],

        wd['vit.encoder.layer.5.attention.attention.query.weight'],
        wd['vit.encoder.layer.5.attention.attention.key.weight'],
        wd['vit.encoder.layer.5.attention.attention.value.weight'],
        wd['vit.encoder.layer.5.intermediate.dense.weight'],
        wd['vit.encoder.layer.5.output.dense.weight'],

        wd['vit.encoder.layer.6.attention.attention.query.weight'],
        wd['vit.encoder.layer.6.attention.attention.key.weight'],
        wd['vit.encoder.layer.6.attention.attention.value.weight'],
        wd['vit.encoder.layer.6.intermediate.dense.weight'],
        wd['vit.encoder.layer.6.output.dense.weight'],

        wd['vit.encoder.layer.7.attention.attention.query.weight'],
        wd['vit.encoder.layer.7.attention.attention.key.weight'],
        wd['vit.encoder.layer.7.attention.attention.value.weight'],
        wd['vit.encoder.layer.7.intermediate.dense.weight'],
        wd['vit.encoder.layer.7.output.dense.weight'],

        wd['vit.encoder.layer.8.attention.attention.query.weight'],
        wd['vit.encoder.layer.8.attention.attention.key.weight'],
        wd['vit.encoder.layer.8.attention.attention.value.weight'],
        wd['vit.encoder.layer.8.intermediate.dense.weight'],
        wd['vit.encoder.layer.8.output.dense.weight'],

        wd['vit.encoder.layer.9.attention.attention.query.weight'],
        wd['vit.encoder.layer.9.attention.attention.key.weight'],
        wd['vit.encoder.layer.9.attention.attention.value.weight'],
        wd['vit.encoder.layer.9.intermediate.dense.weight'],
        wd['vit.encoder.layer.9.output.dense.weight'],


        wd['vit.encoder.layer.10.attention.attention.query.weight'],
        wd['vit.encoder.layer.10.attention.attention.key.weight'],
        wd['vit.encoder.layer.10.attention.attention.value.weight'],
        wd['vit.encoder.layer.10.intermediate.dense.weight'],
        wd['vit.encoder.layer.10.output.dense.weight'],

        wd['vit.encoder.layer.11.attention.attention.query.weight'],
        wd['vit.encoder.layer.11.attention.attention.key.weight'],
        wd['vit.encoder.layer.11.attention.attention.value.weight'],
        wd['vit.encoder.layer.11.intermediate.dense.weight'],
        wd['vit.encoder.layer.11.output.dense.weight']]
    b = [
        wd['vit.encoder.layer.0.attention.attention.query.bias'],
        wd['vit.encoder.layer.0.attention.attention.key.bias'],
        wd['vit.encoder.layer.0.attention.attention.value.bias'],
        wd['vit.encoder.layer.0.intermediate.dense.bias'],
        wd['vit.encoder.layer.0.output.dense.bias'],

        wd['vit.encoder.layer.1.attention.attention.query.bias'],
        wd['vit.encoder.layer.1.attention.attention.key.bias'],
        wd['vit.encoder.layer.1.attention.attention.value.bias'],
        wd['vit.encoder.layer.1.intermediate.dense.bias'],
        wd['vit.encoder.layer.1.output.dense.bias'],

        wd['vit.encoder.layer.2.attention.attention.query.bias'],
        wd['vit.encoder.layer.2.attention.attention.key.bias'],
        wd['vit.encoder.layer.2.attention.attention.value.bias'],
        wd['vit.encoder.layer.2.intermediate.dense.bias'],
        wd['vit.encoder.layer.2.output.dense.bias'],

        wd['vit.encoder.layer.3.attention.attention.query.bias'],
        wd['vit.encoder.layer.3.attention.attention.key.bias'],
        wd['vit.encoder.layer.3.attention.attention.value.bias'],
        wd['vit.encoder.layer.3.intermediate.dense.bias'],
        wd['vit.encoder.layer.3.output.dense.bias'],

        wd['vit.encoder.layer.4.attention.attention.query.bias'],
        wd['vit.encoder.layer.4.attention.attention.key.bias'],
        wd['vit.encoder.layer.4.attention.attention.value.bias'],
        wd['vit.encoder.layer.4.intermediate.dense.bias'],
        wd['vit.encoder.layer.4.output.dense.bias'],

        wd['vit.encoder.layer.5.attention.attention.query.bias'],
        wd['vit.encoder.layer.5.attention.attention.key.bias'],
        wd['vit.encoder.layer.5.attention.attention.value.bias'],
        wd['vit.encoder.layer.5.intermediate.dense.bias'],
        wd['vit.encoder.layer.5.output.dense.bias'],

        wd['vit.encoder.layer.6.attention.attention.query.bias'],
        wd['vit.encoder.layer.6.attention.attention.key.bias'],
        wd['vit.encoder.layer.6.attention.attention.value.bias'],
        wd['vit.encoder.layer.6.intermediate.dense.bias'],
        wd['vit.encoder.layer.6.output.dense.bias'],

        wd['vit.encoder.layer.7.attention.attention.query.bias'],
        wd['vit.encoder.layer.7.attention.attention.key.bias'],
        wd['vit.encoder.layer.7.attention.attention.value.bias'],
        wd['vit.encoder.layer.7.intermediate.dense.bias'],
        wd['vit.encoder.layer.7.output.dense.bias'],

        wd['vit.encoder.layer.8.attention.attention.query.bias'],
        wd['vit.encoder.layer.8.attention.attention.key.bias'],
        wd['vit.encoder.layer.8.attention.attention.value.bias'],
        wd['vit.encoder.layer.8.intermediate.dense.bias'],
        wd['vit.encoder.layer.8.output.dense.bias'],

        wd['vit.encoder.layer.9.attention.attention.query.bias'],
        wd['vit.encoder.layer.9.attention.attention.key.bias'],
        wd['vit.encoder.layer.9.attention.attention.value.bias'],
        wd['vit.encoder.layer.9.intermediate.dense.bias'],
        wd['vit.encoder.layer.9.output.dense.bias'],

        wd['vit.encoder.layer.10.attention.attention.query.bias'],
        wd['vit.encoder.layer.10.attention.attention.key.bias'],
        wd['vit.encoder.layer.10.attention.attention.value.bias'],
        wd['vit.encoder.layer.10.intermediate.dense.bias'],
        wd['vit.encoder.layer.10.output.dense.bias'],

        wd['vit.encoder.layer.11.attention.attention.query.bias'],
        wd['vit.encoder.layer.11.attention.attention.key.bias'],
        wd['vit.encoder.layer.11.attention.attention.value.bias'],
        wd['vit.encoder.layer.11.intermediate.dense.bias'],
        wd['vit.encoder.layer.11.output.dense.bias']
        ]
    base_dict = {
        'vit.encoder.layer.0.attention.attention.query.weight': wd['vit.encoder.layer.0.attention.attention.query.weight'],
        'vit.encoder.layer.0.attention.attention.query.bias': wd['vit.encoder.layer.0.attention.attention.query.bias'],
        'vit.encoder.layer.0.attention.attention.key.weight': wd['vit.encoder.layer.0.attention.attention.key.weight'],
        'vit.encoder.layer.0.attention.attention.key.bias': wd['vit.encoder.layer.0.attention.attention.key.bias'],
        'vit.encoder.layer.0.attention.attention.value.weight': wd['vit.encoder.layer.0.attention.attention.value.weight'],
        'vit.encoder.layer.0.attention.attention.value.bias': wd['vit.encoder.layer.0.attention.attention.value.bias'],
        'vit.encoder.layer.0.intermediate.dense.weight': wd['vit.encoder.layer.0.intermediate.dense.weight'],
        'vit.encoder.layer.0.intermediate.dense.bias': wd['vit.encoder.layer.0.intermediate.dense.bias'],
        'vit.encoder.layer.0.output.dense.weight': wd['vit.encoder.layer.0.output.dense.weight'],
        'vit.encoder.layer.0.output.dense.bias': wd['vit.encoder.layer.0.output.dense.bias'],

        'vit.encoder.layer.1.attention.attention.query.weight': wd['vit.encoder.layer.1.attention.attention.query.weight'],
        'vit.encoder.layer.1.attention.attention.query.bias': wd['vit.encoder.layer.1.attention.attention.query.bias'],
        'vit.encoder.layer.1.attention.attention.key.weight': wd['vit.encoder.layer.1.attention.attention.key.weight'],
        'vit.encoder.layer.1.attention.attention.key.bias': wd['vit.encoder.layer.1.attention.attention.key.bias'],
        'vit.encoder.layer.1.attention.attention.value.weight': wd['vit.encoder.layer.1.attention.attention.value.weight'],
        'vit.encoder.layer.1.attention.attention.value.bias': wd['vit.encoder.layer.1.attention.attention.value.bias'],
        'vit.encoder.layer.1.intermediate.dense.weight': wd['vit.encoder.layer.1.intermediate.dense.weight'],
        'vit.encoder.layer.1.intermediate.dense.bias': wd['vit.encoder.layer.1.intermediate.dense.bias'],
        'vit.encoder.layer.1.output.dense.weight': wd['vit.encoder.layer.1.output.dense.weight'],
        'vit.encoder.layer.1.output.dense.bias': wd['vit.encoder.layer.1.output.dense.bias'],

        'vit.encoder.layer.2.attention.attention.query.weight': wd['vit.encoder.layer.2.attention.attention.query.weight'],
        'vit.encoder.layer.2.attention.attention.query.bias': wd['vit.encoder.layer.2.attention.attention.query.bias'],
        'vit.encoder.layer.2.attention.attention.key.weight': wd['vit.encoder.layer.2.attention.attention.key.weight'],
        'vit.encoder.layer.2.attention.attention.key.bias': wd['vit.encoder.layer.2.attention.attention.key.bias'],
        'vit.encoder.layer.2.attention.attention.value.weight': wd['vit.encoder.layer.2.attention.attention.value.weight'],
        'vit.encoder.layer.2.attention.attention.value.bias': wd['vit.encoder.layer.2.attention.attention.value.bias'],
        'vit.encoder.layer.2.intermediate.dense.weight': wd['vit.encoder.layer.2.intermediate.dense.weight'],
        'vit.encoder.layer.2.intermediate.dense.bias': wd['vit.encoder.layer.2.intermediate.dense.bias'],
        'vit.encoder.layer.2.output.dense.weight': wd['vit.encoder.layer.2.output.dense.weight'],
        'vit.encoder.layer.2.output.dense.bias': wd['vit.encoder.layer.2.output.dense.bias'],

        'vit.encoder.layer.3.attention.attention.query.weight': wd['vit.encoder.layer.3.attention.attention.query.weight'],
        'vit.encoder.layer.3.attention.attention.query.bias': wd['vit.encoder.layer.3.attention.attention.query.bias'],
        'vit.encoder.layer.3.attention.attention.key.weight': wd['vit.encoder.layer.3.attention.attention.key.weight'],
        'vit.encoder.layer.3.attention.attention.key.bias': wd['vit.encoder.layer.3.attention.attention.key.bias'],
        'vit.encoder.layer.3.attention.attention.value.weight': wd['vit.encoder.layer.3.attention.attention.value.weight'],
        'vit.encoder.layer.3.attention.attention.value.bias': wd['vit.encoder.layer.3.attention.attention.value.bias'],
        'vit.encoder.layer.3.intermediate.dense.weight': wd['vit.encoder.layer.3.intermediate.dense.weight'],
        'vit.encoder.layer.3.intermediate.dense.bias': wd['vit.encoder.layer.3.intermediate.dense.bias'],
        'vit.encoder.layer.3.output.dense.weight': wd['vit.encoder.layer.3.output.dense.weight'],
        'vit.encoder.layer.3.output.dense.bias': wd['vit.encoder.layer.3.output.dense.bias'],

        'vit.encoder.layer.4.attention.attention.query.weight': wd['vit.encoder.layer.4.attention.attention.query.weight'],
        'vit.encoder.layer.4.attention.attention.query.bias': wd['vit.encoder.layer.4.attention.attention.query.bias'],
        'vit.encoder.layer.4.attention.attention.key.weight': wd['vit.encoder.layer.4.attention.attention.key.weight'],
        'vit.encoder.layer.4.attention.attention.key.bias': wd['vit.encoder.layer.4.attention.attention.key.bias'],
        'vit.encoder.layer.4.attention.attention.value.weight': wd['vit.encoder.layer.4.attention.attention.value.weight'],
        'vit.encoder.layer.4.attention.attention.value.bias': wd['vit.encoder.layer.4.attention.attention.value.bias'],
        'vit.encoder.layer.4.intermediate.dense.weight': wd['vit.encoder.layer.4.intermediate.dense.weight'],
        'vit.encoder.layer.4.intermediate.dense.bias': wd['vit.encoder.layer.4.intermediate.dense.bias'],
        'vit.encoder.layer.4.output.dense.weight': wd['vit.encoder.layer.4.output.dense.weight'],
        'vit.encoder.layer.4.output.dense.bias': wd['vit.encoder.layer.4.output.dense.bias'],

        'vit.encoder.layer.5.attention.attention.query.weight': wd['vit.encoder.layer.5.attention.attention.query.weight'],
        'vit.encoder.layer.5.attention.attention.query.bias': wd['vit.encoder.layer.5.attention.attention.query.bias'],
        'vit.encoder.layer.5.attention.attention.key.weight': wd['vit.encoder.layer.5.attention.attention.key.weight'],
        'vit.encoder.layer.5.attention.attention.key.bias': wd['vit.encoder.layer.5.attention.attention.key.bias'],
        'vit.encoder.layer.5.attention.attention.value.weight': wd['vit.encoder.layer.5.attention.attention.value.weight'],
        'vit.encoder.layer.5.attention.attention.value.bias': wd['vit.encoder.layer.5.attention.attention.value.bias'],
        'vit.encoder.layer.5.intermediate.dense.weight': wd['vit.encoder.layer.5.intermediate.dense.weight'],
        'vit.encoder.layer.5.intermediate.dense.bias': wd['vit.encoder.layer.5.intermediate.dense.bias'],
        'vit.encoder.layer.5.output.dense.weight': wd['vit.encoder.layer.5.output.dense.weight'],
        'vit.encoder.layer.5.output.dense.bias': wd['vit.encoder.layer.5.output.dense.bias'],

        'vit.encoder.layer.6.attention.attention.query.weight': wd['vit.encoder.layer.6.attention.attention.query.weight'],
        'vit.encoder.layer.6.attention.attention.query.bias': wd['vit.encoder.layer.6.attention.attention.query.bias'],
        'vit.encoder.layer.6.attention.attention.key.weight': wd['vit.encoder.layer.6.attention.attention.key.weight'],
        'vit.encoder.layer.6.attention.attention.key.bias': wd['vit.encoder.layer.6.attention.attention.key.bias'],
        'vit.encoder.layer.6.attention.attention.value.weight': wd['vit.encoder.layer.6.attention.attention.value.weight'],
        'vit.encoder.layer.6.attention.attention.value.bias': wd['vit.encoder.layer.6.attention.attention.value.bias'],
        'vit.encoder.layer.6.intermediate.dense.weight': wd['vit.encoder.layer.6.intermediate.dense.weight'],
        'vit.encoder.layer.6.intermediate.dense.bias': wd['vit.encoder.layer.6.intermediate.dense.bias'],
        'vit.encoder.layer.6.output.dense.weight': wd['vit.encoder.layer.6.output.dense.weight'],
        'vit.encoder.layer.6.output.dense.bias': wd['vit.encoder.layer.6.output.dense.bias'],

        'vit.encoder.layer.7.attention.attention.query.weight': wd['vit.encoder.layer.7.attention.attention.query.weight'],
        'vit.encoder.layer.7.attention.attention.query.bias': wd['vit.encoder.layer.7.attention.attention.query.bias'],
        'vit.encoder.layer.7.attention.attention.key.weight': wd['vit.encoder.layer.7.attention.attention.key.weight'],
        'vit.encoder.layer.7.attention.attention.key.bias': wd['vit.encoder.layer.7.attention.attention.key.bias'],
        'vit.encoder.layer.7.attention.attention.value.weight': wd['vit.encoder.layer.7.attention.attention.value.weight'],
        'vit.encoder.layer.7.attention.attention.value.bias': wd['vit.encoder.layer.7.attention.attention.value.bias'],
        'vit.encoder.layer.7.intermediate.dense.weight': wd['vit.encoder.layer.7.intermediate.dense.weight'],
        'vit.encoder.layer.7.intermediate.dense.bias': wd['vit.encoder.layer.7.intermediate.dense.bias'],
        'vit.encoder.layer.7.output.dense.weight': wd['vit.encoder.layer.7.output.dense.weight'],
        'vit.encoder.layer.7.output.dense.bias': wd['vit.encoder.layer.7.output.dense.bias'],

        'vit.encoder.layer.8.attention.attention.query.weight': wd['vit.encoder.layer.8.attention.attention.query.weight'],
        'vit.encoder.layer.8.attention.attention.query.bias': wd['vit.encoder.layer.8.attention.attention.query.bias'],
        'vit.encoder.layer.8.attention.attention.key.weight': wd['vit.encoder.layer.8.attention.attention.key.weight'],
        'vit.encoder.layer.8.attention.attention.key.bias': wd['vit.encoder.layer.8.attention.attention.key.bias'],
        'vit.encoder.layer.8.attention.attention.value.weight': wd['vit.encoder.layer.8.attention.attention.value.weight'],
        'vit.encoder.layer.8.attention.attention.value.bias': wd['vit.encoder.layer.8.attention.attention.value.bias'],
        'vit.encoder.layer.8.intermediate.dense.weight': wd['vit.encoder.layer.8.intermediate.dense.weight'],
        'vit.encoder.layer.8.intermediate.dense.bias': wd['vit.encoder.layer.8.intermediate.dense.bias'],
        'vit.encoder.layer.8.output.dense.weight': wd['vit.encoder.layer.8.output.dense.weight'],
        'vit.encoder.layer.8.output.dense.bias': wd['vit.encoder.layer.8.output.dense.bias'],

        'vit.encoder.layer.9.attention.attention.query.weight': wd['vit.encoder.layer.9.attention.attention.query.weight'],
        'vit.encoder.layer.9.attention.attention.query.bias': wd['vit.encoder.layer.9.attention.attention.query.bias'],
        'vit.encoder.layer.9.attention.attention.key.weight': wd['vit.encoder.layer.9.attention.attention.key.weight'],
        'vit.encoder.layer.9.attention.attention.key.bias': wd['vit.encoder.layer.9.attention.attention.key.bias'],
        'vit.encoder.layer.9.attention.attention.value.weight': wd['vit.encoder.layer.9.attention.attention.value.weight'],
        'vit.encoder.layer.9.attention.attention.value.bias': wd['vit.encoder.layer.9.attention.attention.value.bias'],
        'vit.encoder.layer.9.intermediate.dense.weight': wd['vit.encoder.layer.9.intermediate.dense.weight'],
        'vit.encoder.layer.9.intermediate.dense.bias': wd['vit.encoder.layer.9.intermediate.dense.bias'],
        'vit.encoder.layer.9.output.dense.weight': wd['vit.encoder.layer.9.output.dense.weight'],
        'vit.encoder.layer.9.output.dense.bias': wd['vit.encoder.layer.9.output.dense.bias'],

        'vit.encoder.layer.10.attention.attention.query.weight': wd['vit.encoder.layer.10.attention.attention.query.weight'],
        'vit.encoder.layer.10.attention.attention.query.bias': wd['vit.encoder.layer.10.attention.attention.query.bias'],
        'vit.encoder.layer.10.attention.attention.key.weight': wd['vit.encoder.layer.10.attention.attention.key.weight'],
        'vit.encoder.layer.10.attention.attention.key.bias': wd['vit.encoder.layer.10.attention.attention.key.bias'],
        'vit.encoder.layer.10.attention.attention.value.weight': wd['vit.encoder.layer.10.attention.attention.value.weight'],
        'vit.encoder.layer.10.attention.attention.value.bias': wd['vit.encoder.layer.10.attention.attention.value.bias'],
        'vit.encoder.layer.10.intermediate.dense.weight': wd['vit.encoder.layer.10.intermediate.dense.weight'],
        'vit.encoder.layer.10.intermediate.dense.bias': wd['vit.encoder.layer.10.intermediate.dense.bias'],
        'vit.encoder.layer.10.output.dense.weight': wd['vit.encoder.layer.10.output.dense.weight'],
        'vit.encoder.layer.10.output.dense.bias': wd['vit.encoder.layer.10.output.dense.bias'],

        'vit.encoder.layer.11.attention.attention.query.weight': wd['vit.encoder.layer.11.attention.attention.query.weight'],
        'vit.encoder.layer.11.attention.attention.query.bias': wd['vit.encoder.layer.11.attention.attention.query.bias'],
        'vit.encoder.layer.11.attention.attention.key.weight': wd['vit.encoder.layer.11.attention.attention.key.weight'],
        'vit.encoder.layer.11.attention.attention.key.bias': wd['vit.encoder.layer.11.attention.attention.key.bias'],
        'vit.encoder.layer.11.attention.attention.value.weight': wd['vit.encoder.layer.11.attention.attention.value.weight'],
        'vit.encoder.layer.11.attention.attention.value.bias': wd['vit.encoder.layer.11.attention.attention.value.bias'],
        'vit.encoder.layer.11.intermediate.dense.weight': wd['vit.encoder.layer.11.intermediate.dense.weight'],
        'vit.encoder.layer.11.intermediate.dense.bias': wd['vit.encoder.layer.11.intermediate.dense.bias'],
        'vit.encoder.layer.11.output.dense.weight': wd['vit.encoder.layer.11.output.dense.weight'],
        'vit.encoder.layer.11.output.dense.bias': wd['vit.encoder.layer.11.output.dense.bias']
    }

    if basepath != "":
        if not os.path.exists(basepath):
            os.makedirs(basepath)
        fp = os.path.join(basepath, "lora_bases.pt")
        torch.save(base_dict, fp)

    return w, b

def load_sd_decomp(org_sd, model, decomposed_layers):
    """
    @param org_sd : The state_dict when the model is ongoing.
    @param model : The decomp model with decomposed layers.
    @param decomposed_layers : The decomposed layers in decomp model.

    @return The new model with the old state dictionary loaded in.
    """
    new_sd = model.state_dict()
    for k, v in org_sd.items():
        if k not in decomposed_layers:
            new_sd[k] = v
    model.load_state_dict(new_sd)

def replace_linear_with_lowrank(module, additional_weights, additional_bias):
    for i, layer in enumerate(module.encoder.layer):
        layer.attention.attention.query = LowRankLinear(768, 768, additional_weights[5*i], additional_bias[5*i], rank=RANK)
        layer.attention.attention.key = LowRankLinear(768, 768, additional_weights[5*i+1], additional_bias[5*i+1], rank=RANK)
        layer.attention.attention.value = LowRankLinear(768, 768, additional_weights[5*i+2], additional_bias[5*i+2], rank=RANK)
    for i, layer in enumerate(module.encoder.layer):
        layer.intermediate.dense = LowRankLinear(768, 3072, additional_weights[5*i+3], additional_bias[5*i+3], rank=RANK)
        layer.output.dense = LowRankLinear(3072, 768, additional_weights[5*i+4], additional_bias[5*i+4], rank=RANK)

# Load DataLoaders
train_loader1, train_loader2, train_loader3, train_loader4, test_loader = data_loader()

# Vérification des tailles des DataLoaders
print(f'Size of train_loader1: {len(train_loader1.dataset)}')
print(f'Size of train_loader2: {len(train_loader2.dataset)}')
print(f'Size of train_loader3: {len(train_loader3.dataset)}')
print(f'Size of train_loader4: {len(train_loader4.dataset)}')
print(f'Size of train_loader: {len(train_loader1.dataset) + len(train_loader2.dataset) + len(train_loader3.dataset) + len(train_loader4.dataset)}')
print(f'Size of test_loader: {len(test_loader.dataset)}')

HDFP = "./volumes/Ultra Touch" # Load HHD

SAVE_LOC = HDFP + "/lobranch-snapshot/diffbitwidth-adaptive-rank/vit/lobranch"
if not os.path.exists(SAVE_LOC):
    os.makedirs(SAVE_LOC)

SAVE_LOC_OLC = HDFP + "/lobranch-snapshot/diffbitwidth-adaptive-rank/vit/old-lc"
if not os.path.exists(SAVE_LOC_OLC):
    os.makedirs(SAVE_LOC_OLC)

SAVE_BRANCH_PRETRAINING = HDFP + "/lobranch-snapshot/branchpoints/vit/pretraining"
if not os.path.exists(SAVE_BRANCH_PRETRAINING):
    os.makedirs(SAVE_BRANCH_PRETRAINING)

SAVE_BRANCH = HDFP + "/lobranch-snapshot/branchpoints/vit/branch"
if not os.path.exists(SAVE_BRANCH):
    os.makedirs(SAVE_BRANCH)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
original = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True).to(device)
model_original = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True).to(device)
model_no_touch = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True).to(device)
model_hf = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True).to(device)
w, b = getBase(original)
model = copy.deepcopy(model_hf).to(device)
replace_linear_with_lowrank(model.vit, w, b)

optimizer = optim.Adam(model.parameters(), lr=learning_rate_dloralc)
optimizer_lc_only = optim.Adam(model_original.parameters(), lr=learning_rate)
optimizer_no_touch = optim.Adam(model_no_touch.parameters(), lr=learning_rate)

full_accuracy = []
decomposed_full_accuracy = []
restored_accuracy = []
lc_accuracy = []

# Initialize the current iteration and set to 0
current_iter = 0
current_set = 0

# Initialize the current iteration and set to 0 for the old LC method
current_iter_old_lc = 0
current_set_old_lc = 0

acc = lambda x, y : (torch.max(x, 1)[1] == y).sum().item() / y.size(0)

train_loader_list = [train_loader1, train_loader2, train_loader3, train_loader4]

# Initialiser wandb
wandb.init(project="ViT_CIFAR10_pretrained",
           name="ViT-B-16-With-Incremental-Learning-rank1",
           tags=["ViT-B", "With-Incremental-Learning", "CIFAR10"],
           config={"num_epochs": NUM_EPOCHES,
                "model_hf": "ViT-B-16",
                "train dataset": "CIFAR10 train dataset[:]",
                "test dataset": "CIFAR10 test dataset[:]",
                "splitting": "25-25-25-25",
                "train_batch_size": train_batch_size,
                "test_batch_size": test_batch_size,
                "num_workers": num_work,
                "learning_rate": learning_rate,
                "optimizer": "Adam",
                "rank": 1,
           })

early_stopping_patience = 2*len(train_loader1)  # Stop if no improvement in accuracy after this many epochs
best_accuracy = 0
epochs_without_improvement = 0

for j in range(len(train_loader_list)):
    train_loader_txt = "train_loader{}".format(j+1)
    print("--------------------------")
    print("Beginning of model training on {}...".format(train_loader_txt))

    full_accuracy_dloralc = 0
    decomposed_full_accuracy_dloralc = 0
    restored_accuracy_dloralc = 0
    lc_accuracy_dloralc = 0

    for epch in range(NUM_EPOCHES):
        for i, data in enumerate(train_loader_list[j], 0):

            SAVE_LOC_j = SAVE_LOC + "/"+train_loader_txt
            if not os.path.exists(SAVE_LOC_j):
                os.makedirs(SAVE_LOC_j)
                
            SAVE_LOC_OLC_j = SAVE_LOC_OLC + "/"+train_loader_txt
            if not os.path.exists(SAVE_LOC_OLC_j):
                os.makedirs(SAVE_LOC_OLC_j)
            print("Epoch: {}, Iteration: {}".format(epch, i))
            
            set_path = "/set_{}".format(current_set)
            if not os.path.exists(SAVE_LOC_j + set_path):
                os.makedirs(SAVE_LOC_j + set_path)

            if i == 0 and epch == 0: # first iteration, create baseline model
                base, base_decomp = lc.extract_weights_gpu(model, SAVE_LOC_j + 
                                                        "/set_{}".format(current_set), DECOMPOSED_LAYERS)
            else:
                if i % 10 == 0: 
                    # full snapshot!
                    new_model = lazy_restore_gpu(base, base_decomp, bias, ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True), 
                                            original.state_dict(), DECOMPOSED_LAYERS, rank = RANK, scaling = SCALING)
                    original = new_model # Changing previous "original model" used to restore the loRA model.
                    
                    current_set += 1
                    current_iter = 0

                    set_path = "/set_{}".format(current_set)
                    if not os.path.exists(SAVE_LOC_j + set_path):
                        os.makedirs(SAVE_LOC_j + set_path)
                    
                    # Rebuilding LoRA layers => reset model!
                    w, b = getBase(original)
                    model = copy.deepcopy(model_hf).to(device)
                    replace_linear_with_lowrank(model.vit, w, b)
                    optimizer = optim.Adam(model.parameters(), lr = learning_rate_dloralc)
                    load_sd_decomp(original.state_dict(), model, DECOMPOSED_LAYERS)
                    base, base_decomp = lc.extract_weights_gpu(model, SAVE_LOC_j + 
                                                        "/set_{}".format(current_set), DECOMPOSED_LAYERS)

                else:
                    # Delta-compression
                    delta, decomp_delta, bias = lc.generate_delta_gpu(base, 
                                                                    base_decomp, model.state_dict(), DECOMPOSED_LAYERS)
                    compressed_delta, full_delta, compressed_dcomp_delta, full_dcomp_delta  = lc.compress_delta(delta, 
                                                                                                                decomp_delta)
                    
                    # Saving checkpoint
                    lc.save_checkpoint(compressed_delta, compressed_dcomp_delta, bias, current_iter, SAVE_LOC_j + 
                                    "/set_{}".format(current_set))
        
                    base = np.add(base, full_delta) # Replace base with latest for delta to accumulate.
                    base_decomp = np.add(full_dcomp_delta, base_decomp)

                    current_iter += 1
                
            # ==========================
            # Saving using LC-Checkpoint
            # ==========================
                    
            if i == 0 and epch == 0:
                cstate = model_original.state_dict()
                set_path = "/set_{}".format(current_set_old_lc)
                if not os.path.exists(SAVE_LOC_OLC_j + set_path):
                    os.makedirs(SAVE_LOC_OLC_j + set_path)
                torch.save(cstate, SAVE_LOC_OLC_j + set_path + "/initial_model.pt")
                prev_state = olc.extract_weights_gpu(cstate, SAVE_LOC_OLC_j + set_path, DECOMPOSED_LAYERS)
            else:
                if i % 10 == 0:
                    cstate = model_original.state_dict()
                    current_set_old_lc += 1
                    current_iter_old_lc = 0
                    set_path = "/set_{}".format(current_set_old_lc)
                    if not os.path.exists(SAVE_LOC_OLC_j + set_path):
                        os.makedirs(SAVE_LOC_OLC_j + set_path)
                    torch.save(cstate, SAVE_LOC_OLC_j + set_path + "/initial_model.pt")
                    prev_state = olc.extract_weights_gpu(cstate, SAVE_LOC_OLC_j + set_path, DECOMPOSED_LAYERS)
                else:
                    cstate = model_original.state_dict()
                    old_lc_delta, old_lc_bias = olc.generate_delta_gpu(prev_state, cstate, DECOMPOSED_LAYERS)
                    olc_compressed_delta, update_prev = olc.compress_data(old_lc_delta, num_bits = 3)
                    olc.save_checkpoint(SAVE_LOC_OLC_j + "/set_{}".format(current_set_old_lc), olc_compressed_delta, 
                                        old_lc_bias, current_iter_old_lc)
                    prev_state = np.add(prev_state, update_prev)
                    current_iter_old_lc += 1
            
            # ==========================
            # Training on Low-Rank Model
            # ==========================

            # Get the inputs and labels
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs).logits
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            print("LoRA+LC Training Loss (Decomposed): {}".format(loss.item()))
            optimizer.step()
                
            # ======================
            # Training on Full Model
            # ======================

            # Zero the parameter gradients
            optimizer_lc_only.zero_grad()

            # Forward + backward + optimize
            outputs_full = model_original(inputs).logits
            loss_full = torch.nn.CrossEntropyLoss()(outputs_full,labels)
            loss_full.backward()
            print("LC Training Loss (Full): {}".format(loss_full.item()))
            optimizer_lc_only.step()

            if i % 20 == 0:
                print("Training Accuracy | Decomposed: {}, Full : {}".format(acc(outputs, labels), 
                                                                            acc(outputs_full, labels)))

            if i != 0  and i % 5 == 0: # Evaluation on testing set
                full_accuracy_dloralc = evaluate_accuracy_vit_gpu(model_original, test_loader, device)
                full_accuracy.append(full_accuracy_dloralc)

                decomposed_full_accuracy_dloralc = evaluate_accuracy_vit_gpu(model, test_loader, device)
                decomposed_full_accuracy.append(decomposed_full_accuracy_dloralc)
                
                restored_model = lazy_restore_gpu(base, base_decomp, bias, ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True), 
                                            original.state_dict(), DECOMPOSED_LAYERS, 
                                            rank = RANK, scaling = SCALING)
                restored_accuracy_dloralc_restored = evaluate_accuracy_vit(restored_model, test_loader)
                restored_accuracy.append(restored_accuracy_dloralc_restored)

                restored_lc_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True).to(device)
                restored_lc_model.load_state_dict(olc.restore_state_dict(prev_state, old_lc_bias, 
                                                                    restored_model.state_dict(), DECOMPOSED_LAYERS))
                lc_accuracy_lc = evaluate_accuracy_vit_gpu(restored_lc_model, test_loader, device)
                lc_accuracy.append(lc_accuracy_lc)
                print("Full accuracy (w/o dLoRA+LC): {}, LC accuracy: {}, Decomposed-Full (w/dLoRA+LC) accuracy: {}, Decomposed-Restored (w/dLoRA+LC restored) accuracy: {}".format(
                    full_accuracy[-1], lc_accuracy[-1], decomposed_full_accuracy[-1], restored_accuracy[-1]))
                
                wandb.log({
                    "accuracy without dLoRALC": full_accuracy[-1],
                    "accuracy with LC": lc_accuracy[-1],
                    "accuracy with dLoRALC": decomposed_full_accuracy[-1],
                    "accuracy with dLoRALC after restoration": restored_accuracy[-1]
                })
        
        # Early stopping check
        current_accuracy = restored_accuracy_dloralc_restored
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epch+1} epochs for {train_loader_txt}")
            break
                

    print("End of model training on {}...".format(train_loader_txt))

    rounded_valid_acc = decomposed_full_accuracy_dloralc
    torch.save(model.state_dict(), HDFP + "/lobranch-snapshot/branchpoints/vit/branch/branch_{}.pt".format(rounded_valid_acc))
    print("Model saved at accuracy: {:.4f}".format(rounded_valid_acc))

    w, b = getBase(original)
    model = copy.deepcopy(model_hf).to(device)
    replace_linear_with_lowrank(model.vit, w, b)

    model.load_state_dict(torch.load(HDFP + "/lobranch-snapshot/branchpoints/vit/branch/branch_{}.pt".format(rounded_valid_acc)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_dloralc)

    # Initialize the current iteration and set to 0
    current_iter = 0
    current_set = 0

    # Initialize the current iteration and set to 0 for the old LC method
    current_iter_old_lc = 0
    current_set_old_lc = 0

torch.save(model_original.state_dict(), './vitB.pt')

import math
def getsize(sl):
    dir = [x for x in os.listdir(sl)]
    csize, usize = 0, 0
    for dataloader in dir:
        for set in os.listdir(sl + "/" + dataloader):
            # print(set)
            for f in os.listdir(sl + "/" + dataloader + "/" + set):
                # print(f)
                fp = sl + "/" + dataloader + "/{}/{}".format(set, f)
                csize += os.path.getsize(fp)
                usize += 327.0 * math.pow(2, 20) # torch checkpoint same size
    return csize, usize

compressed_size, uncompressed_size = getsize(SAVE_LOC)
a, b = evaluate_compression(uncompressed_size, compressed_size)
compressed_size, uncompressed_size = getsize(SAVE_LOC_OLC)
a1, b1 = evaluate_compression(uncompressed_size, compressed_size)

print("LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a1, b1))
print("LoRA + LC-Checkpoint + GZIP")
print("Compression Ratio: {}%, Space Savings: {}%".format(a, b))