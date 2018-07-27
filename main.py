import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import argparse
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from multiprocessing import Process

from utils.torch_utils import initNetParams, ChunkSampler, show_images, device_agnostic_selection
from model import GTM_SM
from config import *
from show_results import show_experiment_information
from train import train, test

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def main():
    data_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
    training_dataset = dset.ImageFolder(root='./datasets/CelebA/training', transform=data_transform)
    loader_train = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    val_dataset = dset.ImageFolder(root='./datasets/CelebA/val', transform=data_transform)
    loader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    GTM_SM_model = GTM_SM(batch_size=args.batch_size, total_dim=256 + 32).to(device=device)
    initNetParams(GTM_SM_model)

    optimizer = optim.Adam(GTM_SM_model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        # training + testing
        train(epoch, GTM_SM_model, optimizer, loader_train)
        test(epoch, GTM_SM_model, loader_val)
        # saving model
        if (epoch - 1) % args.save_interval == 0:
            fn = 'saves/gtm_sm_state_dict_' + str(epoch) + '.pth'
            torch.save(GTM_SM_model.state_dict(), fn)
            print('Saved model to ' + fn)

if __name__ == "__main__":
    main()