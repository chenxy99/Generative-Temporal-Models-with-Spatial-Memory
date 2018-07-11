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

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
training_dataset = dset.ImageFolder(root='./datasets/CelebA/training',
                                           transform=data_transform)
loader_train = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

val_dataset = dset.ImageFolder(root='./datasets/CelebA/val',
                                           transform=data_transform)
loader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

#imgs = loader_train.__iter__().next()[0].view(args.batch_size, 3, 32, 32)
#show_images(imgs)

GTM_SM_model = GTM_SM(batch_size = args.batch_size, total_dim=256+32).to(device=device)

lr_list = np.linspace(1e-3, 5e-5, num=50000)

optimizer = optim.Adam(GTM_SM_model.parameters(), lr=lr_list[0])
# optimizer = torch.optim.SGD(GTM_SM_model.parameters(), lr = lr, momentum=0.9)

initNetParams(GTM_SM_model)

updating_counter = 0

train_loss_arr = np.zeros((args.epochs))
train_kld_loss_arr = np.zeros((args.epochs))
train_nll_loss_arr = np.zeros((args.epochs))
test_nll_loss_arr = np.zeros((args.epochs))

def train(epoch):
    global updating_counter
    GTM_SM_model.train()
    train_loss = 0
    train_kld_loss = 0
    train_nll_loss = 0
    for batch_idx, (data, _) in enumerate(loader_train):

        # transforming data
        training_data = data.to(device=device)

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.forward(training_data)

        loss = (nll_loss + kld_loss)
        loss_to_optimize = (nll_loss + kld_loss) / args.batch_size
        loss_to_optimize.backward()

        # grad norm clipping, only in pytorch version >= 1.10
        #nn.utils.clip_grad_norm_(GTM_SM_model.parameters(), args.gradient_clip)

        if updating_counter >= 50000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[-1]
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[updating_counter]

        optimizer.step()
        updating_counter += 1

        # printing
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_train.dataset),
                       100. * batch_idx * len(data) / len(loader_train.dataset),
                       kld_loss.item() / len(data),
                       nll_loss.item() / len(data)))

        train_loss += loss.item()
        train_kld_loss += nll_loss.item()
        train_nll_loss += kld_loss.item()

    train_loss_arr[epoch - 1] = train_loss / len(loader_train.dataset)
    train_kld_loss_arr[epoch - 1] = train_kld_loss / len(loader_train.dataset)
    train_nll_loss_arr[epoch - 1] = train_nll_loss / len(loader_train.dataset)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader_train.dataset)))


def test(epoch):
    GTM_SM_model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader_val):
            data = data.to(device=device)
            kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.forward(
                data)
            test_loss += nll_loss

    test_loss /= len(loader_val.dataset)
    test_nll_loss_arr[epoch - 1] = test_loss
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':
    #imgs = loader_train.__iter__().next()[0].view(args.batch_size, 3, 32, 32)
    #show_images(imgs)
    for epoch in range(1, args.epochs + 1):
        # training + testing
        train(epoch)
        test(epoch)
        # saving model
        if (epoch - 1) % args.save_interval == 0:
            fn = 'saves/gtm_sm_state_dict_' + str(epoch) + '.pth'
            torch.save(GTM_SM_model.state_dict(), fn)
            print('Saved model to ' + fn)

    root = os.getcwd()
    folder_name = "result_folder"
    os.chdir(os.path.join(root, folder_name))
    np.savez("result.npz", train_loss_arr=train_loss_arr, train_kld_loss_arr=train_kld_loss_arr, train_nll_loss_arr=train_nll_loss_arr, val_nll_loss_arr=val_nll_loss_arr)
