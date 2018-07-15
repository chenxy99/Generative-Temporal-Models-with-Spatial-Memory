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

def train(epoch, model, optimizer, loader_train, lr_list, train_loss_arr, train_kld_loss_arr, train_nll_loss_arr, updating_counter):
    model.train()
    train_loss = 0
    train_kld_loss = 0
    train_nll_loss = 0
    for batch_idx, (data, _) in enumerate(loader_train):

        # transforming data
        training_data = data.to(device=device)

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = model.forward(training_data)

        loss = nll_loss.item() + kld_loss.item()
        loss_to_optimize = (nll_loss + kld_loss) / args.batch_size
        #loss_to_optimize = nll_loss + kld_loss
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

        train_loss += loss
        train_kld_loss += nll_loss.item()
        train_nll_loss += kld_loss.item()

    train_loss_arr[epoch - 1] = train_loss / len(loader_train.dataset)
    train_kld_loss_arr[epoch - 1] = train_kld_loss / len(loader_train.dataset)
    train_nll_loss_arr[epoch - 1] = train_nll_loss / len(loader_train.dataset)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(loader_train.dataset)))

    return updating_counter


def test(epoch, model, loader_val, test_nll_loss_arr):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader_val):
            data = data.to(device=device)
            kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = model.forward(
                data)
            test_loss += nll_loss

    test_loss /= len(loader_val.dataset)
    test_nll_loss_arr[epoch - 1] = test_loss
    print('====> Test set loss: {:.4f}'.format(test_loss))
