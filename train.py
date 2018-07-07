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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.torch_utils import initNetParams, ChunkSampler, show_images, device_agnostic_selection
from model import GTM_SM
from config import *

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

NUM_TRAIN = 2560
NUM_VAL = 400

data_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
MTFL_training_dataset = dset.ImageFolder(root='./datasets/MTFL_data/training',
                                           transform=data_transform)
loader_train = DataLoader(MTFL_training_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

#imgs = loader_train.__iter__().next()[0].view(batch_size, 3, 32, 32)
#show_images(imgs)

GTM_SM_model = GTM_SM(batch_size = args.batch_size).to(device=device)

lr_list = np.linspace(1e-3, 5e-5, num=50000)

optimizer = optim.Adam(GTM_SM_model.parameters(), lr=lr_list[0])
# optimizer = torch.optim.SGD(GTM_SM_model.parameters(), lr = lr, momentum=0.9)

initNetParams(GTM_SM_model)

updating_counter = 0

def train(epoch):
    global updating_counter
    GTM_SM_model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(loader_train):

        # transforming data
        training_data = data.to(device=device)

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.forward(training_data)

        if epoch == 10:
            GTM_SM_model.show_experiment_information(data, st_observation_list, st_prediction_list, xt_prediction_list, position)

        loss = (nll_loss + kld_loss)
        #loss = nll_loss
        loss.backward()


        # grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(GTM_SM_model.parameters(), args.gradient_clip)

        if updating_counter >= 50000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[-1]
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[updating_counter]

        optimizer.step()

        updating_counter += 1

        # printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), NUM_TRAIN,
                       100. * batch_idx * len(data) / float(NUM_TRAIN),
                       kld_loss / args.batch_size,
                       nll_loss.item() / args.batch_size))

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / float(NUM_TRAIN)))


n_epochs = 5000
print_every = 10
save_every = 1

for epoch in range(1, n_epochs + 1):

    # training + testing
    train(epoch)

    # saving model
    if (epoch - 1) % save_every == 0:
        fn = 'saves/gtm_sm_state_dict_' + str(epoch) + '.pth'
        torch.save(GTM_SM_model.state_dict(), fn)
        print('Saved model to ' + fn)
