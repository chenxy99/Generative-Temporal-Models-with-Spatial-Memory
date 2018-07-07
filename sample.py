import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
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
from show_results import show_experiment_information

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#load data
data_transform = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
    ])
MTFL_testing_dataset = dset.ImageFolder(root='./datasets/MTFL_data/testing',
                                           transform=data_transform)
loader_val = DataLoader(MTFL_testing_dataset, batch_size=args.batch_size, shuffle=True)


state_dict = torch.load('saves/GTM_SM_state_dict_1.pth')
GTM_SM_model = GTM_SM(batch_size = args.batch_size)
GTM_SM_model.load_state_dict(state_dict)
GTM_SM_model.to(device=device)

def sample():
    GTM_SM_model.eval()
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader_val):

            #transforming data
            training_data = data.to(device=device)
            #forward
            kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model(data)

            show_experiment_information(GTM_SM_model, data, st_observation_list, st_prediction_list, xt_prediction_list, position)

sample()