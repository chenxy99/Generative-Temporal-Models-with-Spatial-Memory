import torch
import torch.nn as nn
#from torch.nn import init
from torch.autograd import Variable
import torch.nn.init as init
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import pyflann
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import GTM_SM

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


#load data
batch_size = 1

NUM_TRAIN = 1000
NUM_VAL = 26032


svhn_val = dset.SVHN('./datasets/SVHN_data', download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(svhn_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

state_dict = torch.load('saves/GTM_SM_state_dict_1.pth')
GTM_SM_model = GTM_SM(batch_size = batch_size)
GTM_SM_model.load_state_dict(state_dict)
if torch.cuda.is_available():
    GTM_SM_model.cuda() 


def sample():
    for batch_idx, (data, _) in enumerate(loader_val):

        #transforming data  
        data = Variable(data, requires_grad = False).type(dtype)        
        #forward 
        st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.Sampling(data)
        
        GTM_SM_model.show_experiment_information(data, st_observation_list, st_prediction_list, xt_prediction_list, position)

sample()