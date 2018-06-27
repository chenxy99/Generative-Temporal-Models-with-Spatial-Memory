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
    

NUM_TRAIN = 64*1
NUM_VAL = 64*144

latent_size = 16
batch_size = 4
clip = 10


'''
NUM_TRAIN = 73257
NUM_VAL = 26032
'''

svhn_train = dset.SVHN('./datasets/SVHN_data', download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(svhn_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))

svhn_val = dset.SVHN('./datasets/SVHN_data', download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(svhn_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
'''

cifar_train = dset.CIFAR10('./datasets/CIFAR_data', train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))

cifar_val = dset.CIFAR10('./datasets/CIFAR_data', train=False, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
'''
						
def show_images(images):
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis = 0)
        print(images.shape)
    images = np.reshape(images.cpu().numpy(), [images.shape[0], 3, -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[2])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([3,sqrtimg,sqrtimg]).transpose((1, 2, 0)))
    return 


def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-1)
            #init.xavier_uniform_(m.weight)
            if m.bias is not None:
                #init.constant_(m.bias, 0)
                init.normal_(m.bias, std=1e-1)
                

use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

               

GTM_SM_model = GTM_SM(batch_size = batch_size)
if torch.cuda.is_available():
    GTM_SM_model.cuda()  

lr_list = np.linspace(1e-3, 5e-5, num = 50000)
optimizer = optim.Adam(GTM_SM_model.parameters(), lr=lr_list[0])
#optimizer = torch.optim.SGD(GTM_SM_model.parameters(), lr = lr, momentum=0.9)

initNetParams(GTM_SM_model)

updating_counter = 0

def train(epoch):
    global updating_counter
    train_loss = 0
    for batch_idx, (data, _) in enumerate(loader_train):

        #transforming data  
        data = Variable(data, requires_grad = False).type(dtype)
        
        #forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position = GTM_SM_model.forward(data)
        #GTM_SM_model.show_experiment_information(data, st_observation_list, st_prediction_list, xt_prediction_list, position)
        loss = (nll_loss + kld_loss) 
        loss.backward()
        
        params=GTM_SM_model.state_dict() 
        for k,v in params.items():
            #if torch.sum(torch.isnan(v.grad)) != 0:
            print(v.grad)
        
        #grad norm clipping, only in pytorch version >= 1.10
        nn.utils.clip_grad_norm_(GTM_SM_model.parameters(), clip)
        
        if updating_counter >= 50000:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[-1]
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_list[updating_counter]
        
        optimizer.step()
        
        updating_counter += 1

        #printing
        if batch_idx % print_every == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), NUM_TRAIN,
                100. * batch_idx * len(data) / float(NUM_TRAIN),
                kld_loss / batch_size,
                nll_loss.item() / batch_size))
          
        train_loss += loss.item()


    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / float(NUM_TRAIN)))

n_epochs = 100
print_every = 10
save_every = 1

for epoch in range(1, n_epochs + 1):

    #training + testing        
    train(epoch)

    #saving model
    if (epoch - 1) % save_every == 0:
        fn = 'saves/gtm_sm_state_dict_'+str(epoch)+'.pth'
        torch.save(GTM_SM_model.state_dict(), fn)
        print('Saved model to '+fn)		