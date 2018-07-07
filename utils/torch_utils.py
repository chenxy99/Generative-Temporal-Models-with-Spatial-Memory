import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import sampler
import argparse

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def initNetParams(net):
    '''Init net parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            #init.normal_(m.weight, std=1e-1)
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
                # init.normal_(m.bias, std=1e-1)


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


def show_images(images):
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        print(images.shape)
    images = np.reshape(images.cpu().detach().numpy(), [images.shape[0], 3, -1])  # images reshape to (batch_size, D)
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
        plt.imshow(img.reshape([3, sqrtimg, sqrtimg]).transpose((1, 2, 0)))

    # show figure
    plt.show()
    return


def device_agnostic_selection():
    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    args = parser.parse_args()
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    return args.device