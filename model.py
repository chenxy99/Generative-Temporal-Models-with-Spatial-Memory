import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

import numpy as np
import pyflann
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.torch_utils import initNetParams, ChunkSampler, show_images, device_agnostic_selection
from config import *
from roam import random_walk
"""implementation of the Generative Temporal Models 
with Spatial Memory (GTM-SM) from https://arxiv.org/abs/1804.09401
"""

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.contiguous().view(N, -1)


class Exponent(nn.Module):
    def forward(self, x):
        return torch.exp(x)


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=3, H=8, W=8):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

class GTM_SM(nn.Module):
    def __init__(self, x_dim=8, a_dim=5, s_dim=2, z_dim=16, observe_dim=256, total_dim=288, \
                 r_std=0.001, k_nearest_neighbour=5, delta=0.0001, kl_samples=1000, batch_size=1):
        super(GTM_SM, self).__init__()

        self.x_dim = x_dim
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.observe_dim = observe_dim
        self.z_dim = z_dim
        self.total_dim = total_dim
        self.r_std = r_std
        self.k_nearest_neighbour = k_nearest_neighbour
        self.delta = delta
        self.kl_samples = kl_samples
        self.batch_size = batch_size
        self.flanns = pyflann.FLANN()

        # feature-extracting transformations
        # encoder
        # for zt
        self.enc_zt = nn.Sequential(nn.Conv2d(3, 16, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(2),
                                    nn.BatchNorm2d(16),
                                    nn.Conv2d(16, 32, kernel_size=2, stride=2),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(32),
                                    Flatten())

        self.enc_zt_mean = nn.Sequential(nn.Linear(32, z_dim))

        self.enc_zt_std = nn.Sequential(nn.Linear(32, z_dim),
                                        Exponent())

        # for st
        self.enc_st_matrix = nn.Sequential(
            nn.Linear(a_dim, s_dim, bias=False))

        self.enc_st_sigmoid = nn.Sequential(
            nn.Linear(s_dim, 5),
            nn.Tanh(),
            nn.Linear(5, s_dim),
            nn.Sigmoid())

        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            Unflatten(-1, 32, 2, 2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        if not self.training:
            origin_total_dim = self.total_dim
            self.total_dim = 512
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        '''
        action_one_hot_value        tensor  (self.batch_size, self.a_dim, self.total_dim)
        position                    np      (self.batch_size, self.s_dim, self.total_dim)
        action_selection            np      (self.batch_size, self.total_dim)
        st_observation_list         list    (self.observe_dim)(self.batch_size, self.s_dim)
        st_prediction_list          list    (self.total_dim - self.observe_dim)(self.batch_size, self.s_dim)
        zt_mean_observation_list    list    (self.observe_dim)(self.batch_size, self.z_dim)
        zt_std_observation_list     list    (self.observe_dim)(self.batch_size, self.z_dim)
        zt_mean_prediction_list     list    (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        zt_std_prediction_list      list    (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        x_resconstruct_t            list    (self.total_dim - self.observe_dim)(self.batch_size, self.x_dim)

        after construct them, we will use torch.cat to eliminate the list object

        st_observation_tensor       tensor      (self.observe_dim)(self.batch_size, self.s_dim)
        st_prediction_tensor        tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.s_dim)
        zt_mean_observation_tensor  tensor      (self.observe_dim)(self.batch_size, self.z_dim)
        zt_std_observation_tensor   tensor      (self.observe_dim)(self.batch_size, self.z_dim)
        zt_mean_prediction_tensor   tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        zt_std_prediction_tensor    tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)

        '''

        action_one_hot_value, position, action_selection = random_walk(self)
        st_observation_list = []
        st_prediction_list = []
        zt_mean_observation_list = []
        zt_std_observation_list = []
        zt_mean_prediction_list = []
        zt_std_prediction_list = []
        xt_prediction_list = []

        kld_loss = 0
        nll_loss = 0

        # observation phase: construct st
        for t in range(self.observe_dim):
            if t == 0:
                st_observation_t = torch.rand(self.batch_size, self.s_dim, device=device) - 1
            else:
                st_observation_t = st_observation_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t]) * \
                                   self.enc_st_sigmoid(st_observation_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t])) + \
                                   torch.normal(mean=torch.zeros(self.batch_size, self.s_dim, device=device), \
                                                std=self.r_std * torch.ones(self.batch_size, self.s_dim, device=device))
            st_observation_list.append(st_observation_t)
        st_observation_tensor = torch.cat(st_observation_list, 0).view(self.observe_dim, self.batch_size, self.s_dim)

        # prediction phase: construct st
        for t in range(self.total_dim - self.observe_dim):
            if t == 0:
                st_prediction_t = st_observation_list[-1] + self.enc_st_matrix(
                    action_one_hot_value[:, :, t + self.observe_dim]) * \
                                  self.enc_st_sigmoid(st_observation_list[-1] + self.enc_st_matrix(
                                      action_one_hot_value[:, :, t + self.observe_dim])) + \
                                  torch.normal(mean=torch.zeros(self.batch_size, self.s_dim, device=device),
                                               std=self.r_std * torch.ones(self.batch_size, self.s_dim, device=device))
            else:
                st_prediction_t = st_prediction_list[t - 1] + self.enc_st_matrix(
                    action_one_hot_value[:, :, t + self.observe_dim]) * \
                                  self.enc_st_sigmoid(st_prediction_list[t - 1] + self.enc_st_matrix(
                                      action_one_hot_value[:, :, t + self.observe_dim])) + \
                                  torch.normal(mean=torch.zeros(self.batch_size, self.s_dim, device=device),
                                               std=self.r_std * torch.ones(self.batch_size, self.s_dim, device=device))
            st_prediction_list.append(st_prediction_t)
        st_prediction_tensor = torch.cat(st_prediction_list, 0).view(self.total_dim - self.observe_dim, self.batch_size,
                                                                     self.s_dim)

        # observation phase: construct zt from xt
        for t in range(self.observe_dim):
            index_mask = torch.zeros((self.batch_size, 3, 32, 32), device=device)
            for index_sample in range(self.batch_size):
                position_h_t = position[index_sample, 0, t]
                position_w_t = position[index_sample, 1, t]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,
                3 * position_w_t:3 * position_w_t + 8] = torch.ones([1], device=device)
                index_mask_bool = index_mask.ge(0.5)
            x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            zt_observation_t = self.enc_zt(x_feed)
            zt_mean_observation_t = self.enc_zt_mean(zt_observation_t)
            zt_std_observation_t = self.enc_zt_std(zt_observation_t)
            zt_mean_observation_list.append(zt_mean_observation_t)
            zt_std_observation_list.append(zt_std_observation_t)
        zt_mean_observation_tensor = torch.cat(zt_mean_observation_list, 0).view(self.observe_dim, self.batch_size,
                                                                                 self.z_dim)
        zt_std_observation_tensor = torch.cat(zt_std_observation_list, 0).view(self.observe_dim, self.batch_size,
                                                                               self.z_dim)


        if self.training:
            # prediction phase: construct zt from xt
            for t in range(self.total_dim - self.observe_dim):
                index_mask = torch.zeros((self.batch_size, 3, 32, 32), device=device)
                for index_sample in range(self.batch_size):
                    position_h_t = position[index_sample, 0, t + self.observe_dim]
                    position_w_t = position[index_sample, 1, t + self.observe_dim]
                    index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,
                    3 * position_w_t:3 * position_w_t + 8] = torch.ones([1], device=device)
                    index_mask_bool = index_mask.ge(0.5)
                x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
                zt_prediction_t = self.enc_zt(x_feed)
                zt_mean_prediction_t = self.enc_zt_mean(zt_prediction_t)
                zt_std_prediction_t = self.enc_zt_std(zt_prediction_t)
                zt_mean_prediction_list.append(zt_mean_prediction_t)
                zt_std_prediction_list.append(zt_std_prediction_t)
            zt_mean_prediction_tensor = torch.cat(zt_mean_prediction_list, 0).view(self.total_dim - self.observe_dim,
                                                                                   self.batch_size, self.z_dim)
            zt_std_prediction_tensor = torch.cat(zt_std_prediction_list, 0).view(self.total_dim - self.observe_dim,
                                                                                 self.batch_size, self.z_dim)

            # reparameterized_sample to calculate the reconstruct error
            for t in range(self.total_dim - self.observe_dim):
                zt_prediction_sample = self._reparameterized_sample(zt_mean_prediction_list[t], zt_std_prediction_list[t])
                index_mask = torch.zeros((self.batch_size, 3, 32, 32), device=device)
                for index_sample in range(self.batch_size):
                    position_h_t = position[index_sample, 0, t + self.observe_dim]
                    position_w_t = position[index_sample, 1, t + self.observe_dim]
                    index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,
                    3 * position_w_t:3 * position_w_t + 8] = torch.ones([1], device=device)
                    index_mask_bool = index_mask.ge(0.5)
                x_ground_true_t = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
                x_resconstruct_t = self.dec(zt_prediction_sample)
                nll_loss += self._nll_gauss(x_resconstruct_t, x_ground_true_t)
                xt_prediction_list.append(x_resconstruct_t)

        # construct kd tree
        st_observation_memory = np.zeros((self.observe_dim, self.batch_size, self.s_dim))
        for t in range(self.observe_dim):
            st_observation_memory[t] = st_observation_list[t].cpu().detach().numpy()

        st_prediction_memory = np.zeros((self.total_dim - self.observe_dim, self.batch_size, self.s_dim))
        for t in range(self.total_dim - self.observe_dim):
            st_prediction_memory[t] = st_prediction_list[t].cpu().detach().numpy()

        results = []
        for index_sample in range(self.batch_size):
            param = self.flanns.build_index(st_observation_memory[:, index_sample, :], algorithm='kdtree',
                                                     trees=4)
            result, _ = self.flanns.nn_index(st_prediction_memory[:, index_sample, :],
                                                      self.k_nearest_neighbour, checks=param["checks"])
            results.append(result)

        if self.training:
            # calculate the kld
            for index_sample in range(self.batch_size):
                for t in range(self.total_dim - self.observe_dim):
                    t_knn_index = results[index_sample][t]
                    t_knn_st_memory = st_observation_tensor[t_knn_index, index_sample]
                    dk2 = ((t_knn_st_memory - st_prediction_tensor[t, index_sample, :]) ** 2).sum(1)
                    wk = 1 / (dk2 + self.delta)
                    normalized_wk = wk / torch.sum(wk)
                    log_normalized_wk = torch.log(normalized_wk)

                    # sampling
                    zt_sampling = self._reparameterized_sample_cluster(zt_mean_prediction_tensor[t, index_sample],
                                                                       zt_std_prediction_tensor[t, index_sample])
                    log_q_phi = self._log_gaussian_pdf(zt_sampling, zt_mean_prediction_tensor[t, index_sample],
                                                      zt_std_prediction_tensor[t, index_sample])
                    log_p_theta_element = ((self._log_gaussian_element_pdf(zt_sampling, zt_mean_observation_tensor[
                        t_knn_index, index_sample],zt_std_observation_tensor[t_knn_index, index_sample]).t()).t()) + log_normalized_wk
                    # print(log_p_theta_element_minus_log_q_phi)
                    (log_p_theta_element_max, _) = torch.max(log_p_theta_element, 1)
                    log_p_theta_element_nimus_max = (log_p_theta_element.t() - log_p_theta_element_max).t()
                    p_theta_nimus_max = torch.exp(log_p_theta_element_nimus_max).sum(1)
                    kld_loss += torch.mean(log_q_phi - log_p_theta_element_max - torch.log(p_theta_nimus_max))
        else:
            xt_prediction_tensor = torch.zeros(self.total_dim - self.observe_dim, self.batch_size, 3, 8, 8,
                                               device=device)

            for index_sample in range(self.batch_size):
                for t in range(self.total_dim - self.observe_dim):
                    t_knn_index = results[index_sample][t]
                    t_knn_st_memory = st_observation_tensor[t_knn_index, index_sample]
                    dk2 = ((t_knn_st_memory - st_prediction_tensor[t, index_sample, :]) ** 2).sum(1)
                    wk = 1 / (dk2 + self.delta)
                    normalized_wk = wk / torch.sum(wk)
                    cumsum_normalized_wk = torch.cumsum(normalized_wk, dim=0)
                    rand_sample_value = torch.rand(1, device=device)

                    for sample_knn in range(self.k_nearest_neighbour):
                        if sample_knn == 0:
                            if cumsum_normalized_wk[sample_knn] > rand_sample_value:
                                break
                        else:
                            if cumsum_normalized_wk[sample_knn] > rand_sample_value and cumsum_normalized_wk[
                                sample_knn - 1] <= rand_sample_value:
                                break
                                # sampling
                    zt_sampling = self._reparameterized_sample(
                        zt_mean_observation_tensor[t_knn_index[sample_knn], index_sample],
                        zt_std_observation_tensor[t_knn_index[sample_knn], index_sample])
                    xt_prediction_tensor[t, index_sample] = self.dec(zt_sampling)

            # reparameterized_sample to calculate the reconstruct error
            for t in range(self.total_dim - self.observe_dim):
                xt_prediction_list.append(xt_prediction_tensor[t])
        if not self.training:
            self.total_dim = origin_total_dim

        return kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position

    def _log_gaussian_pdf(self, zt, zt_mean, zt_std):
        constant_value = torch.tensor(2 * 3.1415926535, device = device)
        log_exp_term = - torch.sum((((zt - zt_mean) ** 2) / (zt_std ** 2) / 2.0), 1)
        log_other_term = - (self.z_dim / 2.0) * torch.log(constant_value) - torch.sum(torch.log(zt_std))
        return log_exp_term + log_other_term

    def _log_gaussian_element_pdf(self, zt, zt_mean, zt_std):
        constant_value = torch.tensor(2 * 3.1415926535, device = device)
        log_exp_term = - torch.sum(
            (((zt.unsqueeze(1).repeat(1, self.k_nearest_neighbour, 1) - zt_mean) ** 2) / (zt_std ** 2) / 2.0), 2)
        log_other_term = - (self.z_dim / 2.0) * torch.log(constant_value) - torch.sum(torch.log(zt_std), 1)
        return log_exp_term + log_other_term

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.randn_like(std, device = device)
        return eps.mul(std).add_(mean)


    def _reparameterized_sample_cluster(self, mean, std):
        """using std to sample"""
        if self.training:
            eps = torch.randn_like(mean.repeat(self.kl_samples, 1), device = device)
            return eps.mul(std).add_(mean)
        else:
            return mean

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x * torch.log(theta) + (1 - x) * torch.log(1 - theta))

    def _nll_gauss(self, x, mean):
        # n, _ = x.size()
        return torch.sum((x - mean) ** 2)