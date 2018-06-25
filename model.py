import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
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

"""implementation of the Generative Temporal Models 
with Spatial Memory (GTM-SM) from https://arxiv.org/abs/1804.09401
"""

class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
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
    
def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)
        
use_cuda = torch.cuda.is_available()
if use_cuda:
    dtype = torch.cuda.FloatTensor
    dbooltype = torch.cuda.ByteTensor
else:
    dtype = torch.FloatTensor 
    dbooltype = torch.ByteTensor       

class GTM_SM(nn.Module):
    def __init__(self, x_dim = 8, a_dim = 5, s_dim = 2, z_dim = 16, observe_dim = 256, total_dim = 288, \
                 r_std = 0.001, k_nearest_neighbour = 5, delta = 0.0001, kl_samples = 40, batch_size = 1):
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
        
        #feature-extracting transformations
        #encoder
        #for zt
        self.enc_zt = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1),
            nn.Tanh(),
            nn.Conv2d(8, z_dim, kernel_size=4, stride=2),
            Flatten())
        self.enc_zt_mean = nn.Sequential(
            nn.Linear(4 * z_dim, z_dim))

        self.enc_zt_var = nn.Sequential(
            nn.Linear(4 * z_dim, z_dim),
            Exponent())

        #for st
        self.enc_st_matrix = nn.Sequential(
            nn.Linear(a_dim, s_dim, bias=False))

        self.enc_st_sigmoid = nn.Sequential(
            nn.Linear(s_dim, 5),
            nn.Tanh(),
            nn.Linear(5, s_dim),
            nn.Sigmoid())

        
        #decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 2 * 2 * 16),
            nn.Tanh(),
            Unflatten(-1, 16, 2, 2),
            nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = 4, stride = 2),
            nn.Tanh(),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 3, kernel_size = 3, stride = 1),
            nn.Sigmoid())

    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        '''
        action_one_hot_value        tensor  (self.batch_size, self.a_dim, self.total_dim)
        position                    np      (self.batch_size, self.s_dim, self.total_dim)
        action_selection            np      (self.batch_size, self.total_dim)
        st_observation_list         list    (self.observe_dim)(self.batch_size, self.s_dim)
        st_prediction_list          list    (self.total_dim - self.observe_dim)(self.batch_size, self.s_dim)
        zt_mean_observation_list    list    (self.observe_dim)(self.batch_size, self.z_dim)
        zt_var_observation_list     list    (self.observe_dim)(self.batch_size, self.z_dim)
        zt_mean_prediction_list     list    (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        zt_var_prediction_list      list    (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        x_resconstruct_t            list    (self.total_dim - self.observe_dim)(self.batch_size, self.x_dim)
        
        after construct them, we will use torch.cat to eliminate the list object
        
        st_observation_tensor       tensor      (self.observe_dim)(self.batch_size, self.s_dim)
        st_prediction_tensor        tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.s_dim)
        zt_mean_observation_tensor  tensor      (self.observe_dim)(self.batch_size, self.z_dim)
        zt_var_observation_tensor   tensor      (self.observe_dim)(self.batch_size, self.z_dim)
        zt_mean_prediction_tensor   tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        zt_var_prediction_tensor    tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        
        '''
        
        action_one_hot_value, position, action_selection = self.random_walk()
        st_observation_list = []
        st_prediction_list = []
        zt_mean_observation_list = []
        zt_var_observation_list = []
        zt_mean_prediction_list = []
        zt_var_prediction_list = []
        xt_prediction_list = []
        
        kld_loss = 0
        nll_loss = 0
        
        flanns = [pyflann.FLANN() for _ in range(self.batch_size)]

        #observation phase: construct st
        for t in range(self.observe_dim):
            if t == 0:
                st_observation_t = torch.rand(self.batch_size, self.s_dim).type(dtype) - 1
            else:
                st_observation_t = st_observation_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t]) * \
                                            self.enc_st_sigmoid(st_observation_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t])) + \
                                            torch.normal(mean=torch.zeros(self.batch_size, self.s_dim), std=self.r_std * torch.ones(self.batch_size, self.s_dim)).type(dtype)
            st_observation_list.append(st_observation_t)
        st_observation_tensor = torch.cat(st_observation_list, 0).view(self.observe_dim, self.batch_size, self.s_dim)

        
        #prediction phase: construct st        
        for t in range(self.total_dim - self.observe_dim):  
            if t == 0:
                st_prediction_t = st_observation_list[-1] + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim]) * \
                                            self.enc_st_sigmoid(st_observation_list[-1] + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim])) + \
                                            torch.normal(mean=torch.zeros(self.batch_size, self.s_dim), std=self.r_std * torch.ones(self.batch_size, self.s_dim)).type(dtype)
            else:
                st_prediction_t = st_prediction_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim]) * \
                                            self.enc_st_sigmoid(st_prediction_list[t - 1]  + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim])) + \
                                            torch.normal(mean=torch.zeros(self.batch_size, self.s_dim), std=self.r_std * torch.ones(self.batch_size, self.s_dim)).type(dtype)
            st_prediction_list.append(st_prediction_t)
        st_prediction_tensor = torch.cat(st_prediction_list, 0).view(self.total_dim - self.observe_dim, self.batch_size, self.s_dim) 

        
        #observation phase: construct zt from xt         
        for t in range(self.observe_dim):
            index_mask = torch.zeros((self.batch_size, 3, 32, 32)).type(dtype)
            for index_sample in range(self.batch_size):
                position_h_t = position[index_sample, 0, t]
                position_w_t = position[index_sample, 1, t]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,3 * position_w_t:3 * position_w_t + 8] = torch.ones([1]).type(dtype)
                index_mask_bool = index_mask.ge(0.5)
            x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            zt_observation_t = self.enc_zt(x_feed)
            zt_mean_observation_t = self.enc_zt_mean(zt_observation_t)
            zt_var_observation_t = self.enc_zt_var(zt_observation_t)
            zt_mean_observation_list.append(zt_mean_observation_t)
            zt_var_observation_list.append(zt_var_observation_t)
        zt_mean_observation_tensor = torch.cat(zt_mean_observation_list, 0).view(self.observe_dim, self.batch_size, self.z_dim) 
        zt_var_observation_tensor = torch.cat(zt_var_observation_list, 0).view(self.observe_dim, self.batch_size, self.z_dim) 
            
        #prediction phase: construct zt from xt
        for t in range(self.total_dim - self.observe_dim):
            index_mask = torch.zeros((self.batch_size, 3, 32, 32)).type(dtype)
            for index_sample in range(self.batch_size):
                position_h_t = position[index_sample, 0, t + self.observe_dim]
                position_w_t = position[index_sample, 1, t + self.observe_dim]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,3 * position_w_t:3 * position_w_t + 8] = torch.ones([1]).type(dtype)
                index_mask_bool = index_mask.ge(0.5)
            x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            zt_prediction_t = self.enc_zt(x_feed)
            zt_mean_prediction_t = self.enc_zt_mean(zt_prediction_t)
            zt_var_prediction_t = self.enc_zt_var(zt_prediction_t)
            zt_mean_prediction_list.append(zt_mean_prediction_t)
            zt_var_prediction_list.append(zt_var_prediction_t)
        zt_mean_prediction_tensor = torch.cat(zt_mean_prediction_list, 0).view(self.total_dim - self.observe_dim, self.batch_size, self.z_dim) 
        zt_var_prediction_tensor = torch.cat(zt_var_prediction_list, 0).view(self.total_dim - self.observe_dim, self.batch_size, self.z_dim)


        #reparameterized_sample to calculate the reconstruct error           
        for t in range(self.total_dim - self.observe_dim):
            zt_prediction_sample = self._reparameterized_sample(zt_mean_prediction_list[t], zt_var_prediction_list[t]) 
            index_mask = torch.zeros((self.batch_size, 3, 32, 32)).type(dtype)
            for index_sample in range(self.batch_size):
                position_h_t = position[index_sample, 0, t + self.observe_dim]
                position_w_t = position[index_sample, 1, t + self.observe_dim]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,3 * position_w_t:3 * position_w_t + 8] = torch.ones([1]).type(dtype)
                index_mask_bool = index_mask.ge(0.5)
            x_ground_true_t = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            x_resconstruct_t = self.dec(zt_prediction_sample)
            nll_loss += self._nll_gauss(x_resconstruct_t, x_ground_true_t)
            xt_prediction_list.append(x_resconstruct_t) 
            
        
        #construct kd tree
        st_observation_memory = np.zeros((self.observe_dim, self.batch_size, self.s_dim))
        for t in range(self.observe_dim):
            st_observation_memory[t] = st_observation_list[t].cpu().detach().numpy()
            
        st_prediction_memory = np.zeros((self.total_dim - self.observe_dim, self.batch_size, self.s_dim))
        for t in range(self.total_dim - self.observe_dim):
            st_prediction_memory[t] = st_prediction_list[t].cpu().detach().numpy()
        
        
        results = []
        for index_sample in range(self.batch_size):
            param = flanns[index_sample].build_index(st_observation_memory[:, index_sample, :], algorithm = 'kdtree', trees=4)
            result, _ = flanns[index_sample].nn_index(st_prediction_memory[:, index_sample, :], self.k_nearest_neighbour, checks=param["checks"])
            results.append(result)
        
        #calculate the kld
        for index_sample in range(self.batch_size):
            for t in range(self.total_dim - self.observe_dim):
                t_knn_index = results[index_sample][t]
                t_knn_st_memory = st_observation_tensor[t_knn_index, index_sample]
                dk2 = ((t_knn_st_memory - st_prediction_tensor[t, index_sample, :]) ** 2).sum(1)
                wk = 1 / (dk2 + self.delta)
                normalized_wk = wk / torch.sum(wk)
                
                #sampling
                zt_sampling = self._reparameterized_sample_cluster(zt_mean_prediction_tensor[t, index_sample], zt_var_prediction_tensor[t, index_sample])
                log_q_phi = self.log_gaussian_pdf(zt_sampling, zt_mean_prediction_tensor[t, index_sample], zt_var_prediction_tensor[t, index_sample])
                log_p_theta_element_minus_log_q_phi = (self.log_gaussian_element_pdf(zt_sampling, zt_mean_observation_tensor[t_knn_index, index_sample], zt_var_observation_tensor[t_knn_index, index_sample]).t() - log_q_phi).t()
                #print(log_p_theta_element_minus_log_q_phi)
                p_theta = torch.exp(log_p_theta_element_minus_log_q_phi).matmul(normalized_wk)
                kld_loss += - torch.mean(torch.log(p_theta))
                


        return kld_loss, nll_loss, st_observation_list, st_prediction_list, xt_prediction_list, position
    
    
    
    
    def Sampling(self, x):
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
        zt_var_observation_list     list    (self.observe_dim)(self.batch_size, self.z_dim)
        xt_prediction_list          list    (self.total_dim - self.observe_dim)(self.batch_size, self.x_dim)
        
        after construct them, we will use torch.cat to eliminate the list object
        
        st_observation_tensor       tensor      (self.observe_dim)(self.batch_size, self.s_dim)
        st_prediction_tensor        tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.s_dim)
        zt_mean_observation_tensor  tensor      (self.observe_dim)(self.batch_size, self.z_dim)
        zt_var_observation_tensor   tensor      (self.observe_dim)(self.batch_size, self.z_dim)
        zt_mean_prediction_tensor   tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        zt_var_prediction_tensor    tensor      (self.total_dim - self.observe_dim)(self.batch_size, self.z_dim)
        
        '''
        
        action_one_hot_value, position, action_selection = self.random_walk()
        st_observation_list = []
        st_prediction_list = []
        zt_mean_observation_list = []
        zt_var_observation_list = []
        xt_prediction_list = []
        
        
        flanns = [pyflann.FLANN() for _ in range(self.batch_size)]

        #observation phase: construct st
        for t in range(self.observe_dim):
            if t == 0:
                st_observation_t = torch.rand(self.batch_size, self.s_dim).type(dtype) - 1
            else:
                st_observation_t = st_observation_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t]) * \
                                            self.enc_st_sigmoid(st_observation_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t])) + \
                                            torch.normal(mean=torch.zeros(self.batch_size, self.s_dim), std=self.r_std * torch.ones(self.batch_size, self.s_dim)).type(dtype)
            st_observation_list.append(st_observation_t)
        st_observation_tensor = torch.cat(st_observation_list, 0).view(self.observe_dim, self.batch_size, self.s_dim)

        
        #prediction phase: construct st        
        for t in range(self.total_dim - self.observe_dim):  
            if t == 0:
                st_prediction_t = st_observation_list[-1] + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim]) * \
                                            self.enc_st_sigmoid(st_observation_list[-1] + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim])) + \
                                            torch.normal(mean=torch.zeros(self.batch_size, self.s_dim), std=self.r_std * torch.ones(self.batch_size, self.s_dim)).type(dtype)
            else:
                st_prediction_t = st_prediction_list[t - 1] + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim]) * \
                                            self.enc_st_sigmoid(st_prediction_list[t - 1]  + self.enc_st_matrix(action_one_hot_value[:, :, t + self.observe_dim])) + \
                                            torch.normal(mean=torch.zeros(self.batch_size, self.s_dim), std=self.r_std * torch.ones(self.batch_size, self.s_dim)).type(dtype)
            st_prediction_list.append(st_prediction_t)
        st_prediction_tensor = torch.cat(st_prediction_list, 0).view(self.total_dim - self.observe_dim, self.batch_size, self.s_dim) 

        
        #observation phase: construct zt from xt         
        for t in range(self.observe_dim):
            index_mask = torch.zeros((self.batch_size, 3, 32, 32)).type(dtype)
            for index_sample in range(self.batch_size):
                position_h_t = position[index_sample, 0, t]
                position_w_t = position[index_sample, 1, t]
                index_mask[index_sample, :, 3 * position_h_t:3 * position_h_t + 8,3 * position_w_t:3 * position_w_t + 8] = torch.ones([1]).type(dtype)
                index_mask_bool = index_mask.ge(0.5)
            x_feed = torch.masked_select(x, index_mask_bool).view(-1, 3, 8, 8)
            zt_observation_t = self.enc_zt(x_feed)
            zt_mean_observation_t = self.enc_zt_mean(zt_observation_t)
            zt_var_observation_t = self.enc_zt_var(zt_observation_t)
            zt_mean_observation_list.append(zt_mean_observation_t)
            zt_var_observation_list.append(zt_var_observation_t)
        zt_mean_observation_tensor = torch.cat(zt_mean_observation_list, 0).view(self.observe_dim, self.batch_size, self.z_dim) 
        zt_var_observation_tensor = torch.cat(zt_var_observation_list, 0).view(self.observe_dim, self.batch_size, self.z_dim) 
            
        #prediction phase: construct zt from prior
        #construct kd tree
        st_observation_memory = np.zeros((self.observe_dim, self.batch_size, self.s_dim))
        for t in range(self.observe_dim):
            st_observation_memory[t] = st_observation_list[t].cpu().detach().numpy()
            
        st_prediction_memory = np.zeros((self.total_dim - self.observe_dim, self.batch_size, self.s_dim))
        for t in range(self.total_dim - self.observe_dim):
            st_prediction_memory[t] = st_prediction_list[t].cpu().detach().numpy()
        
        results = []
        for index_sample in range(self.batch_size):
            param = flanns[index_sample].build_index(st_observation_memory[:, index_sample, :], algorithm = 'kdtree', trees=4)
            result, _ = flanns[index_sample].nn_index(st_prediction_memory[:, index_sample, :], self.k_nearest_neighbour, checks=param["checks"])
            results.append(result)
            
        xt_prediction_tensor = torch.zeros(self.total_dim - self.observe_dim, self.batch_size, 3, 8, 8).type(dtype)

        for index_sample in range(self.batch_size):
            for t in range(self.total_dim - self.observe_dim):
                t_knn_index = results[index_sample][t]
                t_knn_st_memory = st_observation_tensor[t_knn_index, index_sample]
                dk2 = ((t_knn_st_memory - st_prediction_tensor[t, index_sample, :]) ** 2).sum(1)
                wk = 1 / (dk2 + self.delta)
                normalized_wk = wk / torch.sum(wk)
                cumsum_normalized_wk = torch.cumsum(normalized_wk, dim=0)
                rand_sample_value = torch.rand(1).type(dtype)
                
                for sample_knn in range(self.k_nearest_neighbour):
                    if sample_knn == 0:
                        if cumsum_normalized_wk[sample_knn] > rand_sample_value:
                            break
                    else:
                        if cumsum_normalized_wk[sample_knn] > rand_sample_value and cumsum_normalized_wk[sample_knn - 1] <= rand_sample_value:
                             break  
                #sampling
                zt_sampling = self._reparameterized_sample(zt_mean_observation_tensor[t, index_sample], zt_var_observation_tensor[t, index_sample])
                xt_prediction_tensor[t, index_sample] = self.dec(zt_sampling)


        #reparameterized_sample to calculate the reconstruct error           
        for t in range(self.total_dim - self.observe_dim):
            xt_prediction_list.append(xt_prediction_tensor[t])
            
        self.total_dim = origin_total_dim
        
        return st_observation_list, st_prediction_list, xt_prediction_list, position


    def random_walk(self):    
    #construct position and action
        action_one_hot_value_numpy = np.zeros((self.batch_size, self.a_dim, self.total_dim))
        position = np.zeros((self.batch_size, self.s_dim, self.total_dim), np.int32)
        action_selection = np.zeros((self.batch_size, self.total_dim), np.int32)
        for index_sample in range(self.batch_size):
            new_continue_action_flag = True
            for t in range(self.total_dim):
                if t == 0:
                    position[index_sample, :, t] = np.random.randint(0, 9, size = (2))
                else:
                    if new_continue_action_flag:
                        new_continue_action_flag = False
                        need_to_stop = False
                        action_random_selection = np.random.randint(0, 4, size = (1))
                        action_duriation = np.random.poisson(2, 1)
                        
                    if action_duriation > 0 and not need_to_stop:
                        if action_random_selection == 0:
                            if position[index_sample, 1, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, 1])
                                action_selection[index_sample, t] = action_random_selection
                        elif action_random_selection == 1:  
                            if position[index_sample, 1, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([0, -1])
                                action_selection[index_sample, t] = action_random_selection
                        elif action_random_selection == 2:  
                            if position[index_sample, 0, t - 1] == 0:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([-1, 0])
                                action_selection[index_sample, t] = action_random_selection
                        else:
                            if position[index_sample, 0, t - 1] == 8:
                                need_to_stop = True
                                position[index_sample, :, t] = position[index_sample, :, t - 1]
                                action_selection[index_sample, t] = 4
                            else:
                                position[index_sample, :, t] = position[index_sample, :, t - 1] + np.array([1, 0]) 
                                action_selection[index_sample, t] = action_random_selection                            
                    else:
                        action_selection[index_sample, t] = 4
                        position[index_sample, :, t] = position[index_sample, :, t - 1]
                    action_duriation -= 1
                    if action_duriation <= 0:
                        new_continue_action_flag = True
                    
        for index_sample in range(self.batch_size):
            action_one_hot_value_numpy[index_sample, action_selection[index_sample], np.array(range(self.total_dim))] = 1
            
        action_one_hot_value = Variable(torch.from_numpy(action_one_hot_value_numpy)).type(dtype)

        return action_one_hot_value, position, action_selection
                        
    
    def show_experiment_information(self, x, st_observation_list, st_prediction_list, xt_prediction_list, position, training = False):
        if not training:
            origin_total_dim = self.total_dim
            self.total_dim = 512
            
                
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        sample_id  = np.random.randint(0, self.batch_size, size = (1))
        sample_imgs = x[sample_id]
        
        st_observation_sample = np.zeros((self.observe_dim, self.s_dim))
        for t in range(self.observe_dim):
            st_observation_sample[t] = st_observation_list[t][sample_id].cpu().detach().numpy()
            
        st_prediction_sample = np.zeros((self.total_dim - self.observe_dim, self.s_dim))   
        for t in range(self.total_dim - self.observe_dim):
            st_prediction_sample[t] = st_prediction_list[t][sample_id].cpu().detach().numpy()
            
        st_2_max = np.maximum(np.max(st_observation_sample[:, 0]), np.max(st_prediction_sample[:, 0]))
        st_2_min = np.minimum(np.min(st_observation_sample[:, 0]), np.min(st_prediction_sample[:, 0]))
        st_1_max = np.maximum(np.max(st_observation_sample[:, 1]), np.max(st_prediction_sample[:, 1]))
        st_1_min = np.minimum(np.min(st_observation_sample[:, 1]), np.min(st_prediction_sample[:, 1]))
        axis_st_1_max = st_1_max + (st_1_max - st_1_min) / 10.0
        axis_st_1_min = st_1_min - (st_1_max - st_1_min) / 10.0
        axis_st_2_max = st_2_max + (st_2_max - st_2_min) / 10.0
        axis_st_2_min = st_2_min - (st_2_max - st_2_min) / 10.0

        fig = plt.figure()
        #interaction mode
        plt.ion()
        
        #observation phase
        for t in range(240, self.observe_dim):
            position_h_t = np.asscalar(position[sample_id, 0, t])
            position_w_t = np.asscalar(position[sample_id, 1, t])
            sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())
            observed_img = np.copy(sample_imgs[:, :, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t : 3 * position_w_t + 8].cpu().detach().numpy())

            sample_imgs_t[0, 0, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t ] = 1.0
            sample_imgs_t[0, 0, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
            sample_imgs_t[0, 0, 3 * position_h_t , 3 * position_w_t : 3 * position_w_t + 8] = 1.0
            sample_imgs_t[0, 0, 3 * position_h_t + 8 - 1, 3 * position_w_t : 3 * position_w_t + 8] = 1.0
            sample_imgs_t[0, 1, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t] = 0.0
            sample_imgs_t[0, 1, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
            sample_imgs_t[0, 1, 3 * position_h_t , 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            sample_imgs_t[0, 1, 3 * position_h_t + 8 - 1, 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t, 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t + 8 - 1, 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            
            fig.clf()
            
            plt.suptitle('t = ' + str(t) +'\n' + 'OBSERVATION PHASE', fontsize = 25) 
            
            gs = gridspec.GridSpec(20, 20)
          
            #subfigure 1
            #ax1 = fig.add_subplot(221)
            ax1 = plt.subplot(gs[1:10, 1:10])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            plt.axis('off')
            plt.imshow(sample_imgs_t.reshape([3, 32, 32]).transpose((1, 2, 0)))
            
            #subfigure 2
            ax2 = plt.subplot(gs[4:8, 11:15])
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_aspect('equal')
            ax2.set_title('Observation')
            plt.axis('off')
            plt.imshow(observed_img.reshape([3, 8, 8]).transpose((1, 2, 0)))

            #subfigure 3

            #subfigure 4
            ax4 = plt.subplot(gs[11:20, 1:10])
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            ax4.set_title('True states')
            ax4.set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.axis([-1, 9, -1, 9])
            plt.plot(position[sample_id, 1, 0 : t + 1].T, position[sample_id, 0, 0 : t + 1].T, color = 'k', linestyle = 'solid', marker = 'o')
            plt.plot(position[sample_id, 1, t], position[sample_id, 0, t], 'bs')
                       
            #subfigure 5
            ax5 = plt.subplot(gs[11:20, 11:20])
            ax5.set_xlabel('$s_1$')
            ax5.set_ylabel('$s_2$')
            ax5.set_title('Inferred states')
            #ax4.set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])            
            plt.plot(st_observation_sample[0 : t + 1, 1], st_observation_sample[0 : t + 1, 0], color = 'k', linestyle = 'solid', marker = 'o')
            plt.plot(st_observation_sample[t, 1], st_observation_sample[t, 0], 'bs')
            
            plt.pause(0.05)  
            
            
        #predition phase    
        for t in range(self.total_dim - self.observe_dim):
            position_h_t = np.asscalar(position[sample_id, 0, t + self.observe_dim])
            position_w_t = np.asscalar(position[sample_id, 1, t + self.observe_dim])
            sample_imgs_t = np.copy(sample_imgs.cpu().detach().numpy())
            observed_img = np.copy(sample_imgs[:, :, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t : 3 * position_w_t + 8].cpu().detach().numpy())
            predict_img = xt_prediction_list[t][np.asscalar(sample_id)].cpu().detach().numpy()
            
            sample_imgs_t[0, 0, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t ] = 1.0
            sample_imgs_t[0, 0, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 1.0
            sample_imgs_t[0, 0, 3 * position_h_t , 3 * position_w_t : 3 * position_w_t + 8] = 1.0
            sample_imgs_t[0, 0, 3 * position_h_t + 8 - 1, 3 * position_w_t : 3 * position_w_t + 8] = 1.0
            sample_imgs_t[0, 1, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t] = 0.0
            sample_imgs_t[0, 1, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
            sample_imgs_t[0, 1, 3 * position_h_t , 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            sample_imgs_t[0, 1, 3 * position_h_t + 8 - 1, 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t : 3 * position_h_t + 8, 3 * position_w_t + 8 - 1] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t, 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            sample_imgs_t[0, 2, 3 * position_h_t + 8 - 1, 3 * position_w_t : 3 * position_w_t + 8] = 0.0
            
            fig.clf()
            
            plt.suptitle('t = ' + str(t + self.observe_dim) +'\n' + 'PREDICTION PHASE', fontsize = 25) 
            
            gs = gridspec.GridSpec(20, 20)
            '''
            ax1 = plt.subplot(gs[0:4, 0:4])
            ax2 = plt.subplot(gs[1:3, 4:6])
            ax3 = plt.subplot(gs[1:3, 6:8])
            ax4 = plt.subplot(gs[4:8, 0:4])
            ax5 = plt.subplot(gs[4:8, 4:8])
            '''            
            #subfigure 1
            #ax1 = fig.add_subplot(221)
            ax1 = plt.subplot(gs[1:10, 1:10])
            ax1.set_xticklabels([])
            ax1.set_yticklabels([])
            ax1.set_aspect('equal')
            plt.axis('off')
            plt.imshow(sample_imgs_t.reshape([3, 32, 32]).transpose((1, 2, 0)))
            
            #subfigure 2
            ax2 = plt.subplot(gs[4:8, 11:15])
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.set_aspect('equal')
            ax2.set_title('Observation')
            plt.axis('off')
            plt.imshow(observed_img.reshape([3, 8, 8]).transpose((1, 2, 0)))

            #subfigure 3
            ax3 = plt.subplot(gs[4:8, 16:20])
            ax3.set_xticklabels([])
            ax3.set_yticklabels([])
            ax3.set_aspect('equal')
            ax3.set_title('Prediction')
            plt.axis('off')
            plt.imshow(predict_img.reshape([3, 8, 8]).transpose((1, 2, 0)))

            #subfigure 4
            ax4 = plt.subplot(gs[11:20, 1:10])
            ax4.set_xlabel('x')
            ax4.set_ylabel('y')
            ax4.set_title('True states')
            ax4.set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.axis([-1, 9, -1, 9])
            plt.plot(position[sample_id, 1, 0 : self.observe_dim + 1].T, position[sample_id, 0, 0 : self.observe_dim + 1].T, color = 'k', linestyle = 'solid', marker = 'o')
            plt.plot(position[sample_id, 1, t + self.observe_dim], position[sample_id, 0, t + self.observe_dim], 'bs')
                       
            #subfigure 5
            ax5 = plt.subplot(gs[11:20, 11:20])
            ax5.set_xlabel('$s_1$')
            ax5.set_ylabel('$s_2$')
            ax5.set_title('Inferred states')
            #ax4.set_aspect('equal')
            plt.gca().invert_yaxis()
            plt.axis([axis_st_1_min, axis_st_1_max, axis_st_2_min, axis_st_2_max])        
            plt.plot(st_observation_sample[:, 1], st_observation_sample[:, 0], color = 'k', linestyle = 'solid', marker = 'o')
            plt.plot(st_prediction_sample[t, 1], st_prediction_sample[t, 0], 'bs')
            
            plt.pause(0.01)                
        

        #show figure
        plt.show()
        
        #close figure
        plt.close(fig)
        
        #close interaction mode
        plt.ioff()
        
        if not training:
            self.total_dim = origin_total_dim
        
                   
    def multi_gaussian_pdf(self, zt, zt_mean, zt_var):
        exp_term = torch.exp(( - ((zt - zt_mean.transpose(1, 0)) ** 2) / zt_var.transpose(1, 0)/ 2 ).sum(1))
        other_term = (1 / (2 * 3.1415926535)) **(self.z_dim / 2.0) /torch.sqrt((torch.prod(zt_var, 0)))
        return exp_term*other_term
            
    
    def gaussian_pdf(self, zt, zt_mean, zt_var):
        exp_term = torch.exp(- torch.sum((((zt - zt_mean) ** 2 ) / zt_var/ 2.0)))
        other_term = (1 / (2 * 3.1415926535))**(self.z_dim / 2.0) /torch.sqrt(torch.prod(zt_var))
        return exp_term*other_term
    
    def log_gaussian_pdf(self, zt, zt_mean, zt_var):
        constant_value = torch.tensor(2 * 3.1415926535).type(dtype)
        log_exp_term = - torch.sum((((zt - zt_mean) ** 2 ) / zt_var/ 2.0), 1)
        log_other_term = - (self.z_dim / 2.0) * torch.log(constant_value) - 0.5 * torch.sum(torch.log(zt_var))
        return log_exp_term + log_other_term

    def log_gaussian_element_pdf(self, zt, zt_mean, zt_var):
        constant_value = torch.tensor(2 * 3.1415926535).type(dtype)
        log_exp_term = - torch.sum((((zt.unsqueeze(1).repeat(1, self.k_nearest_neighbour, 1) - zt_mean) ** 2 ) / zt_var/ 2.0), 2)
        log_other_term = - (self.z_dim / 2.0) * torch.log(constant_value) - 0.5 * torch.sum(torch.log(zt_var), 1)
        return log_exp_term + log_other_term
        
    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, var):
        """using std to sample"""
        #eps = torch.normal(mean, torch.sqrt(var)).type(dtype)
        eps = torch.randn(mean.shape).type(dtype) * torch.sqrt(var) + mean
        return eps
    
    def _reparameterized_sample_cluster(self, mean, var):
        """using std to sample"""
        eps = torch.randn(mean.repeat(self.kl_samples, 1).shape).type(dtype) * torch.sqrt(var) + mean
        return eps

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))


    def _nll_gauss(self, x, mean):
        #n, _ = x.size()
        return torch.sum((x - mean)**2)