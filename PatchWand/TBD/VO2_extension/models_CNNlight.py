import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import numpy as np

import scipy
from scipy.fftpack import fft, ifft

import sys

import sys
sys.path.append('../') # add this line so Data and data are visible in this file
from setting import *

    
class MyAvgPool1dPadSame(nn.Module):
    """
    extend nn.AvgPool1d to support SAME padding
    """
    def __init__(self, kernel_size, stride):
        super(MyAvgPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)
    def forward(self, x):
        net = x
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left

        net = F.pad(net, (pad_left, pad_right), "reflect")

        net = self.avg_pool(net)

        return net
    
class InceptionBlock(nn.Module):

    def __init__(self,training_params, stride=2, input_channel=None):
        """
        Inputs:
            c_in - Number of input feature maps from the previous layers
            c_red - Dictionary with keys "3x3" and "5x5" specifying the output of the dimensionality reducing 1x1 convolutions
            c_out - Dictionary with keys "1x1", "3x3", "5x5", and "max"
            act_fn - Activation class constructor (e.g. nn.ReLU)
        """
        super().__init__()
        
        input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]
            
        channel_n = training_params['channel_n']
        
        k1 = training_params['kernels']['k1']
        k2 = training_params['kernels']['k2']
        k3 = training_params['kernels']['k3']
        
        # 1x1 convolution branch
        self.conv_k1x1 = nn.Sequential(
            nn.Conv1d(input_channel, channel_n, kernel_size=k1, stride=1, padding=math.floor(k1/2)),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.Conv1d(channel_n, channel_n, kernel_size=k1, stride=1, padding=math.floor(k1/2)),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            MyAvgPool1dPadSame(kernel_size=k1, stride=stride),
        )

        # 3x3 convolution branch
        self.conv_k2x1 = nn.Sequential(
            nn.Conv1d(input_channel, channel_n, kernel_size=k2, stride=1, padding=math.floor(k2/2)),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.Conv1d(channel_n, channel_n, kernel_size=k2, stride=1, padding=math.floor(k2/2)),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            MyAvgPool1dPadSame(kernel_size=k2, stride=stride),
        )

        # 5x5 convolution branch
        self.conv_k3x1 = nn.Sequential(
            nn.Conv1d(input_channel, channel_n, kernel_size=k3, stride=1, padding=math.floor(k3/2)),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.Conv1d(channel_n, channel_n, kernel_size=k3, stride=1, padding=math.floor(k3/2)),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            MyAvgPool1dPadSame(kernel_size=k3, stride=stride),
        )
        
        self.ch_pooling = torch.nn.Conv1d(channel_n*3, channel_n, 1) # out_channels->1 channel, kernel size = 1

    def forward(self, x):
        x_k1x1 = self.conv_k1x1(x)
        x_k2x1 = self.conv_k2x1(x)
        x_k3x1 = self.conv_k3x1(x)

        x_out = torch.cat([x_k1x1, x_k2x1, x_k3x1], dim=1)
        x_out = self.ch_pooling(x_out)
        
        return x_out
    

class FeatureExtractor_CNNlight(nn.Module):
    def __init__(self, training_params=None, input_channel=None):
        super(FeatureExtractor_CNNlight, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]
            
        channel_n = training_params['channel_n']
        stride = training_params['stride']

        self.n_block = training_params['n_block']
        in_ch = input_channel

        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            self.basicblock_list.append(InceptionBlock(training_params, stride=stride, input_channel=in_ch))
            in_ch = channel_n

#         self.layer1 =  InceptionBlock(training_params, stride=stride, input_channel=input_channel)
    
    
        self.ch_pooling = torch.nn.Conv1d(channel_n, 1, 1) # out_channels->1 channel, kernel size = 1
        self.basicblock_list.append(self.ch_pooling) 

#         self.layer2 =  InceptionBlock(training_params, stride=stride, input_channel=channel_n)
        last_layer_dim = input_dim
        for n in range(self.n_block):
            last_layer_dim = round(last_layer_dim/stride)

        self.last_layer_dim = last_layer_dim
    
        self.regressor_type = training_params['regressor_type']

        # prepare for FFT
        self.xf_dict = training_params['xf_dict']
        self.FS_Extracted = self.xf_dict['FS_Extracted']
        
#         xf = np.linspace(0.0, 1.0/2.0*self.FS_Extracted , self.last_layer_dim//2)*60
#         mask = (xf>=label_range_dict['HR_DL'][0]) & (xf<=label_range_dict['HR_DL'][1])
#         self.xf = xf
#         self.xf_masked = xf[mask]
# #         print(self.xf_masked)
#         self.mask = mask


        if self.regressor_type=='DominantFreqRegression':
            self.xf = self.xf_dict['xf']
            self.xf_masked = self.xf_dict['xf_masked']
            self.mask = self.xf_dict['mask']
    #         self.feature_out_dim = int(last_layer_dim * channel_n)
            self.feature_out_dim = int(self.mask.sum() * 1)
            
        elif self.regressor_type=='FFTRegression':
            self.xf = self.xf_dict['xf']
            self.xf_masked = self.xf_dict['xf_masked']
            self.mask = self.xf_dict['mask']
            self.feature_out_dim = int(self.mask.sum() * 1)
            self.bn1 = nn.BatchNorm1d(self.feature_out_dim)

        elif self.regressor_type=='CardioRespRegression':
            self.feature_out_dim = last_layer_dim
            self.bn1 = nn.BatchNorm1d(last_layer_dim)
        else:
            self.feature_out_dim = last_layer_dim

            # self.feature_out_dim = last_layer_dim

        self.channel_n = int(channel_n)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('feature_out_dim   :', self.feature_out_dim)

    def forward(self, x):   
        
        out = x.float()
        # print(out.size())

#         for i_block in range(self.n_block):
        for i_block in range(len(self.basicblock_list)):
            net = self.basicblock_list[i_block]
#             if self.verbose:
#                 print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            # print(out.size())

        # covert signal to spectral domain

#         if self.regressor_type=='DominantFreqRegression':
#             # FFT layer, mask it, reshape it, normalize feature to 1
#             out = torch.fft.fft(out) # compute fft over the last axis
#             out = 2.0/self.last_layer_dim  * (torch.abs(out[:,:,:self.last_layer_dim //2])**2) # normalize based on number of spectral feature
#             print('out 1', out.size())
#             out = out[:,:,self.mask]
#             print('out 2', out.size())

#             out = out.reshape(out.size(0), -1)
#             out = out / torch.sum(out, axis=1)[:,None]
#             print('out 3', out.size())

            
            
# #             out = torch.softmax(out, dim=1)

#         elif self.regressor_type=='FFTRegression':
#             # FFT layer, mask it, reshape it, normalize feature to 1
#             out = torch.fft.fft(out) # compute fft over the last axis
#             out = 2.0/self.last_layer_dim  * (torch.abs(out[:,:,:self.last_layer_dim //2])**2) # normalize based on number of spectral feature
#             out = out[:,:,self.mask]

#             out = out.reshape(out.size(0), -1)
#             out = self.bn1(out)

# #             out = out / torch.sum(out, axis=1)[:,None]
            
#         elif self.regressor_type=='CardioRespRegression':
#             out = out.reshape(out.size(0), -1)
#             out = self.bn1(out)

#             print(out.size())
#             sys.exit()



        debug = False
        if debug == True:
            # size of x is  torch.Size([2, 1, 1000])
            # size of out1 is  torch.Size([2, 4, 500])
            # size of out2 is  torch.Size([2, 4, 250])
            # size of out3 is  torch.Size([2, 4, 125])
            # size of out4 is  torch.Size([2, 4, 62])
            # size of out5 is  torch.Size([2, 4, 31])
            # size of out is  torch.Size([2, 124])
            print('-----------------------------')
            print('size of x is ', x.size())
#             print('size of out1 is ', out1.size())
#             print('size of out2 is ', out2.size())
#             print('size of out3 is ', out3.size())
#             print('size of out4 is ', out4.size())
#             print('size of out5 is ', out5.size())
            print('size of out is ', out.size())
            print('-----------------------------')
            sys.exit()

#         print(out.size())
        return out

