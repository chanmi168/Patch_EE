import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import sys


# KERNEL_SIZE = 5


class FeatureExtractor_CNN2(nn.Module):
    def __init__(self, training_params=None, input_channel=None):
        super(FeatureExtractor_CNN2, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]
            
        channel_n = training_params['channel_n']
        kernel_size = training_params['kernel_size'][0]
        
        kernel_scale = 2.5

        kernel_size1 = int(kernel_size*(kernel_scale**0))
        kernel_size2 = int(kernel_size*(kernel_scale**1))
        kernel_size3 = int(kernel_size*(kernel_scale**2))
        kernel_size4 = int(kernel_size*(kernel_scale**3))
        kernel_size5 = int(kernel_size*(kernel_scale**4))
        kernel_size6 = int(kernel_size*(kernel_scale**5))
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channel, channel_n, kernel_size=kernel_size1, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size2, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size3, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size4, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size5, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size6, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
#         self.layer7 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))

        cnn_layer1_dim = (input_dim+2*2-1*(kernel_size1-1)-1)+1
#         pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

        cnn_layer2_dim = (cnn_layer1_dim+2*2-1*(kernel_size2-1)-1)+1
#         pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

        cnn_layer3_dim = (cnn_layer2_dim+2*2-1*(kernel_size3-1)-1)+1
#         pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

        cnn_layer4_dim = (cnn_layer3_dim+2*2-1*(kernel_size4-1)-1)+1
#         pool_layer4_dim = math.floor((cnn_layer4_dim-1*(2-1)-1)/2+1)

        cnn_layer5_dim = (cnn_layer4_dim+2*2-1*(kernel_size5-1)-1)+1
#         cnn_layer6_dim = (cnn_layer5_dim+2*2-1*(kernel_size6-1)-1)+1
#         pool_layer5_dim = math.floor((cnn_layer5_dim-1*(2-1)-1)/2+1)
        
#         cnn_layer6_dim = (pool_layer5_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer6_dim = math.floor((cnn_layer6_dim-1*(2-1)-1)/2+1)
        
#         cnn_layer7_dim = (pool_layer6_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer7_dim = math.floor((cnn_layer7_dim-1*(2-1)-1)/2+1)
        
#         self.feature_out_dim = int(pool_layer5_dim*channel_n)
#         print(cnn_layer4_dim)
        self.feature_out_dim = int(cnn_layer5_dim*channel_n)
        self.channel_n = int(channel_n)
        #       self.feature_out_dim = int(pool_layer2_dim*channel_n)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #       print('FeatureExtractor_total_params:', pytorch_total_params)
        print('feature_out_dim:', self.feature_out_dim)

    def forward(self, x):
#         print(x.size())
#         x = x[:,None,:]
#         print(x.size())

#         print(x.size())
        
        out = self.layer1(x.float())
#         print(out.size())
        out = self.layer2(out)
#         print(out.size())
        out = self.layer3(out)
        
#         print(out.size())

        out = self.layer4(out)
        
#         print(out.size())

        out = self.layer5(out)
#         out = self.layer6(out)
#         out = self.layer7(out)
#         print(out3.size())

        out = out.reshape(out.size(0), -1)
        
        

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

