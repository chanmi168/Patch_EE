import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import sys


# KERNEL_SIZE = 5
class MyAvgPool1dPadSame(nn.Module):
    """
    extend nn.AvgPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyAvgPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 2
        self.avg_pool = torch.nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)
#         print(self.avg_pool)
    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        
#         print(net.size(), pad_left, pad_right)
#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = F.pad(net, (pad_left, pad_right), "reflect")

        net = self.avg_pool(net)

        return net
    

class FeatureExtractor_CNN(nn.Module):
    def __init__(self, training_params=None, input_channel=None):
        super(FeatureExtractor_CNN, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        if input_channel is None:
            input_channel = training_params['data_dimensions'][0]
            
        channel_n = training_params['channel_n']
        
        self.kernel_size = training_params['kernel_size']
        
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channel, channel_n, kernel_size=kernel_size1, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            MyAvgPool1dPadSame(kernel_size=self.kernel_size),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size1, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            MyAvgPool1dPadSame(kernel_size=self.kernel_size),
#             nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size1, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size1, stride=1, padding=2),
            nn.BatchNorm1d(channel_n),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
#         self.layer5 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size5, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
# #             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         self.layer6 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size6, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
# #             nn.MaxPool1d(kernel_size=2, stride=2)
#         )
#         self.layer7 = nn.Sequential(
#             nn.Conv1d(channel_n, channel_n, kernel_size=kernel_size, stride=1, padding=2),
#             nn.BatchNorm1d(channel_n),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2))

        cnn_layer1_dim = (input_dim+2*2-1*(kernel_size1-1)-1)+1
        pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

        cnn_layer2_dim = (pool_layer1_dim+2*2-1*(kernel_size1-1)-1)+1
        pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

        cnn_layer3_dim = (pool_layer2_dim+2*2-1*(kernel_size1-1)-1)+1
        pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

        cnn_layer4_dim = (pool_layer3_dim+2*2-1*(kernel_size1-1)-1)+1
        pool_layer4_dim = math.floor((cnn_layer4_dim-1*(2-1)-1)/2+1)

#         cnn_layer5_dim = (cnn_layer4_dim+2*2-1*(kernel_size5-1)-1)+1
#         cnn_layer6_dim = (cnn_layer5_dim+2*2-1*(kernel_size6-1)-1)+1
#         pool_layer5_dim = math.floor((cnn_layer5_dim-1*(2-1)-1)/2+1)
        
#         cnn_layer6_dim = (pool_layer5_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer6_dim = math.floor((cnn_layer6_dim-1*(2-1)-1)/2+1)
        
#         cnn_layer7_dim = (pool_layer6_dim+2*2-1*(kernel_size-1)-1)+1
#         pool_layer7_dim = math.floor((cnn_layer7_dim-1*(2-1)-1)/2+1)
        
        self.feature_out_dim = int(pool_layer4_dim*channel_n)
#         print(cnn_layer4_dim)
#         self.feature_out_dim = int(cnn_layer6_dim*channel_n)
        self.channel_n = int(channel_n)
        #       self.feature_out_dim = int(pool_layer2_dim*channel_n)
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #       print('FeatureExtractor_total_params:', pytorch_total_params)
        print('feature_out_dim:', self.feature_out_dim)

    def forward(self, x):        
        out = self.layer1(x.float())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

#         out = self.layer5(out)
#         out = self.layer6(out)

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
