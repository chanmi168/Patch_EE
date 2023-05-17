import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = F.pad(net, (pad_left, pad_right), "reflect")

        
        net = self.conv(net)

        return net
    

       
# class MFCConv1dPad(nn.Module):
#     """
#     extend nn.Conv1d to support Reflect padding
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
#         super(MyConv1dPadReflect, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.groups = groups
#         self.conv = torch.nn.Conv1d(
#             in_channels=self.in_channels, 
#             out_channels=self.out_channels, 
#             kernel_size=self.kernel_size, 
#             stride=self.stride, 
#             groups=self.groups,
#             padding_mode='reflect')

#     def forward(self, x):
        
#         net = x
        
# #         # compute pad shape
# #         in_dim = net.shape[-1]
# #         out_dim = (in_dim + self.stride - 1) // self.stride
# #         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
# #         pad_left = p // 2
# #         pad_right = p - pad_left
# # #         net = F.pad(net, (pad_left, pad_right), "constant", 0)
# #         net = F.pad(net, (pad_left, pad_right), "reflect")

#         net = self.conv(net)

#         return net
    
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size, stride):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left

#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = F.pad(net, (pad_left, pad_right), "reflect")
        net = self.max_pool(net)

        return net
    
class MyAvgPool1dPadSame(nn.Module):
    """
    extend nn.AvgPool1d to support SAME padding
    """
    def __init__(self, kernel_size, stride):
        super(MyAvgPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
#         self.stride = 2
        self.stride = stride
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
    
# class MyAvgPool1dPadReflect(nn.Module):
#     """
#     extend nn.AvgPool1d to support Reflect padding
#     """
#     def __init__(self, kernel_size):
#         super(MyAvgPool1dPadReflect, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = 2
#         self.avg_pool = torch.nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)
# #         print(self.avg_pool)
#     def forward(self, x):
        
#         net = x
        
#         # compute pad shape
#         in_dim = net.shape[-1]
#         out_dim = (in_dim + self.stride - 1) // self.stride
#         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
#         pad_left = p // 2
#         pad_right = p - pad_left
        
# #         print(net.size(), pad_left, pad_right)
# #         net = F.pad(net, (pad_left, pad_right), "constant", 0)
#         net = F.pad(net, (pad_left, pad_right), "reflect")
#         net = self.avg_pool(net)

#         return net
    
# class MyAvgPool1dPadSame(nn.Module):
#     """
#     extend nn.AvgPool1d to support SAME padding
#     """
#     def __init__(self, kernel_size):
#         super(MyAvgPool1dPadSame, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = 1
#         self.avg_pool = torch.nn.AvgPool1d(kernel_size=self.kernel_size)
#         print(self.avg_pool)
#     def forward(self, x):
        
#         net = x
        
#         # compute pad shape
#         in_dim = net.shape[-1]
#         out_dim = (in_dim + self.stride - 1) // self.stride
#         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
#         pad_left = p // 2
#         pad_right = p - pad_left
        
#         print(net.size(), pad_left, pad_right)
#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
#         net = self.avg_pool(net)

#         return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, use_sc, pooling_type, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_sc = use_sc

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
#         self.conv1 = MyConv1dPadReflect(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)
    


        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
#         self.conv2 = MyConv1dPadReflect(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)

#         self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

#         print(pooling_type)
#         sys.exit()
        if pooling_type == 'max_pooling':
            self.pooling = MyMaxPool1dPadSame(kernel_size=self.kernel_size, stride=self.stride)
#             self.pooling = MyMaxPool1dPadSame(kernel_size=self.kernel_size)
        elif pooling_type == 'avg_pooling':
            self.pooling = MyAvgPool1dPadSame(kernel_size=self.kernel_size, stride=self.stride)
#             self.pooling = MyAvgPool1dPadSame(kernel_size=self.kernel_size)
#             self.pooling = MyAvgPool1dPadSame(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
#         print(x.size())

        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
#             identity = self.max_psool(identity)
#             print(identity.size())
            identity = self.pooling(identity)
#             print(identity.size(), out.size())
#             sys.exit()

            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        if self.use_sc:
            out += identity

#         print(out.size())
        return out
    
    

    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

#     def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, use_sc=True, verbose=False):
    def __init__(self, training_params, input_channel=None, use_bn=True, use_do=True, use_sc=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        

        training_params['n_block'] = training_params['n_block_macro'] * training_params['downsample_gap']
        training_params['increasefilter_gap'] = training_params['downsample_gap']
        
        self.verbose = verbose
        self.n_block = training_params['n_block']
        self.kernel_size = training_params['kernel_size']
        self.stride = training_params['stride']
        self.groups = training_params['groups']
        self.use_bn = use_bn
        self.use_do = use_do
        self.use_sc = use_sc
        
        if input_channel is None:
            self.in_channels = training_params['data_dimensions'][0]
        else:
            self.in_channels = input_channel
            
#         self.in_channels = training_params['in_channels']
#         self.n_classes = training_params['n_classes'][0]
        self.n_classes = len(training_params['output_names'])

        self.base_filters = training_params['channel_n']

        self.downsample_gap = training_params['downsample_gap'] # 2 for base model
        self.increasefilter_gap = training_params['increasefilter_gap'] # 4 for base model

        self.pooling_type =  training_params['pooling_type']
        self.pad_type =  training_params['pad_type']
        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=self.in_channels, out_channels=self.base_filters, kernel_size=self.kernel_size, stride=1)

#         if self.pad_type=='same':
#             self.first_block_conv = MyConv1dPadSame(in_channels=self.in_channels, out_channels=self.base_filters, kernel_size=self.kernel_size, stride=1)
#         if self.pad_type=='reflect':
#             self.first_block_conv = MyConv1dPadReflect(in_channels=self.in_channels, out_channels=self.base_filters, kernel_size=self.kernel_size, stride=1)
        
#         self.first_block_pooling = MyMaxPool1dPadSame(kernel_size=self.kernel_size)

#         if self.pooling_type == 'max_pooling':
#             self.first_block_pooling = MyMaxPool1dPadSame(kernel_size=self.kernel_size)
#         elif self.pooling_type == 'avg_pooling':
#             self.pooling = MyAvgPool1dPadSame(kernel_size=self.stride)

#             if self.pad_type=='same':
#                 self.first_block_pooling = MyAvgPool1dPadSame(kernel_size=self.kernel_size)
#             if self.pad_type=='reflect':
#                 self.first_block_pooling = MyAvgPool1dPadReflect(kernel_size=self.kernel_size)

#             self.pooling = MyAvgPool1dPadSame(kernel_size=self.kernel_size, stride=self.stride)
        
        self.first_block_bn = nn.BatchNorm1d(self.base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = self.base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
#             if i_block % self.downsample_gap == 1:
            if (i_block % self.downsample_gap == 0) and (i_block != 0):
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = self.base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(self.base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
                    
#             print(i_block, downsample, in_channels, out_channels)
            
#             print(in_channels, self.groups)
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                use_sc = self.use_sc,
                pooling_type = self.pooling_type,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)
        
        # compute output dimension
        input_dim = training_params['data_dimensions'][1]

        output_dim = input_dim
#         for i_macro in range(training_params['n_block_macro']):
        for i_macro in range(training_params['n_block_macro']-1):
            output_dim = np.ceil(output_dim/self.stride)

        output_dim = int(output_dim)

        # final prediction
#         self.final_bn = nn.BatchNorm1d(output_dim)
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        self.do = nn.Dropout(p=0.5)
#         print(out_channels, self.n_classes)
    
        # self.softmax = nn.Softmax(dim=1)
        
        

#         print(out_channels, output_dim)

#         self.final_ch_pooling = torch.nn.Conv1d(out_channels, 1, 1) # out_channels->1 channel, kernel size = 1                                                                                                                                                                           

        self.channel_out = out_channels

#         self.feature_out_dim = output_dim*out_channels
        self.feature_out_dim = output_dim

#         self.dense = nn.Linear(self.feature_out_dim, 100)


#         self.feature_out_dim = 100
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        
#         out = self.first_block_pooling(out)
        
        
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

                
#         # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.do(out)
        
        
#         print(out.size())
#         print(out.size())
#         sys.exit()

        return out
#         # reduce the number of channels to 1
#         out = self.final_ch_pooling(out)
#         out = torch.squeeze(out, 1) # out dim = (batch_size, N_feature)




# #         print(out.size())
# #         print(out.mean(-1).size())
# #         print( out.reshape(out.size(0), -1).size())
# #         out = out.reshape(out.size(0), -1)
# #         out = out.mean(-2)

# #         print(out.size())

# # #         print(out.size())
# #         sys.exit()

#         if self.verbose:
#             print('final pooling', out.shape)
        
# #         out = self.do(out)
# #         out_final = 0
# # #         print(self.dense, out.size())
# #         for i_ch in range(out.size()[1]):
# #             out_final += self.dense(out[:,i_ch, :])
        
# #         out_final = out_final/out.size()[1]
# #         return out_final    

# #         out = self.dense(out)
# #         if self.verbose:
# #             print('dense', out.shape)
        
#         return out    
