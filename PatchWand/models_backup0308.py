# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import math
# import numpy as np
# import sys

# class MyConv1dPadSame(nn.Module):
#     """
#     extend nn.Conv1d to support SAME padding
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
#         super(MyConv1dPadSame, self).__init__()
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
#             groups=self.groups)

#     def forward(self, x):
        
#         net = x
        
#         # compute pad shape
#         in_dim = net.shape[-1]
#         out_dim = (in_dim + self.stride - 1) // self.stride
#         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
#         pad_left = p // 2
#         pad_right = p - pad_left
#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
#         net = self.conv(net)

#         return net
        
# class MyMaxPool1dPadSame(nn.Module):
#     """
#     extend nn.MaxPool1d to support SAME padding
#     """
#     def __init__(self, kernel_size):
#         super(MyMaxPool1dPadSame, self).__init__()
#         self.kernel_size = kernel_size
#         self.stride = 1
#         self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

#     def forward(self, x):
        
#         net = x
        
#         # compute pad shape
#         in_dim = net.shape[-1]
#         out_dim = (in_dim + self.stride - 1) // self.stride
#         p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
#         pad_left = p // 2
#         pad_right = p - pad_left

#         net = F.pad(net, (pad_left, pad_right), "constant", 0)
#         net = self.max_pool(net)

#         return net
    
# class BasicBlock(nn.Module):
#     """
#     ResNet Basic Block
#     """
#     def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, use_sc, is_first_block=False):
#         super(BasicBlock, self).__init__()
        
#         self.in_channels = in_channels
#         self.kernel_size = kernel_size
#         self.out_channels = out_channels
#         self.stride = stride
#         self.groups = groups
#         self.downsample = downsample
#         if self.downsample:
#             self.stride = stride
#         else:
#             self.stride = 1
#         self.is_first_block = is_first_block
#         self.use_bn = use_bn
#         self.use_do = use_do
#         self.use_sc = use_sc

#         # the first conv
#         self.bn1 = nn.BatchNorm1d(in_channels)
#         self.relu1 = nn.ReLU()
#         self.do1 = nn.Dropout(p=0.5)
#         self.conv1 = MyConv1dPadSame(
#             in_channels=in_channels, 
#             out_channels=out_channels, 
#             kernel_size=kernel_size, 
#             stride=self.stride,
#             groups=self.groups)

#         # the second conv
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.relu2 = nn.ReLU()
#         self.do2 = nn.Dropout(p=0.5)
#         self.conv2 = MyConv1dPadSame(
#             in_channels=out_channels, 
#             out_channels=out_channels, 
#             kernel_size=kernel_size, 
#             stride=1,
#             groups=self.groups)
                
#         self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

#     def forward(self, x):
        
#         identity = x
        
#         # the first conv
#         out = x
#         if not self.is_first_block:
#             if self.use_bn:
#                 out = self.bn1(out)
#             out = self.relu1(out)
#             if self.use_do:
#                 out = self.do1(out)
#         out = self.conv1(out)
        
#         # the second conv
#         if self.use_bn:
#             out = self.bn2(out)
#         out = self.relu2(out)
#         if self.use_do:
#             out = self.do2(out)
#         out = self.conv2(out)
        
#         # if downsample, also downsample identity
#         if self.downsample:
#             identity = self.max_pool(identity)
            
#         # if expand channel, also pad zeros to identity
#         if self.out_channels != self.in_channels:
#             identity = identity.transpose(-1,-2)
#             ch1 = (self.out_channels-self.in_channels)//2
#             ch2 = self.out_channels-self.in_channels-ch1
#             identity = F.pad(identity, (ch1, ch2), "constant", 0)
#             identity = identity.transpose(-1,-2)
        
#         # shortcut
#         if self.use_sc:
#             out += identity

#         return out
    
    

    
# class ResNet1D(nn.Module):
#     """
    
#     Input:
#         X: (n_samples, n_channel, n_length)
#         Y: (n_samples)
        
#     Output:
#         out: (n_samples)
        
#     Pararmetes:
#         in_channels: dim of input, the same as n_channel
#         base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
#         kernel_size: width of kernel
#         stride: stride of kernel moving
#         groups: set larget to 1 as ResNeXt
#         n_block: number of blocks
#         n_classes: number of classes
        
#     """

# #     def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, use_sc=True, verbose=False):
#     def __init__(self, training_params, use_bn=True, use_do=True, use_sc=True, verbose=False):
#         super(ResNet1D, self).__init__()
        
#         self.verbose = verbose
#         self.n_block = training_params['n_block'][0]
#         self.kernel_size = training_params['kernel_size'][0]
#         self.stride = training_params['stride'][0]
#         self.groups = training_params['groups'][0]
#         self.use_bn = use_bn
#         self.use_do = use_do
#         self.use_sc = use_sc
#         self.in_channels = training_params['in_channels']
#         self.n_classes = training_params['n_classes'][0]
#         self.base_filters = training_params['base_filters'][0]

#         self.downsample_gap = training_params['downsample_gap'][0] # 2 for base model
#         self.increasefilter_gap = training_params['increasefilter_gap'][0] # 4 for base model

#         # first block
#         self.first_block_conv = MyConv1dPadSame(in_channels=self.in_channels, out_channels=self.base_filters, kernel_size=self.kernel_size, stride=1)
#         self.first_block_bn = nn.BatchNorm1d(self.base_filters)
#         self.first_block_relu = nn.ReLU()
#         out_channels = self.base_filters
                
#         # residual blocks
#         self.basicblock_list = nn.ModuleList()
#         for i_block in range(self.n_block):
#             # is_first_block
#             if i_block == 0:
#                 is_first_block = True
#             else:
#                 is_first_block = False
#             # downsample at every self.downsample_gap blocks
#             if i_block % self.downsample_gap == 1:
#                 downsample = True
#             else:
#                 downsample = False
#             # in_channels and out_channels
#             if is_first_block:
#                 in_channels = self.base_filters
#                 out_channels = in_channels
#             else:
#                 # increase filters at every self.increasefilter_gap blocks
#                 in_channels = int(self.base_filters*2**((i_block-1)//self.increasefilter_gap))
#                 if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
#                     out_channels = in_channels * 2
#                 else:
#                     out_channels = in_channels
            
# #             print(in_channels, self.groups)
#             tmp_block = BasicBlock(
#                 in_channels=in_channels, 
#                 out_channels=out_channels, 
#                 kernel_size=self.kernel_size, 
#                 stride = self.stride, 
#                 groups = self.groups, 
#                 downsample=downsample, 
#                 use_bn = self.use_bn, 
#                 use_do = self.use_do, 
#                 use_sc = self.use_sc,
#                 is_first_block=is_first_block)
#             self.basicblock_list.append(tmp_block)

#         # final prediction
#         self.final_bn = nn.BatchNorm1d(out_channels)
#         self.final_relu = nn.ReLU(inplace=True)
#         # self.do = nn.Dropout(p=0.5)
# #         print(out_channels, self.n_classes)
#         self.dense = nn.Linear(out_channels, self.n_classes)
#         # self.softmax = nn.Softmax(dim=1)
        
        
#         input_dim = training_params['data_dimensions'][1]

#         output_dim = input_dim
#         for i_macro in range(training_params['n_block_macro'][0]):
#             output_dim = np.ceil(output_dim/2)

#         output_dim = int(output_dim)
#         self.feature_out_dim = output_dim*out_channels
        
#     def forward(self, x):
        
#         out = x
        
#         # first conv
#         if self.verbose:
#             print('input shape', out.shape)
#         out = self.first_block_conv(out)
#         if self.verbose:
#             print('after first conv', out.shape)
#         if self.use_bn:
#             out = self.first_block_bn(out)
#         out = self.first_block_relu(out)
        
#         # residual blocks, every block has two conv
#         for i_block in range(self.n_block):
#             net = self.basicblock_list[i_block]
#             if self.verbose:
#                 print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
#             out = net(out)
#             if self.verbose:
#                 print(out.shape)

# #         print(out.size())
#         out = out.reshape(out.size(0), -1)
# #         print(out.size())

# #         # final prediction
# #         if self.use_bn:
# #             out = self.final_bn(out)
# #         out = self.final_relu(out)
# #         out = out.mean(-1)
# #         if self.verbose:
# #             print('final pooling', out.shape)
# #         # out = self.do(out)
# #         out = self.dense(out)
# #         if self.verbose:
# #             print('dense', out.shape)
# #         # out = self.softmax(out)

# #         if self.verbose:
# #             print('softmax', out.shape)
        
#         return out    
    
    
# class RespiratoryRegression(nn.Module):
#     def __init__(self, num_classes=10, input_dim=50):
#         super(RespiratoryRegression, self).__init__()
# #         self.bn = nn.BatchNorm1d(channel_n)
#         self.relu = nn.ReLU()
# #         self.fc1 = nn.Linear(input_dim, num_classes)
#         self.fc1 = nn.Linear(input_dim, 20)
#         self.fc2 = nn.Linear(20, num_classes)

#     def forward(self, x):  
# #         print(x.size())
#         out = self.fc1(x)
#         out = F.relu(out)
#         out = self.fc2(out)
        
#         return out

# class resnet_multiverse(nn.Module):
# #     def __init__(self, inputDim=1000, input_channel=1, class_N=2, training_params=None):
#     def __init__(self, training_params=None):
#         super(resnet_multiverse, self).__init__()
        
#         input_dim = training_params['data_dimensions'][1]
#         input_channel = training_params['data_dimensions'][0]
#         channel_n = training_params['channel_n']
#         kernel_size = training_params['kernel_size']
#         self.tasks = training_params['tasks']
#         self.class_N = training_params['n_classes'][0]
# #         self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n)

#         self.feature_extractor = ResNet1D(training_params=training_params)
# #         self.feature_extractor = FeatureExtractor_CNN(input_dim=inputDim, input_channel=input_channel, training_params=training_params)

#         feature_out_dim = self.feature_extractor.feature_out_dim
# #         print('feature_out_dim', feature_out_dim)
# #         channel_n = self.feature_extractor.channel_n

# #         self.gender_classfier = GenderClassifier(num_classes=class_N, input_dim=feature_out_dim)
# #         self.respiratory_regressor = RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)
#         self.regressors = {}
    
#         self.regressors = nn.ModuleDict(
# #             [[task, RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)] for task in self.tasks]
#             [[task, RespiratoryRegression(num_classes=self.class_N, input_dim=feature_out_dim+2)] for task in self.tasks]
#         )
            
            
            
            
            
# #             [
            
# #                 ['lrelu', RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)],
# #                 ['prelu', nn.PReLU()]
# #         ])
    
# #         regresso = [task: RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim) for task in self.tasks]


        
# #         for task in training_params['tasks']:
# #             self.regressors[task] = RespiratoryRegression(num_classes=class_N, input_dim=feature_out_dim)
# #         self.EE_regressor = EERegression(num_classes=class_N, input_dim=feature_out_dim)
# #         self.RR_regressor = RRRegression(num_classes=class_N, input_dim=feature_out_dim)

#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
# #     print('DannModel_total_params:', pytorch_total_params)

#     def forward(self, x, feature):
#         feature_out = self.feature_extractor(x)
        
#         feature = feature.squeeze()
# #         print('feature', feature.size())
# #         feature_out, feature
# #         print(feature_out.size(), feature.size())
#         feature_out = torch.cat((feature_out, feature), 1)
# #         print(feature_out.size())
# #         sys.exit()
# #         print('feature_out size:', feature_out.size())
#         output = {}
#         for task in self.tasks:
# #             next(self.regressors[task].parameters()).is_cuda 
# #             print(self.regressors[task])
#             output[task] = self.regressors[task](feature_out)
            
            
            
# #         EE_output = self.EE_regressor(feature_out)
# #         RR_output = self.RR_regressor(feature_out)
# # #         print('regression_output:', regression_output.size())
# # #         return regression_output
# #         output = {
# #             'RR_cosmed': RR_output,
# #             'EE_cosmed': EE_output
# #         }
# #         output = [RR_output, EE_output]
#         return output
