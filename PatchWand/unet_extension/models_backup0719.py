import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import sys

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch, training_params=None, groups=1, hidden_ch=None):
        super(conv_block, self).__init__()
        self.inception = False
#         self.SE_block = False
        
        if hidden_ch is None:
            hidden_ch = out_ch
        
        if training_params is not None:
            if training_params['variant']=='inception':
                self.inception = True

#             if  training_params['variant']=='SE_block':
#                 self.SE_block = True
        
        if not self.inception:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=groups),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True, groups=groups),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
        else:
            k1 = training_params['kernels']['k1']
            k2 = training_params['kernels']['k2']
            k3 = training_params['kernels']['k3']

            # k1xk1 convolution branch
            self.conv_k1 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=k1, stride=1, padding='same', bias=True),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=k1, stride=1, padding='same', bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

            # k2xk2 convolution branch
            self.conv_k2 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=k2, stride=1, padding='same', bias=True),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=k2, stride=1, padding='same', bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

            # k3xk3 convolution branch
            self.conv_k3 = nn.Sequential(
                nn.Conv2d(in_ch, hidden_ch, kernel_size=k3, stride=1, padding='same', bias=True),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=k3, stride=1, padding='same', bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        
            self.ch_pooling = torch.nn.Conv2d(out_ch*3, out_ch, 1) # out_channels->1 channel, kernel size = 1


    def forward(self, x):
        if not self.inception:
            x = self.conv(x)
        else:
            x_k1 = self.conv_k1(x)
            x_k2 = self.conv_k2(x)
            x_k3 = self.conv_k3(x)
            
#             print(x_k1.size(), x_k2.size(), x_k3.size())

            x_out = torch.cat([x_k1, x_k2, x_k3], dim=1)
            x = self.ch_pooling(x_out)

        return x


    
# prior to 5/7
class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
def cat_crop(x1, x2):
#     print(x1.size(), x2.size())
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[2] - x1.size()[2]
    
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2))
    x = torch.cat([x2, x1], dim=1)
    return x    


# added on 7/5/2022 
class Late_UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, training_params=None):
        super(Late_UNet, self).__init__()

        self.input_names = training_params['input_names']
        self.unets = nn.ModuleDict()

        for input_name in self.input_names:
            self.unets[input_name] = UNet(in_ch=1, out_ch=2, n1=training_params['N_channels'], training_params=training_params)
            
        if training_params['variant']=='SE_block':
#             self.SE_block = nn.ModuleDict()
# #             for input_name in self.input_names:
#             self.SE_block['0']= SELayer(channel=len( self.input_names )) # for label=0
#             self.SE_block['1']= SELayer(channel=len( self.input_names )) # for label=1
            self.SE_block= SELayer(channel=len( self.input_names )) # for label=1
        else:
            self.SE_block =  nn.Identity()
#         self.SE_block0 = SELayer(channel=len( self.input_names ))
#         self.SE_block1 = SELayer(channel=len( self.input_names ))
        
#         self.ch_pooling = 
#         self.ch_pooling = nn.ModuleDict()
#         for input_name in self.input_names:
#             self.ch_pooling[input_name] = torch.nn.Conv2d(2, 1, 1) # 2*N_input->2 channel, kernel size = 1

    
    def forward(self, x):
        # x dim: torch.rand(32,4,20,58)

#         output = []
        unet_output = {}
        for i_input, input_name in enumerate(self.unets.keys()):
            x_input = x[:,[i_input], :, :]
            # out dim: (N_batch, 2, 20, 58)
            out = self.unets[input_name](x_input)
#             out = nn.functional.softmax(out, dim=1)
#             output.append(out)
            unet_output[input_name] = out
#             output[input_name] = nn.functional.sigmoid(out)

#         output = sum(output) # (N_batch, 2, 20, 58)
    
        output0 = []
        output1 = []
        for i_input, input_name in enumerate(self.input_names):
            output0.append( unet_output[input_name][:,[0],:,:] )
            output1.append( unet_output[input_name][:,[1],:,:] )
        
#         output0 = self.SE_block['0'](torch.concat(output0, axis=1))
        output0 = self.SE_block(torch.concat(output0, axis=1))
        output0 = torch.sum(output0, axis=1, keepdim=True)

#         output1 = self.SE_block['1'](torch.concat(output1, axis=1))
        output1 = self.SE_block(torch.concat(output1, axis=1))
        output1 = torch.sum(output1, axis=1, keepdim=True)

        # out dim: (N_batch, 2, 20, 58)
        output = torch.concat([output0, output1], axis=1)
#         output = self.SE_block(output)

    

        # out dim: (N_batch, 2 x N_input, 20, 58)
#         output = torch.concat(output, axis=1)
#         print(out.size())
#         out_concat = []
#         for i_ch in range(out.size()[1]):
#             out_concat.append( [] )
#             for i_input, input_name in enumerate(self.unets.keys()):
#                 out_concat[i_ch].append(output[input_name][:,[i_ch],:,:])
            
#             out_concat[i_ch] = torch.concat(out_concat[i_ch], axis=1)
#             out_concat[i_ch] = self.ch_pooling[input_name](out_concat[i_ch])
#             # out_concat[i_ch] dim will be (N_batch, 1, 20, 58)

#         out_concat = torch.concat(out_concat, axis=1)
        # out_concat dim will be (N_batch, 2, 20, 58)

#         output = sum(output)
#         output = self.SE_block(output)
        
#         output = self.ch_pooling(output)
        
        return output
#         return out_concat
#         output1 = torch.concat(output1, axis=1)

#         output0 = torch.sum(self.SE_block0(output0), axis=1, keepdim=True)
#         output1 = torch.sum(self.SE_block1(output1), axis=1, keepdim=True)

#         output = torch.concat([output0, output1], axis=1)
        
#         return output
        
        
#         output0 = []
#         output1 = []

#         for i_input, input_name in enumerate(self.unets.keys()):
#             x_input = x[:,[i_input], :, :]
#             out = self.unets[input_name](x_input)
#             output0.append(out[:,[0], :, :])
#             output1.append(out[:,[1], :, :])

#         output0 = torch.concat(output0, axis=1)
#         output1 = torch.concat(output1, axis=1)

#         output0 = torch.sum(self.SE_block0(output0), axis=1, keepdim=True)
#         output1 = torch.sum(self.SE_block1(output1), axis=1, keepdim=True)

#         output = torch.concat([output0, output1], axis=1)
#         return output

# prior to 5/7
class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=None, out_ch=None, n1=None, training_params=None):
        super(UNet, self).__init__()
        self.SE_block = False
        
        if in_ch is None:
            in_ch = training_params['data_dimensions'][0]
        if out_ch is None:
            out_ch = 2
        if n1 is None:
            n1=training_params['N_channels']
#             model_template = U_Net(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)


        self.N_classes = out_ch
#         print(in_ch, out_ch)
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.Conv0 = conv_block(in_ch, in_ch, training_params=training_params, groups=in_ch)
#         self.Conv1 = conv_block(in_ch, filters[0], training_params)
#         self.Conv1 = conv_block(1, filters[0], training_params=training_params)

        self.Conv1 = conv_block(in_ch, filters[0], training_params)
        self.Conv2 = conv_block(filters[0], filters[1], training_params)
        self.Conv3 = conv_block(filters[1], filters[2], training_params)
#         self.Conv4 = conv_block(filters[2], filters[3], training_params)
#         self.Conv5 = conv_block(filters[3], filters[4])

#         self.Up5 = up_conv(filters[4], filters[3])
#         self.Up_conv5 = conv_block(filters[4], filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2], training_params)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1], training_params)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0], training_params)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        
        if training_params['regressor'] is not None:
            self.regressor = regressor_dict[training_params['regressor']](training_params)
        else:
            self.regressor = None
       # self.active = torch.nn.Sigmoid()
    

        if training_params['variant']=='SE_block':
            self.SE_block = True
#             print('filters', filters)

            self.SE_e3 = SELayer(filters[2])
            self.SE_d3 = SELayer(filters[1])
            self.SE_e1 = SELayer(filters[0])
            self.SE_d2 = SELayer(filters[0])
            self.SE_final = SELayer(filters[0])
#             self.SE = SELayer(in_ch)

#             self.SE1 = SELayer(in_ch)
#             self.SE2 = SELayer(filters[0])

    def forward(self, x):

#         x_concat = []
#         for i_ch in range(x.size()[1]):
#             x_concat.append(self.Conv0(x[:, [i_ch], :, :]))
            
#         x_concat = torch.cat(x_concat, dim=1)
        
# #         print(x_concat.size())
# #         x = self.Conv0(x_concat)
# #         x = self.Conv0(x)
#         x = self.SE1(x_concat)
        
        
#         x = self.Conv0(x)
#         if self.SE_block:
#             x = self.SE(x)
            
#         x = torch.mean(x, dim=1, keepdim=True)


        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
    

#         if self.SE_block:
#             e2 = self.SE2(e2)
            
        e2 = self.Conv2(e2)
#         ic(e2.size())

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
#         ic(e3.size())

#         e4 = self.Maxpool3(e3)
#         e4 = self.Conv4(e4) 

#         d4 = self.Up4(e4)
#         d4 = cat_crop(d4, e3)
#         d4 = self.Up_conv4(d4)

        d3 = self.Up3(e3)
        if self.SE_block:
#             print('e3, d3', e3.size(), d3.size())
            e3 = self.SE_e3(e3)
            d3 = self.SE_d3(d3)
    
        d3 = cat_crop(d3, e2)
        d3 = self.Up_conv3(d3)
        
        d2 = self.Up2(d3)
        
        if self.SE_block:
#             print('e1, d2', e1.size(), d2.size())
            e1 = self.SE_e1(e1)
            d2 = self.SE_d2(d2)
        d2 = cat_crop(d2, e1)
        d2 = self.Up_conv2(d2)

        if self.SE_block:
            d2 = self.SE_final(d2)

        out = self.Conv(d2)
        
#         print('out', out.size())

        if self.regressor is not None:
            RR_est = self.regressor(out)
            return out, RR_est
        else:
#             return out, torch.rand(out.size()[0])
            return out


        #d1 = self.active(out)

        
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=1):
#         super(SELayer, self).__init__()        
#         self.conv_layers = nn.Sequential(
#             # CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout
#             nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True),
# #             nn.Sigmoid()
# #             nn.Dropout(p=0.5),
#         )
        
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
        
#         channel_middle = channel // reduction
#         self.fc = nn.Sequential(
#             nn.Linear(channel,channel_middle, bias=True),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm1d( channel_middle ),
#             nn.Linear(channel_middle, channel, bias=True),
# #             nn.Sigmoid()
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, x):
#         b, c, h, w = x.size()
#         residual = x

#         # combine batch size and channel size, apply the filters
#         x = self.conv_layers(x.view(-1, 1, h, w))
# #         x = self.conv_layers(x)

#         # combine batch size and channel size, apply the same filters on all channels
#         x = x.view(b, c, x.size()[-2], x.size()[-1])
        
#         # average pooling for each instance, each channel (y dim: (batch_size, N_channel))
#         y = self.avg_pool(x).view(b, c)

#         # squeeze and excite (reduce N_channel to N_channel//reduction then back to N_channel)
#         # y is normalized to 1 (each channel has its own weight)
#         y = self.fc(y).view(b, c, 1, 1)
        
# #         print(y)

#         return residual * y.expand_as(residual)

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=1):
#         super(SELayer, self).__init__()        
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(4),
#             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(1),
#             nn.ReLU(inplace=True),
# #             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#         )
        
# #         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.avg_pool = nn.AdaptiveMaxPool2d(1)
        
        
#         channel_middle = 1
#         self.fc = nn.Sequential(
# #             nn.Linear(channel, channel // reduction, bias=True),
#             nn.Linear(channel,channel_middle, bias=True),
#             nn.ReLU(inplace=True),
# #             nn.BatchNorm1d( channel // reduction ),
#             nn.BatchNorm1d( channel_middle ),
# #             nn.Linear(channel // reduction, channel, bias=True),
#             nn.Linear(channel_middle, channel, bias=True),
# #             nn.Sigmoid()
#             nn.Softmax(dim=-1)
#         )

#     def forward(self, x):
#         b, c, h, w = x.size()
#         residual = x

#         # combine batch size and channel size, apply the same filters on all channels
#         x = self.conv_layers(x.view(-1, 1, h, w))

#         # combine batch size and channel size, apply the same filters on all channels
#         x = x.view(b, c, x.size()[-2], x.size()[-1])
        
#         # average pooling for each instance, each channel (y dim: (batch_size, N_channel))
#         y = self.avg_pool(x).view(b, c)

#         # squeeze and excite (reduce N_channel to N_channel//reduction then back to N_channel)
#         # y is normalized to 1 (each channel has its own weight)
#         y = self.fc(y).view(b, c, 1, 1)
        
# #         print(y)

#         return residual * y.expand_as(residual)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
#             nn.Softmax(dim=-1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


    
    
class DominantFreq_regressor(nn.Module):
    """
    DominantFreq_regressor - Basic Implementation
    Paper : TBD (may be published in BioCAS 2022)
    """
    def __init__(self, training_params):
        super(DominantFreq_regressor, self).__init__()
        self.xf_masked = training_params['xf_masked'] # has a dim of 58 (# of spectral bins)
        self.flipper = training_params['flipper']

    def forward(self, x):
        # x dim: (N_batch, N_windows/N_channel, N_spectralbins)
        
        # out dim: (N_batch, N_windows, N_spectralbins)
        # normalize so each spectral feature for each instance and each window sums to 1
#         print(x.size())

#         fig, ax = plt.subplots(1,1, figsize=(5, 5), dpi=80, facecolor='white')
#         ax.imshow(x[0,0,:,:].detach().cpu().numpy())
#         fig, ax = plt.subplots(1,1, figsize=(5, 5), dpi=80, facecolor='white')
#         ax.imshow(x[0,1,:,:].detach().cpu().numpy())
        
#         plt.show()
#         sys.exit()
        i_shift = x.size()[2]//2

        x = x[:,0,i_shift,:] # output dim has 2 channel (select the second one because higher values correspond to higher frequency response in this dimension)
            
        if self.flipper:
            N_freq = x.size()[-1]
            x = x[:, :N_freq//2]

        out = x / torch.sum(x, axis=-1, keepdim=True)
#         print(out.size(), self.xf_masked.size())

        # out dim: (N_batch, N_windows * N_spectralbins)
        out = torch.sum(out * self.xf_masked.to(out.device), axis=-1)
#         print(out.size())
#         sys.exit()

        return out


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.loss_weight = 3

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        
#         print(inputs.size(), targets.size())
        
        inputs = inputs.reshape(-1).float()
        targets = targets.reshape(-1).float()
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets)
        Dice_BCE = BCE + dice_loss * self.loss_weight
    
        losses = {
            'BCE': BCE,
            'dice': dice_loss,
            'total': Dice_BCE,
        }
#         return Dice_BCE
        return losses

class DiceBCERRLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCERRLoss, self).__init__()
        self.dice_weight = 3
        self.RR_weight = 0

    def forward(self, inputs, targets, RR_est, RR_label, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        
#         print(inputs.size(), targets.size())
        
        inputs = inputs.reshape(-1).float()
        targets = targets.reshape(-1).float()
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        dice_loss = dice_loss * self.dice_weight
        
        BCE = F.binary_cross_entropy(inputs, targets)

#         RR_label = RR_label[:,None].repeat(1, RR_est.size()[1])
#         print(RR_est.size(), RR_label.size())
#         sys.exit()
        mse_RR = F.mse_loss(RR_est.float(), RR_label.float()) * self.RR_weight
        
        total_loss = BCE + dice_loss + mse_RR

#         print(BCE, dice_loss, mse_RR, total_loss)
#         print(BCE.dtype, dice_loss.dtype, mse_RR.dtype, total_loss.dtype)
        
#         sys.exit()
        losses = {
            'BCE': BCE,
            'dice': dice_loss,
            'mse_RR': mse_RR,
            'total': total_loss,
        }
#         return Dice_BCE
        return losses


regressor_dict = {
    'DominantFreq_regressor': DominantFreq_regressor
}
model_dict = {
    'Late_UNet': Late_UNet,
    'UNet': UNet,
}





# KERNEL_SIZE = 3
# CHANNEL_N_1 = 4
# CHANNEL_N_2 = 8
# CHANNEL_N_3 = 8
# CHANNEL_N_4 = 8

# # Convolutional neural network (two convolutional layers)
# class CNN_sig(nn.Module):
#     def __init__(self, input_dim=50, input_channel=3):
#         super(CNN_sig, self).__init__()
#         self.layer1 = nn.Sequential(
#           nn.Conv1d(input_channel, CHANNEL_N_1, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#           nn.BatchNorm1d(CHANNEL_N_1),
#           nn.ReLU(),
#           nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#           nn.Conv1d(CHANNEL_N_1, CHANNEL_N_2, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#           nn.BatchNorm1d(CHANNEL_N_2),
#           nn.ReLU(),
#           nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer3 = nn.Sequential(
#           nn.Conv1d(CHANNEL_N_2, CHANNEL_N_3, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#           nn.BatchNorm1d(CHANNEL_N_3),
#           nn.ReLU(),
#           nn.MaxPool1d(kernel_size=2, stride=2))
#         self.layer4 = nn.Sequential(
#           nn.Conv1d(CHANNEL_N_3, CHANNEL_N_4, kernel_size=KERNEL_SIZE, stride=1, padding=2),
#           nn.BatchNorm1d(CHANNEL_N_4),
#           nn.ReLU(),
#           nn.MaxPool1d(kernel_size=2, stride=2))
        
#         cnn_layer1_dim = (input_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

#         cnn_layer2_dim = (pool_layer1_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

#         cnn_layer3_dim = (pool_layer2_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)
		
#         cnn_layer4_dim = (pool_layer3_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#         pool_layer4_dim = math.floor((cnn_layer4_dim-1*(2-1)-1)/2+1)


# #         self.feature_out_dim = int(pool_layer3_dim*CHANNEL_N_3)
#         self.feature_out_dim = int(pool_layer4_dim*CHANNEL_N_4)
#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
# #         print('CNN1D feature extractor total_params:', pytorch_total_params)
# #         print('feature_out_dim dim:', self.feature_out_dim)

#     def forward(self, x):
#         out1 = self.layer1(x.float())
#         out2 = self.layer2(out1)
#         out3 = self.layer3(out2)
# #         out = out3.reshape(out3.size(0), -1)
#         out4 = self.layer4(out3)
#         out = out4.reshape(out4.size(0), -1)
# #         print(out1.size(), out2.size(), out3.size(), out.size())
#         return out
    
    
# HIDDEN_DIM_1 = 128
# HIDDEN_DIM_2 = 32
# OUT_DIM_MODEL = 2



# class regression_net(nn.Module):
#     def __init__(self, input_dim=50, output_dim=2):
#         super(regression_net, self).__init__()
#         self.fc1 = nn.Linear(input_dim, HIDDEN_DIM_1)
#         self.bn1 = nn.BatchNorm1d(HIDDEN_DIM_1)
#         self.dp1 = nn.Dropout(p=0.2)
        
#         self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
#         self.bn2 = nn.BatchNorm1d(HIDDEN_DIM_2)
#         self.dp2 = nn.Dropout(p=0.2)
        
#         self.fc3 = nn.Linear(HIDDEN_DIM_2, output_dim)
# #         self.drop = nn.Dropout(p=P_DROPOUT)
#         self.relu = nn.ReLU()
#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
# #         print('ClassClassifier_total_params:', pytorch_total_params)

#     def forward(self, x):
#         self.bn1(self.fc1(x.float()))

#         out1 = self.dp1(self.relu(self.bn1(self.fc1(x.float()))))
#         out2 = self.dp2(self.relu(self.bn2(self.fc2(out1))))
#         out3 = self.fc3(out2)
#         return out3
    
    

    
# class SeismoNet(nn.Module):
#     def __init__(self, device, cnn_N=9):
#         super(SeismoNet, self).__init__()
        
#         feature_dim = 500
        
#         model_sigs = []
#         for i in range(cnn_N):
#             if i%3 == 0:
# #                 model_sigs.append(ResNet(BasicBlock, [2, 2, 2, 2], feature_dim=500, input_channel=3).float())
#                 model_sigs.append(CNN_sig(input_dim=1000, input_channel=3).to(device).float())
#             else:
# #                 model_sigs.append(ResNet(BasicBlock, [2, 2, 2, 2], feature_dim=500, input_channel=2).float())
#                 model_sigs.append(CNN_sig(input_dim=1000, input_channel=2).to(device).float())

#         self.model_sigs = model_sigs
#         cnn_out_dim = self.model_sigs[0].feature_out_dim
# # 		print(cnn_out_dim)
        
# #         cnn_out_dim = self.model_iw_dd.feature_out_dim

# #         self.model_reg = regression_net(input_dim=252*3, output_dim=OUT_DIM_MODEL)
# #         self.model_reg = regression_net(input_dim=cnn_out_dim*9, output_dim=OUT_DIM_MODEL)

#         self.model_reg = regression_net(input_dim=cnn_out_dim*cnn_N, output_dim=OUT_DIM_MODEL)
# #         self.model_reg = regression_net(input_dim=feature_dim*cnn_N, output_dim=OUT_DIM_MODEL)


#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         #     print('DannModel_total_params:', pytorch_total_params)

# #     def forward(self, data_scg, data_is, data_iw):
#     def forward(self, data):
# #         data = data.to(device)
        
# #         feature_sigs = []
# #         for i_sig in range(data.size()[1]):
# #             print(data[:,i_sig,:].size())
# #             feature_sigs.append(self.model_sigs[i_sig](data[:,i_sig,:]))
        
# #         feature_fused = torch.cat(feature_sigs, 1)
#         data_scg = data[:,0:3,:]
#         data_ppg1 = data[:,3:5,:]
#         data_ppg2 = data[:,5:7,:]
        
#         data_scg_d = data[:,7:10,:]
#         data_ppg1_d = data[:,10:12,:]
#         data_ppg2_d = data[:,12:14,:]
        
#         data_scg_dd = data[:,14:17,:]
#         data_ppg1_dd = data[:,17:19,:]
#         data_ppg2_dd = data[:,19:21,:]

#         # raw
#         feature_scg = self.model_sigs[0](data_scg)
#         feature_is = self.model_sigs[1](data_ppg1)
#         feature_iw = self.model_sigs[2](data_ppg2)
        
#         # derivative 
#         feature_scg_d = self.model_sigs[3](data_scg_d)
#         feature_is_d = self.model_sigs[4](data_ppg1_d)
#         feature_iw_d = self.model_sigs[5](data_ppg2_d)
        
#         # 2nd deriv
#         feature_scg_dd = self.model_sigs[6](data_scg_dd)
#         feature_is_dd = self.model_sigs[7](data_ppg1_dd)
#         feature_iw_dd = self.model_sigs[8](data_ppg2_dd)
        
# #         print(feature_iw_dd.size())
# #         # raw
# #         feature_scg = self.model_scg(data_scg)
# #         feature_is = self.model_is(data_is)
# #         feature_iw = self.model_iw(data_iw)
        
# #         # derivative 
# #         feature_scg_d = self.model_scg_d(data_scg_d)
# #         feature_is_d = self.model_is_d(data_is_d)
# #         feature_iw_d = self.model_iw_d(data_iw_d)
        
# #         # 2nd deriv
# #         feature_scg_dd = self.model_scg_dd(data_scg_dd)
# #         feature_is_dd = self.model_is_dd(data_is_dd)
# #         feature_iw_dd = self.model_iw_dd(data_iw_dd)
        
#         feature_fused = torch.cat((feature_scg, feature_is, feature_iw, 
#                                    feature_scg_d, feature_is_d, feature_iw_d, 
#                                    feature_scg_dd, feature_is_dd, feature_iw_dd), 1)
        
#         prediction = self.model_reg(feature_fused)
        
#         return feature_fused, prediction

	
# # model_scg = CNN_sig(input_dim=1000).to(device).float()
# # model_is = CNN_sig(input_dim=1000).to(device).float()
# # model_iw = CNN_sig(input_dim=1000).to(device).float()
# # model_reg = regression_net(input_dim=252*3, output_dim=2)


# # model = SeismoNet(device, cnn_N=9)
# # print(model)


# '''ResNet in PyTorch.
# For Pre-activation ResNet, see 'preact_resnet.py'.
# Reference:
# [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
#     Deep Residual Learning for Image Recognition. arXiv:1512.03385
# '''


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv1d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(planes)
#         self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
#                                stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(planes)
#         self.conv3 = nn.Conv1d(planes, self.expansion *
#                                planes, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm1d(self.expansion*planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv1d(in_planes, self.expansion*planes,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm1d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, feature_dim=10, input_channel=3):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=3,
#                                stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
# #         self.linear = nn.Linear(512*block.expansion, num_classes)
#         self.linear = nn.Linear(3968, feature_dim)
        
        
#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
# #         print(out.size())
#         out = out.view(out.size(0), -1)
# #         print(out.size())

#         out = self.linear(out)
# #         print(out.size())

#         return out


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2], input_channel=2)

# # aaa = ResNet18()
# # aaa



# GRAVE
# class up_conv(nn.Module):
#     """
#     Up Convolution Block
#     """
#     def __init__(self, in_ch, out_ch):
#         super(up_conv, self).__init__()
#         self.up = nn.Sequential(
#             nn.Upsample(scale_factor=(2,1)),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         x = self.up(x)
#         return x
# class U_Net(nn.Module):
#     """
#     UNet - Basic Implementation
#     Paper : https://arxiv.org/abs/1505.04597
#     """
#     def __init__(self, in_ch=3, out_ch=2, n1=8):
#         super(U_Net, self).__init__()
#         self.N_classes = out_ch
# #         print(in_ch, out_ch)
#         filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
#         self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))
#         self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))
#         self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))
#         self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=(2, 1))

#         self.Conv1 = conv_block(in_ch, filters[0])
#         self.Conv2 = conv_block(filters[0], filters[1])
#         self.Conv3 = conv_block(filters[1], filters[2])
#         self.Conv4 = conv_block(filters[2], filters[3])
# #         self.Conv5 = conv_block(filters[3], filters[4])

# #         self.Up5 = up_conv(filters[4], filters[3])
# #         self.Up_conv5 = conv_block(filters[4], filters[3])

#         self.Up4 = up_conv(filters[3], filters[2])
#         self.Up_conv4 = conv_block(filters[3], filters[2])

#         self.Up3 = up_conv(filters[2], filters[1])
#         self.Up_conv3 = conv_block(filters[2], filters[1])

#         self.Up2 = up_conv(filters[1], filters[0])
#         self.Up_conv2 = conv_block(filters[1], filters[0])

#         self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

#        # self.active = torch.nn.Sigmoid()

#     def forward(self, x):

#         e1 = self.Conv1(x)
# #         ic(e1.size())

#         e2 = self.Maxpool1(e1)
#         e2 = self.Conv2(e2)
# #         ic(e2.size())

#         e3 = self.Maxpool2(e2)
#         e3 = self.Conv3(e3)
# #         ic(e3.size())

# #         e4 = self.Maxpool3(e3)
# #         e4 = self.Conv4(e4) 

# #         d4 = self.Up4(e4)
# #         d4 = cat_crop(d4, e3)
# #         d4 = self.Up_conv4(d4)
        


#         d3 = self.Up3(e3)
# #         d3 = self.Up3(d4)
#         d3 = cat_crop(d3, e2)
#         d3 = self.Up_conv3(d3)
        

#         d2 = self.Up2(d3)
#         d2 = cat_crop(d2, e1)
#         d2 = self.Up_conv2(d2)

#         out = self.Conv(d2)

#         #d1 = self.active(out)

#         return out