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

KERNEL_SIZE = 3

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
                nn.Conv2d(in_ch, hidden_ch, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=True, groups=groups),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, out_ch, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=True, groups=groups),
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
            nn.Conv2d(in_ch, out_ch, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=True),
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
#     print(x1.size(), x2.size())

    x = torch.cat([x2, x1], dim=1)
    return x    


# updated 7/22
class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=None, out_ch=None, n1=None, training_params=None):
        super(UNet, self).__init__()
        self.SE_block = False
        self.AT_block = False
        self.Attention_UNet = False
        
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

        if training_params['variant']=='AT_block':
            self.AT_block = True
#             self.Conv0 = conv_block(in_ch, in_ch, training_params=training_params, groups=in_ch)
            self.Conv0 = nn.Sequential(
                            nn.Conv2d(in_ch, in_ch*2, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=True, groups=in_ch),
                            nn.BatchNorm2d(in_ch*2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_ch*2, in_ch, kernel_size=KERNEL_SIZE, stride=1, padding=1, bias=True, groups=in_ch),
                            nn.BatchNorm2d(in_ch),
                            nn.ReLU(inplace=True))
            self.atten_block = sAttentLayer2(training_params, N_freq=training_params['data_dimensions'][-1])
            self.Conv1 = conv_block(1, filters[0], training_params)
        else:
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
            
            dim_freq_depth1 = training_params['data_dimensions'][-1]
            dim_freq_depth2 = dim_freq_depth1//2
            dim_freq_depth3 = dim_freq_depth2//2
            
            self.SE_x = sSELayer(N_freq=dim_freq_depth1 , channel=in_ch)
            self.SE_e2 = sSELayer(N_freq=dim_freq_depth2 , channel=filters[0])
            self.SE_e3 = sSELayer(N_freq=dim_freq_depth3 , channel=filters[1])
            self.SE_d3 = sSELayer(N_freq=dim_freq_depth2 , channel=filters[1])
            self.SE_d2 = sSELayer(N_freq=dim_freq_depth1 , channel=filters[0])       
            
        if training_params['variant']=='Attention_UNet':
            self.Attention_UNet = True
            self.Att2 = AttentionBlock(F_g=filters[1], F_l=filters[1], n_coefficients=4)# bottom-most Attention Gate
            self.Att1 = AttentionBlock(F_g=filters[0], F_l=filters[0], n_coefficients=4) # top-most Attention Gate


    def forward(self, x):

        # AT_block is applied at the begining only
#         if self.AT_block:
# #             out, weights = self.atten_block(self.Conv0(x)) # x will become (N, 1, 20, 58)
#             out, weights = self.atten_block(x) # x will become (N, 1, 20, 58)
#             e1 = self.Conv1(out) # first conv
        
        if self.AT_block:
            x, weights = self.atten_block(x) # x will become (N, 1, 20, 58)
#             e1 = self.Conv1(x) # first conv
#         else:
        # SE_block is applied at the begining
        if self.SE_block:
            x = self.SE_x(x)
        
        e1 = self.Conv1(x) # first conv
        
        e2 = self.Maxpool1(e1)
    
        # SE_block is applied after maxpooling
        if self.SE_block:
            e2 = self.SE_e2(e2)
            
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
    
        # SE_block is applied after maxpooling
        if self.SE_block:
            e3 = self.SE_e3(e3)

        e3 = self.Conv3(e3) # smallest size

        d3 = self.Up3(e3)
    
    
        if self.Attention_UNet:
#             print('')
#             print(d3.size(), e2.size()) 
            # d3: torch.Size([2, 8, 10, 28]) e2: torch.Size([2, 8, 10, 29])
            e2 = self.Att2(gate=d3, skip_connection=e2)
#             print(d3.size(), e2.size()) 

        # d3: torch.Size([2, 8, 10, 29]) | e2: torch.Size([2, 8, 10, 29])
        d3 = cat_crop(d3, e2)
        d3 = self.Up_conv3(d3)

        if self.SE_block:
            d3 = self.SE_d3(d3)
        
        d2 = self.Up2(d3)
        
        if self.Attention_UNet:
#             print('')
#             print('e1, d2', e1.size(), d2.size())
#             print(d2.size(), e1.size())
            e1 = self.Att1(gate=d2, skip_connection=e1)
#             print('e1, d2', e1.size(), d2.size())

#         sys.exit()

        d2 = cat_crop(d2, e1)
        d2 = self.Up_conv2(d2)
        
        if self.SE_block:
            d2 = self.SE_d2(d2)

        out = self.Conv(d2)

        if self.regressor is not None:
            RR_est = self.regressor(out)
            return out, RR_est
        else:
#             return out, torch.rand(out.size()[0])
            return out



class sSELayer(nn.Module):
    """spectral SE block with learnable parameters"""
    def __init__(self, N_freq, channel, reduction=2):
        super(sSELayer, self).__init__()
        
        self.avg_pool = nn.AdaptiveMaxPool2d((1, N_freq))
        # option 1: combine the vectors channel-wise
        self.ch_pooling = nn.Conv1d(channel, 1, kernel_size=1, stride=1, padding=0, bias=True)

        self.fc = nn.Sequential(
            nn.Linear(N_freq, N_freq // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(N_freq // reduction, N_freq, bias=True),
            nn.Sigmoid()
        #     nn.Tanh()
        )
        # option 2: combine the vectors feature-wise (last dimension, WIP) -> channel dimension becomes the feature dimension


    def forward(self, x):
        x_avg = torch.squeeze(self.avg_pool(x), axis=-2) # reduce the temporal dimension to 1        
        x_linear = torch.squeeze(self.ch_pooling(x_avg), axis=-2) # reduce the channel dimension to 1
        weight_spectral = self.fc(x_linear) # squeeze and excite (attention weight)

        return x * weight_spectral[:,None,None,:].expand_as(x) # multiplying input by the weights

    
class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def padder(self, x1, x2):
#         print(x1.size(), x2.size())
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        return x1
        
    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
#         print(gate.size(), skip_connection.size())
#         print(gate, skip_connection)
        gate = self.padder(gate, skip_connection)
#         print(gate.size(), skip_connection.size())
        
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


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

#         print(inputs.min(), targets.min())
#         print(inputs.max(), targets.max())

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


class sAttentLayer(nn.Module):
    """spectral Attention block with learnable parameters"""
    def __init__(self, N_freq=58, channel=1, reduction=2):
        super(sAttentLayer, self).__init__()
        
        # reduce the temporal dimension
        groups = 1
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=KERNEL_SIZE, stride=1, padding='same', bias=True, groups=groups),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=(KERNEL_SIZE,1), padding=(0,1)),
            nn.Conv2d(channel, channel, kernel_size=KERNEL_SIZE, stride=1, padding='same', bias=True, groups=groups),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=KERNEL_SIZE, stride=(KERNEL_SIZE,1), padding=(0,1)),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, N_freq))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.activation = nn.Tanh()
        
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        
        # plan1: try conv + avgpool + maxpool + softmax (sub 4 fail)
        # plan2: try avgpool + maxpool + softmax
        # plan3: try conv + avgpool + normalized maxpool + softmax
        # plan4: try avgpool + normalized maxpool + softmax

        # plan5: try conv + avgpool + maxpool + activation + softmax
        # plan6: try avgpool + maxpool + softmax
        # plan7: try conv + avgpool + normalized maxpool + activation + softmax
        # plan8: try avgpool + normalized maxpool + activation + softmax
        
        outputs = {}
        for input_name in x.keys():
            # (N, 2, 20, 58) -> (N, 1, 20, 58) -> (N, 1, 2, 58) -> (N, 1, 1, 58)
            interm = self.avg_pool( self.conv(x[input_name][:,[1],:,:]) )
#             interm = self.avg_pool( x[input_name][:,[1],:,:] )
            
            # normalization: (N, 1, 1, 58) -> (N, 1, 1, 58) -> (N, 1, 1, 1)/(N, 1, 1, 1) -> (N, 1, 1, 1) 
#             outputs[input_name] = self.max_pool(interm) / interm.sum(dim=-1,keepdim=True)
            outputs[input_name] = self.max_pool(interm)
#             outputs[input_name] = self.activation(outputs[input_name]) # (N, 1, 1, 1)  -> (N,) 
            outputs[input_name] = outputs[input_name].squeeze() # (N, 1, 1, 1)  -> (N,) 
            
        weights = []
        for input_name in outputs.keys():
            weights.append(outputs[input_name])
        
        weights = torch.stack(weights).T
        weights = self.softmax(weights)
        
        out = x
        
        for i_input, input_name in enumerate(out.keys()):
            out[input_name] = out[input_name] * weights[:,i_input][:,None,None,None].expand_as(out[input_name])

        return out, weights



class sAttentLayer2(nn.Module):
    """spectral Attention block with learnable parameters"""
    def __init__(self, training_params, N_freq=58, channel=1, reduction=2):
        super(sAttentLayer2, self).__init__()
        
        self.input_names = training_params['input_names']

        # reduce the temporal dimension
        groups = 1
#         self.conv = nn.Sequential(
#             nn.Conv2d(channel, channel, kernel_size=KERNEL_SIZE, stride=1, padding='same', bias=True, groups=groups),
#             nn.BatchNorm2d(channel),
#             nn.ReLU(inplace=True),
#             nn.AvgPool2d(kernel_size=KERNEL_SIZE, stride=(KERNEL_SIZE,1), padding=(0,1)),
#             nn.Conv2d(channel, channel, kernel_size=KERNEL_SIZE, stride=1, padding='same', bias=True, groups=groups),
#             nn.BatchNorm2d(channel),
# #             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, N_freq)),

# #             nn.AvgPool2d(kernel_size=KERNEL_SIZE, stride=(KERNEL_SIZE,1), padding=(0,1)),
#         )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, N_freq))
        
#         self.fc = nn.Linear(N_freq, 1, bias=True)
        self.fc = nn.Linear(1, 1, bias=True)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

#         self.activation = nn.Tanh()
        self.activation = nn.Sigmoid()
        
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        
        # plan1: try conv + avgpool + maxpool + softmax (sub 4 fail)
        # plan2: try avgpool + maxpool + softmax
        # plan3: try conv + avgpool + normalized maxpool + softmax
        # plan4: try avgpool + normalized maxpool + softmax

        # plan5: try conv + avgpool + maxpool + activation + softmax
        # plan6: try avgpool + maxpool + softmax
        # plan7: try conv + avgpool + normalized maxpool + activation + softmax
        # plan8: try avgpool + normalized maxpool + activation + softmax

        
        outputs = []
        for i_input, input_name in enumerate(self.input_names):
            # (N, 2, 20, 58) -> (N, 1, 20, 58) -> (N, 1, 2, 58) -> (N, 1, 1, 58)
#             interm = self.conv(x[:,[i_input],:,:]) 
#             print(interm)
            interm = self.avg_pool( x[:,[i_input],:,:] )
#             print(interm.size())
#             print(interm.sum(dim=-1,keepdim=True).size())
    
            # normalization: (N, 1, 1, 58) -> (N, 1, 1, 58) -> (N, 1, 1, 1)/(N, 1, 1, 1) -> (N, 1, 1, 1) 
            interm = self.max_pool(interm) / interm.sum(dim=-1,keepdim=True)
#             interm = interm / interm.sum(dim=-1,keepdim=True) # (N, 1, 1, 58)
#             print(interm.sum(dim=-1,keepdim=True))
#             print(interm[:,0,0,:].size())

            
#             interm = fc(interm.squeeze())
#             interm = interm / interm.sum(dim=-1,keepdim=True) # (N, 1, 1, 58)
#             interm = self.fc(interm[:,0,0,:])[:,0]

#             print(interm.size())

#             interm = self.max_pool(interm)
#             interm = self.max_pool(interm) / interm.sum(dim=-1,keepdim=True) # (N, 1, 1, 1)

#             interm = interm[:,0,0,[0]]
            interm = interm[:,0,0,0]

#             print(interm.size())
#             interm = self.fc(interm)[:,0]
#             interm = (interm-0.5)*2
    
    
#             interm= self.activation(interm) # (N, 1, 1, 1)  -> (N,) 
#             interm = interm.squeeze() # (N, 1, 1, 1)  -> (N,) 
            outputs.append(interm)

            
#         weights = []
#         for i_input, input_name in enumerate(self.input_names):
#             weights.append(outputs[input_name])
        
        outputs = torch.stack(outputs).T
        
        weights = self.softmax(outputs)
#         weights = self.activation(outputs)
        
#         out = x
#         print(weights.size(), x.size())
        
        out = torch.sum(x * weights[:,:,None,None].expand_as(x), dim=1,keepdim=True)
#         for i_input, input_name in enumerate(self.input_names):
#             x[:,[i_input],:,:] = x[:,[i_input],:,:] * weights[:,i_input][:,None,None,None].expand_as(x[:,[i_input],:,:])

#         out = torch.sum(x, dim=1, keepdim=True)

        return out, weights



# added on 7/5/2022 
class Late_UNet(nn.Module):
    """
    Late UNet
    This model has dedicated CNN for each modality. 
    """
    def __init__(self, training_params=None):
        super(Late_UNet, self).__init__()

        self.input_names = training_params['input_names']
        
        # one CNN for one modality
        self.unets = nn.ModuleDict()
        for input_name in self.input_names:
            self.unets[input_name] = UNet(in_ch=1, out_ch=2, n1=training_params['N_channels'], training_params=training_params)
#         self.unet = UNet(in_ch=1, out_ch=2, n1=training_params['N_channels'], training_params=training_params)
            

#         self.attent_block = sAttentLayer(N_freq=training_params['data_dimensions'][-1])
    
    def forward(self, x):
        # x dim: torch.rand(32,4,20,58)
        
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
        output0 = torch.concat(output0, axis=1)
        output0 = torch.mean(output0, axis=1, keepdim=True)

#         output1 = self.SE_block['1'](torch.concat(output1, axis=1))
        output1 = torch.concat(output1, axis=1)
        output1 = torch.mean(output1, axis=1, keepdim=True)

        # out dim: (N_batch, 2, 20, 58)
        output = torch.concat([output0, output1], axis=1)
        
        
        return output
        
        




regressor_dict = {
    'DominantFreq_regressor': DominantFreq_regressor
}
model_dict = {
    'Late_UNet': Late_UNet,
    'UNet': UNet,
    'baseline': UNet,
    'AT_block': UNet,
    'Attention_UNet': UNet,
}





