import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys

from xgboost import XGBRegressor

# from EE_extension.models_CNN import *
# from EE_extension.models_CNN2 import *
# from VO2_extension.models_resnet import *
from VO2_extension.models_CNNlight import *
from VO2_extension.SDT import *



# class CardioRespXGBRegression(nn.Module):
#     def __init__(self, training_params, num_classes=10, input_dim=1, feature_dim=0):
#         super(CardioRespXGBRegression, self).__init__()
        
#         concat_dim = input_dim+feature_dim
        
#         self.vanilla_regressor = vanilla_regression(training_params, num_classes=num_classes, input_dim=None, feature_dim=concat_dim, hidden_dim=200, n_layers=5)

#         self.get_attention = attention_module(training_params)
#         self.fft_layer = FFT_layer()
        
#         self.xf_dict = training_params['xf_dict']
#         self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
#         self.mask = torch.from_numpy(self.xf_dict['mask'])

#     def forward(self, x, feature, attention_sig=None):

#         # print(attention[0,:])
#         x_fft = self.fft_layer(x)
#         x_fft = x_fft[:,:,self.mask]
        
#         # sys.exit()
#         if attention_sig is not None:
#             attention = self.get_attention(attention_sig)            
#             # attention = attention > torch.mean(attention, axis=-1, keepdim=True)

#             # TODO: check if x and attention have the same size
#             x_fft = x_fft * attention
            
#         x_fft = torch.sum(x_fft, axis=-1)
#         # print(x_fft[:5,:])
#         # print('feature.size()', feature.size())
#         feature = feature.reshape(feature.size(0), -1)
#         # print(feature.size())
#         out = torch.cat((x_fft, feature), 1)
#         # print('out', out.size(), out)
#         # print('feature', feature.size(), feature)
#         out = self.vanilla_regressor(x_fft, out)
        
        
        
#         # sys.exit()
#         return out


    
    

class CardioRespRegression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=1, feature_dim=0):
        super(CardioRespRegression, self).__init__()
        # self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        concat_dim = input_dim+feature_dim
        
        self.vanilla_regressor = vanilla_regression(training_params, num_classes=num_classes, input_dim=None, feature_dim=concat_dim, hidden_dim=200, n_layers=5)
        
        self.get_attention = attention_module(training_params)
        self.fft_layer = FFT_layer()
        
        self.xf_dict = training_params['xf_dict']
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])

    def forward(self, x, feature, attention_sig=None):

        # print(attention[0,:])
        x_fft = self.fft_layer(x)
        x_fft = x_fft[:,:,self.mask]
        
        # sys.exit()
        if attention_sig is not None:
            attention = self.get_attention(attention_sig)            
            # attention = attention > torch.mean(attention, axis=-1, keepdim=True)

            # TODO: check if x and attention have the same size
            x_fft = x_fft * attention
            
        x_fft = torch.sum(x_fft, axis=-1)
        # print(x_fft[:5,:])
        # print('feature.size()', feature.size())
        feature = feature.reshape(feature.size(0), -1)
        # print(feature.size())
        concat_feature = torch.cat((x_fft, feature), 1)
        # print('out', out.size(), out)
        # print('feature', feature.size(), feature)
        out = self.vanilla_regressor(x_fft, concat_feature)
        # sys.exit()
        # print(feature[1,:])
        # print(concat_feature[1,:])
        return out, concat_feature


class get_SNR(nn.Module):
    """
    This function comptues the SNR for deep spectral features
    
    """

    def __init__(self, training_params):
        super(get_SNR, self).__init__()
        self.fft_layer = FFT_layer()
        
        self.xf_dict = training_params['xf_dict']
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])

        self.avg_pool = nn.AdaptiveAvgPool1d((self.xf_masked.shape[0]))
        self.max_pool = nn.AdaptiveMaxPool1d((1))

    def forward(self, sig):
        # print('attention_module')
        # attention dim: (N_batch, 1, N_spectral)
        spec = self.fft_layer(sig)
        spec = spec[:,:,self.mask]
        spec = self.avg_pool( spec )
        # SNR dim: (N_batch, N_ch, 1)
        SNR = self.max_pool(spec) / spec.sum(dim=-1,keepdim=True) # will get a scalar for an instance, a channel
        
        return SNR
    
    
    
    
class attention_module(nn.Module):
    def __init__(self, training_params):
        super(attention_module, self).__init__()
        self.fft_layer = FFT_layer()
        
        self.xf_dict = training_params['xf_dict']
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])
        
        freq_smooth_dur = 5 # bpm
        
        # print(np.diff(self.xf_masked.data.numpy()))
        freq_smooth_win = round(freq_smooth_dur / np.mean(np.diff(self.xf_masked.data.numpy())))
        self.avg_pool = MyAvgPool1dPadSame(kernel_size=freq_smooth_win, stride=1)

    def forward(self, sig):
        # print('attention_module')
        # attention dim: (N_batch, 1, N_spectral)
        attention = self.fft_layer(sig)
        attention = attention[:,:,self.mask]
        
        attention = self.avg_pool(attention)
        
        # attention dim: (N_batch, 1, N_spectral)
        attention = attention / torch.sum(attention, axis=-1)[:,None]
        
        # create the mask
        attention = attention > torch.mean(attention, axis=-1, keepdim=True)
        
        # sys.exit()
        return attention

    
class SingleLayer_regression(nn.Module):
    def __init__(self, input_dim=10, output_dim=50):
        super(SingleLayer_regression, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.bn(self.fc(x)))
        return out

class vanilla_regression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, feature_dim=10, hidden_dim=100, n_layers=5):
        super(vanilla_regression, self).__init__()
        
        self.basiclayer_list = nn.ModuleList()
        for i_block in range(n_layers-1):
            self.basiclayer_list.append(SingleLayer_regression(input_dim=feature_dim, output_dim=hidden_dim))
            feature_dim = hidden_dim
            
        self.basiclayer_list.append(nn.Linear(feature_dim, num_classes))

    def forward(self, x, feature, attention_sig=0):
        
        out = feature.reshape(feature.size(0), -1)
        
        for i_layer in range(len(self.basiclayer_list)):
            net = self.basiclayer_list[i_layer]
            out = net(out)

        return out

# class vanilla_model(nn.Module):
#     def __init__(self, training_params=None):
#         super(vanilla_model, self).__init__()
#         self.N_features = len(training_params['feature_names'])

#         self.regressors = nn.ModuleDict()
#         self.regressors[self.main_task[0]+'-scgZ'] = vanilla_regression(training_params=training_params, feature_dim=self.N_features)
#         self.main_task = training_params['main_task']

#         pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        
#     def forward(self, x, feature):

#         for regressor_name in self.regressors.keys():
#             output[regressor_name] = self.regressors[regressor_name](feature)

#         return output, feature_out
        
        
        


class attention_module(nn.Module):
    def __init__(self, training_params):
        super(attention_module, self).__init__()
        self.fft_layer = FFT_layer()
        
        self.xf_dict = training_params['xf_dict']
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])
        
        freq_smooth_dur = 5 # bpm
        
        # print(np.diff(self.xf_masked.data.numpy()))
        freq_smooth_win = round(freq_smooth_dur / np.mean(np.diff(self.xf_masked.data.numpy())))
        self.avg_pool = MyAvgPool1dPadSame(kernel_size=freq_smooth_win, stride=1)

    def forward(self, sig):
        # print('attention_module')
        # attention dim: (N_batch, 1, N_spectral)
        attention = self.fft_layer(sig)
        attention = attention[:,:,self.mask]
        
        attention = self.avg_pool(attention)
        
        # attention dim: (N_batch, 1, N_spectral)
        attention = attention / torch.sum(attention, axis=-1)[:,None]
        
        # create the mask
        attention = attention > torch.mean(attention, axis=-1, keepdim=True)
        
        # sys.exit()
        return attention

class cardioresp_multiverse(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(cardioresp_multiverse, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        self.input_names = training_params['input_names']
        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        self.n_classes = 1
        
        feature_extractor = training_params['feature_extractor']
        regressor = training_params['regressor']
        auxillary_regressor = training_params['auxillary_regressor']
        self.auxillary_tasks = training_params['auxillary_tasks']
        self.main_task = training_params['main_task'][0]
        self.attention_sig_name = training_params['attention_sig_name']
        self.output_sig_name = training_params['output_sig_name']
        
        self.N_features = len(training_params['feature_names'])
        
        self.feature_extractors = nn.ModuleDict()
        
        # for input_name in self.input_names:
        #     if 'merged' in input_name:
        #         self.feature_extractors[input_name] = feature_extractor(training_params=training_params, input_channel=1)
        #     else:
        #         self.feature_extractors[input_name] = feature_extractor(training_params=training_params, input_channel=1)

#             if 'merged' in input_name: # the new way to select PPG/SCG channel 
                
        
        self.feature_extractors = nn.ModuleDict(
            [[input_name, feature_extractor(training_params=training_params, input_channel=1)] for input_name in self.input_names]
        )
        
        feature_out_dim = 0
        for input_name in self.feature_extractors.keys():
            # feature_out_dim += self.feature_extractors[input_name].feature_out_dim
            feature_out_dim = self.feature_extractors[input_name].feature_out_dim
        
        # print(self.auxillary_tasks)
        auxillary_task = self.auxillary_tasks[0]
        
        self.auxillary_regressors = nn.ModuleDict(
            [[auxillary_task+'-{}'.format(input_name), auxillary_regressor(training_params=training_params)] for input_name in self.input_names]
        )
        
        # self.auxillary_regressors = nn.ModuleDict(
        #     [[task_name, auxillary_regressor(training_params=training_params, num_classes=self.n_classes)] for task_name in self.auxillary_tasks]
        # )
        
        self.regressors = nn.ModuleDict()
        
        
        self.regressors[self.main_task+'-'+self.output_sig_name] = regressor(training_params=training_params, num_classes=self.n_classes, input_dim=len(self.input_names)-1, feature_dim=self.N_features)
        # self.regressors[self.main_task+'-scgZ'] = regressor(training_params=training_params, num_classes=self.n_classes, input_dim=training_params['xf_dict']['xf_masked'].shape[0], feature_dim=self.N_features)
        

#         feature_out_dim = 0
# #         for input_name in self.input_names:
#         for input_name in self.feature_extractors.keys():
#             print(input_name, self.feature_extractors[input_name].feature_out_dim)
# #             feature_out_dim += self.feature_extractors[input_name].feature_out_dim
#             feature_out_dim = self.feature_extractors[input_name].feature_out_dim

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        
    def forward(self, x, feature):
        # if len(feature.size())==3:
        #     feature = feature[:,:,0]
            
        output = {}
        deep_feature = {}
#         device = self.dummy_param.device
        
        for i, input_name in enumerate(self.feature_extractors.keys()):
            deep_feature[input_name] = (self.feature_extractors[input_name](x[:, [i], :]))
            # print('feature_out size', feature_out[input_name].size())
            # print('self.regressors.keys()',  self.regressors.keys())
            # for regressor_name in self.regressors.keys():
            #     output[regressor_name+'-{}'.format(input_name)] = self.regressors[regressor_name](feature_out[input_name], feature)


            # if 'ECG' in input_name:
                
            if input_name == self.attention_sig_name:
                attention_sig = deep_feature[input_name]

            for regressor_name in self.auxillary_regressors.keys():
                if input_name in regressor_name:
                    if input_name == self.attention_sig_name:
                        output[regressor_name] = self.auxillary_regressors[regressor_name](deep_feature[input_name], attention_sig=attention_sig)
                    else:
                        output[regressor_name] = self.auxillary_regressors[regressor_name](deep_feature[input_name])
            
                # attention = feature_out[input_name] / torch.sum(feature_out[input_name], axis=1)[:,None]
                # print('debugging attention')
                # print(feature_out[input_name].size())
                # print(attention.size(), attention)

        # feature_out_SCG = 
        # print('feature_out', feature_out)
        # print('output', output)
        
        feature_scg = []
        for i, input_name in enumerate(deep_feature.keys()):
            if self.output_sig_name in input_name:
                feature_scg.append(deep_feature[input_name])
        
        # print(feature_out[input_name].size())
        feature_scg = torch.cat(feature_scg, 1)
#         print(feature_scg.size())
        
#         sys.exit()


        for regressor_name in self.regressors.keys():
            # print(self.main_task, regressor_name)
            if self.main_task in regressor_name:
                # print('forward pass (main)', regressor_name)
                
                # output[regressor_name] = self.regressors[regressor_name](feature_scg, feature, attention_sig=attention_sig)
                output[regressor_name], concat_feature = self.regressors[regressor_name](feature_scg, feature, attention_sig=attention_sig)


                # for i, input_name in enumerate(self.input_names):
                #     if 'scg' in input_name:
                #     # print('scg')
                #         # output[regressor_name] = self.regressors[regressor_name](feature_out[input_name], feature)
                #         output[regressor_name] = self.regressors[regressor_name](feature_out[input_name], feature, attention_sig=attention_sig)
            if regressor_name in self.auxillary_tasks:
                # print('forward pass (auxillary)', regressor_name)
                output[regressor_name] = self.regressors[regressor_name](deep_feature[input_name])


            # print(output_name)
            # for regressor_name in self.regressors.keys():
            #     print(regressor_name)
            # if 'scg' in output_name.:
            # output['domain-{}'.format(input_name)] = self.domain_classifier(feature_out[input_name], self.adversarial_weight)

        # print('output, feature_out', output, feature_out)
        # sys.exit()
        # print('feature_out', feature_out)
        # for out_name in output.keys():
        #     print(out_name, output[out_name].size())

        # print(feature_out)
        # return output, feature_out
        return output, deep_feature, concat_feature
        
        
        
#         feature_out = {}

#         print(x)
#         for i, input_name in enumerate(self.feature_extractors.keys()):
#             feature_out[input_name] = (self.feature_extractors[input_name](x[input_name]))
            
#         feature_fused = self.fusion_layer(feature_out)
            
#         output = {}
#         for regressor_name in self.regressors.keys():
#             output[regressor_name] = self.regressors[regressor_name](feature_fused, feature)

#         return output


class FFT_layer(nn.Module):
    """
    FFT layer to transform 1D signal timeseries to spectral domain
    TODO: remove all 
    Referece: M. Chan et al., "Estimating Heart Rate from Seismocardiogram Signal using a Novel Deep Dominant Frequency Regressor and Domain Adversarial Training," 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), Taipei, Taiwan, 2022, pp. 158-162, doi: 10.1109/BioCAS54905.2022.9948650.
    """
    def __init__(self):
        super(FFT_layer, self).__init__()
        # print(training_params)
        # print(training_params['xf_dict'], training_params['xf_dict']['xf_masked'], training_params['xf_dict']['mask'])
        # self.xf_dict = training_params['xf_dict']
        # self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        # self.mask = torch.from_numpy(self.xf_dict['mask'])
        # self.xf_masked = xf_masked.float()
        # self.dominantFreq_detect = training_params['dominantFreq_detect']
        
    def forward(self, x):
        # FFT layer, mask it, reshape it, normalize feature to 1
        # out has the same dimension as xf_masked
        
        out = torch.fft.fft(x) # compute fft over the last axis
        NFFT = x.size()[-1]
        out = 2.0/NFFT  * (torch.abs(out[:,:,:NFFT //2])**2) # normalize based on number of spectral feature
        # print('out 1', out.size())
        # out = out[:,:,self.mask]
        # print('out 2', out.size())
        # sys.exit()
        # out = out.reshape(out.size(0), -1)
        # print('out 3', out.size())

        return out

class DominantFreqRegression(nn.Module):
    """
    The famous Dominant frequency Regressor developed by Chan et al. This model takes spectral features and compute the expectation within the xf_mask frequency range. This regression replaces the fully connected layers that aren't prone to overfitting. Note that there are NO learnable parameters in this layer.
    
    Referece: M. Chan et al., "Estimating Heart Rate from Seismocardiogram Signal using a Novel Deep Dominant Frequency Regressor and Domain Adversarial Training," 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), Taipei, Taiwan, 2022, pp. 158-162, doi: 10.1109/BioCAS54905.2022.9948650.
    """
    def __init__(self, training_params):
        super(DominantFreqRegression, self).__init__()
        
        self.xf_dict = training_params['xf_dict']
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])
        # self.xf_masked = xf_masked.float()
        
        # self.dominantFreq_detect = training_params['dominantFreq_detect']
        self.fft_layer = FFT_layer()
        self.get_attention = attention_module(training_params)

    def forward(self, x, attention_sig=None):
        # x is in time domain
        
        # # FFT layer, mask it, reshape it, normalize feature to 1       
        out = self.fft_layer(x)
        out = out[:,:,self.mask]
        
        if attention_sig is not None:
            attention = self.get_attention(attention_sig)            
            # TODO: check if x and attention have the same size
            # print(out.size(), attention.size())
            # sys.exit()
            out = out * attention
            


        # out dim: (N_batch, N_ch, N_feature)
        out = out / torch.sum(out, axis=-1, keepdim=True)
        
        # TODO: make sure I am computing the expectation correctly
        # xf_masked_torch dim: (1, 1, N_spectral)
        xf_masked_torch = self.xf_masked.to(x.device)[None,None,:]
        
        # expand xf_masked_torch so it has the same size of out
        # xf_masked_torch dim:  (N_batch, N_ch, N_spectral)
        xf_masked_torch = xf_masked_torch.expand(list(out.size()))
        
        # print(xf_masked_torch.size())

        # comptue expectation at the last dimension, assuming out dimension is sum-normalized
        # out dim: (N_batch, N_ch, 1)
        out = torch.sum(out * xf_masked_torch, axis=-1)

        # print(out.size())

        
        # sys.exit()
        # out = torch.sum(out * self.xf_masked.to(x.device), axis=1)[:,None]
        return out
    
    

class sWeightLayer(nn.Module):
    """spectral weighting block without learnable parameters"""
    def __init__(self, training_params, N_freq=58, channel=1, reduction=2):
        super(sAttentLayer2, self).__init__()
        
        self.input_names = training_params['input_names']

        groups = 1

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, N_freq))
        self.avg_pool = MyAvgPool1dPadSame(kernel_size=2, stride=1)
        self.max_pool = nn.AdaptiveMaxPool2d((1,))
        self.activation = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        outputs = []
        for i_ch in range(x.size(1)): # the 1st dim stores the number of channels
            
            # (N, 2, 58) -> (N, 1, 58) -> (N, 1, 58) -> (N, 1, 58)
            interm = self.avg_pool( x[:,[i_ch],:] )   
            
            # normalization: (N, 1, 58) -> (N, 1, 58) -> (N, 1, 1)/(N, 1, 1) -> (N, 1, 1) 
            interm = self.max_pool(interm) / interm.sum(dim=-1,keepdim=True)

            interm = interm[:,0,0]

            outputs.append(interm)

        outputs = torch.stack(outputs).T
        weights = self.softmax(outputs)

        out = torch.sum(x * weights[:,:,None].expand_as(x), dim=1,keepdim=True)

        return out, weights

    

class MultiTaskLoss(nn.Module):
    def __init__(self, training_params):
        super(MultiTaskLoss, self).__init__()
        self.regression_names = training_params['regression_names']
        self.device = training_params['device']

        self.loss_weights  = {}
        self.criterions = {}
        
        # main_task = self.output_names[0]
        # self.main_task = training_params['output_names'][0]
        
        self.main_task = training_params['main_task'][0]
        for regression_name in self.regression_names:
            self.criterions[regression_name] = torch.nn.MSELoss()
            if self.main_task in regression_name:
                self.loss_weights[regression_name] = training_params['loss_weights']['main_task']
            else:
                N_aux_tasks = len(self.regression_names) - 1
                if N_aux_tasks==0:
                    self.loss_weights[regression_name] = 0
                else:
                    self.loss_weights[regression_name] = training_params['loss_weights']['auxillary_task']/N_aux_tasks

    def forward(self, out, label):
        
        # print('out, out.keys()', out, out.keys())
        label = {out_name: label[:, [self.regression_names.index( out_name )]] for out_name in out.keys()}
        
        # print('label, label.keys()', label, label.keys())
        # print('out, out.keys()', out, out.keys())

        losses = {}
        for out_name in out.keys():
            # print(out_name, out[out_name].squeeze(), label[out_name])
            losses[out_name] = self.criterions[ out_name ](out[out_name].squeeze(), label[out_name].to(device=self.device, dtype=torch.float).squeeze())
        # losses = {out_name: self.criterions[ out_name ](out[out_name].squeeze(), label[out_name].to(device=self.device, dtype=torch.float).squeeze()) for out_name in out.keys()}
        
        
        # print(torch.stack([self.loss_weights[ out_name ] * losses[out_name] for out_name in out.keys()]))
        # print(out.keys())
        # sys.exit()

        losses['total'] = torch.sum(torch.stack([self.loss_weights[ out_name ] * losses[out_name] for out_name in out.keys()]))
        
        return losses
# class MultiTaskLoss(nn.Module):
#     def __init__(self, training_params):
#         super(MultiTaskLoss, self).__init__()
#         self.output_names = training_params['output_names']
#         self.device = training_params['device']

#         self.loss_weights  = {}
#         self.criterions = {}
#         # main_task = 'EE_cosmed'
# #         output_names = ['EE_cosmed', 'RR_cosmed', 'HR_patch']
#         main_task = self.output_names[0]

# #         for task in self.tasks:
#         for task in training_params['regressor_names']:
#             self.criterions[task] = torch.nn.MSELoss()
#             if main_task in task:
#                 self.loss_weights[task] = training_params['loss_weights']['main_task']
#             else:
                
#                 N_aux_tasks = len(self.output_names)-1
#                 if N_aux_tasks==0:
#                     self.loss_weights[task] = 0
#                 else:
#                     self.loss_weights[task] = training_params['loss_weights']['auxillary_task']/N_aux_tasks
        
        

#     def forward(self, output, label):
        
#         label = {output_name: label[:, [self.output_names.index( output_name.split('-')[0] )]] for output_name in output.keys()}
        
#         losses = {output_name: self.criterions[ output_name ](output[output_name].squeeze(), label[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}
        
#         losses['total'] = torch.sum(torch.stack([self.loss_weights[ output_name ] * losses[output_name] for output_name in output.keys()]))
        
#         return losses
    
    
    
    
    
    

# class CardioRespSDTRegression(nn.Module):
#     def __init__(self, training_params, num_classes=10, input_dim=1, feature_dim=0):
#         super(CardioRespSDTRegression, self).__init__()
#         # self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
#         concat_dim = input_dim+feature_dim
        
#         self.device = training_params['device']
#         self.SDT_regressor = SDT(input_dim=concat_dim, output_dim=1, depth=20, lamda=1e-3, device=self.device).to(self.device)

#         self.get_attention = attention_module(training_params)
#         self.fft_layer = FFT_layer()
        
#         self.xf_dict = training_params['xf_dict']
#         self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
#         self.mask = torch.from_numpy(self.xf_dict['mask'])

#     def forward(self, x, feature, attention_sig=None):

#         # print(attention[0,:])
#         x_fft = self.fft_layer(x)
#         x_fft = x_fft[:,:,self.mask]
        
#         # sys.exit()
#         if attention_sig is not None:
#             attention = self.get_attention(attention_sig)            
#             # attention = attention > torch.mean(attention, axis=-1, keepdim=True)

#             # TODO: check if x and attention have the same size
#             x_fft = x_fft * attention
            
#         x_fft = torch.sum(x_fft, axis=-1)
#         # print(x_fft[:5,:])
#         # print('feature.size()', feature.size())
#         feature = feature.reshape(feature.size(0), -1)
#         # print(feature.size())
#         out = torch.cat((x_fft, feature), 1)
#         # print('out', out.size(), out)
#         # print('feature', feature.size(), feature)
#         out = self.SDT_regressor(out)
#         # sys.exit()
#         return out
    