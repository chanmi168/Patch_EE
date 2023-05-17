import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys

from xgboost import XGBRegressor

from VO2_extension999.models_CNNlight import *

class cardioresp_multiverse(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(cardioresp_multiverse, self).__init__()
        
        self.input_names = training_params['input_names']
        channel_n = training_params['channel_n']
        self.n_classes = 1
        
        
        self.use_spectral_atn = training_params['use_spectral_atn']
        self.use_channel_atn = training_params['use_channel_atn']

        
        feature_extractor = training_params['feature_extractor']
        regressor = training_params['regressor']
        auxillary_regressor = training_params['auxillary_regressor']
        self.auxillary_tasks = training_params['auxillary_tasks']
        self.main_task = training_params['main_task'][0]
        self.N_features = len(training_params['feature_names'])
        
        self.feature_extractors = nn.ModuleDict()
        self.feature_extractors['ECG-HR_patch'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['ECG']), output_channel=channel_n, n_block=3, xf_dict=training_params['HR_xf_dict'])
        self.feature_extractors['SCG-HR_patch'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['SCG']), output_channel=channel_n, n_block=3, xf_dict=training_params['HR_xf_dict'])


        self.feature_extractors['PPG-RR_cosmed'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['PPG']), output_channel=channel_n, n_block=4, xf_dict=training_params['RR_xf_dict'])
        self.feature_extractors['SCG-RR_cosmed'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['SCG']), output_channel=channel_n, n_block=4, xf_dict=training_params['RR_xf_dict'])


        input_name = list(self.feature_extractors.keys())[0]
        feature_out_dim = self.feature_extractors[input_name].feature_out_dim

        
        self.auxillary_regressors = nn.ModuleDict()
        self.auxillary_regressors['merged-HR_patch'] = DominantFreqRegression(training_params, training_params['HR_xf_dict'])
        self.auxillary_regressors['merged-RR_cosmed'] = DominantFreqRegression(training_params, training_params['RR_xf_dict'])

        self.HR_fusion = fusion_module(training_params, training_params['HR_xf_dict'])
        self.RR_fusion = fusion_module(training_params, training_params['RR_xf_dict'])

        self.regressors = nn.ModuleDict()

        self.HR_xf_dict = training_params['HR_xf_dict']
        self.RR_xf_dict = training_params['RR_xf_dict']
        
        HR_xf_dicts_dim = self.HR_xf_dict['xf_masked'].shape[0]
        RR_xf_dicts_dim = self.RR_xf_dict['xf_masked'].shape[0]
        # concat_dim = channel_n*2*2 + 2 + 3
        
        # concat_dim = (HR_xf_dicts_dim + RR_xf_dicts_dim) * channel_n * 2 + 2 + 3 # spectral features x channel_n * 2 modalities each + HR + RR + demographic
        # concat_dim = (HR_xf_dicts_dim + RR_xf_dicts_dim) * channel_n * 2 + 2 # spectral features x channel_n * 2 modalities each + HR + RR + demographic
        # concat_dim = hidden_dim + 3

        self.hidden_dim = channel_n
        # self.avg_pooling = MyAvgPool1dPadSame(kernel_size=5, stride=1)
        self.max_pooling =  nn.AdaptiveMaxPool1d(1)
        # self.same_pooling = MyAvgPool1dPadSame(kernel_size=5, stride=1)
        # self.avg_pooling =  nn.AdaptiveAvgPool1d(1)

        
        # self.fc = nn.ModuleDict()
        # self.fc['ECG-HR_patch'] = SingleLayer_regression(input_dim=channel_n, output_dim=self.hidden_dim)
        # self.fc['SCG-HR_patch'] = SingleLayer_regression(input_dim=channel_n, output_dim=self.hidden_dim)
        # self.fc['PPG-RR_cosmed'] = SingleLayer_regression(input_dim=channel_n, output_dim=self.hidden_dim)
        # self.fc['SCG-RR_cosmed'] = SingleLayer_regression(input_dim=channel_n, output_dim=self.hidden_dim)

        
        if ('HR_patch' in self.auxillary_tasks) & ('RR_cosmed' in self.auxillary_tasks):
            self.extracted_dim = (self.hidden_dim*2+1)*2
        elif ('HR_patch' in self.auxillary_tasks) & ('RR_cosmed' not in self.auxillary_tasks):
            self.extracted_dim = self.hidden_dim*2+1
        elif ('HR_patch' not in self.auxillary_tasks) & ('RR_cosmed' in self.auxillary_tasks):
            self.extracted_dim = self.hidden_dim*2+1
        else:
            self.extracted_dim = 0
            self.empty_tensor = torch.empty(0).float().to(training_params['device'])
            
        self.regressors[self.main_task] = CardioRespRegression(training_params=training_params, num_classes=self.n_classes, input_dim=self.extracted_dim +3)
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('pytorch_total_params', pytorch_total_params)
        
        
        self.reset_parameters()

    def reset_parameters(self) -> None:       
        # he initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
        
    def apply_attention(self, data, attention, axis=-1):
        out = attention.expand(list(data.size()))*data
        out_avg = torch.sum(out, axis=axis, keepdim=True)
        return out, out_avg
    
    def sum_normalization(self, data, axis=-1):
        return data / torch.sum(data, axis=axis, keepdim=True)
        
    def forward(self, ecg, scg, ppg, feature):
        
        output = {}
        deep_feature = {}
        
        # ecg dim = (N_batch, 1, 6000)
        # scg dim = (N_batch, 3, 6000)
        # ppg dim = (N_batch, 4, 6000)

        # deep_feature['ECG-HR_patch'] and deep_feature['SCG-RR_cosmed'] dim: = (N_batch, 8, 750)
        deep_feature['ECG-HR_patch'] = self.feature_extractors['ECG-HR_patch'](ecg)
        deep_feature['SCG-HR_patch'] = self.feature_extractors['SCG-HR_patch'](scg)

        # deep_feature['PPG-RR_cosmed'] and deep_feature['SCG-RR_cosmed'] dim: = (N_batch, 8, 375)
        deep_feature['PPG-RR_cosmed'] = self.feature_extractors['PPG-RR_cosmed'](ppg)
        deep_feature['SCG-RR_cosmed'] = self.feature_extractors['SCG-RR_cosmed'](scg)
        
        
        
        
        # concat multi-modal ECG and SCG data (time domain)
        merged_HR_deep_features = torch.cat([deep_feature['ECG-HR_patch'], deep_feature['SCG-HR_patch']], axis=1)
        
        # fuse these 8 channel features, compute their fused fft, compute channel and spectral attentions
        features_fft_HR_raw, fused_fft_HR, channel_attention_HR, spectral_attention_HR = self.HR_fusion(merged_HR_deep_features)
        features_fft_HR = features_fft_HR_raw
        
        # apply channel attention then estimate HR
        if self.use_channel_atn:
            output['merged-HR_patch'] = self.auxillary_regressors['merged-HR_patch'](self.apply_attention(self.sum_normalization(merged_HR_deep_features, axis=-1), channel_attention_HR, axis=1)[0])
        else:
            output['merged-HR_patch'] = self.auxillary_regressors['merged-HR_patch'](merged_HR_deep_features)


        # concat multi-modal PPG and SCG data (time domain)
        merged_RR_deep_features = torch.cat([deep_feature['PPG-RR_cosmed'], deep_feature['SCG-RR_cosmed']], axis=1)
        
        # fuse these 8 channel features, compute their fused fft, compute channel and spectral attentions
        features_fft_RR_raw, fused_fft_RR, channel_attention_RR, spectral_attention_RR = self.RR_fusion(merged_RR_deep_features)
        features_fft_RR = features_fft_RR_raw

        # apply channel attention then estimate RR
        # TODO: fix use_channel_atn
        if self.use_channel_atn:
            output['merged-RR_cosmed'] = self.auxillary_regressors['merged-RR_cosmed'](self.apply_attention(self.sum_normalization(merged_RR_deep_features, axis=-1), channel_attention_RR, axis=1)[0])
        else:
            output['merged-RR_cosmed'] = self.auxillary_regressors['merged-RR_cosmed'](merged_RR_deep_features)
        
        
        # apply spectral attention if needed
        if self.use_spectral_atn:
            features_fft_HR, _ = self.apply_attention(features_fft_HR, spectral_attention_HR, axis=-1)
            features_fft_RR, _ = self.apply_attention(features_fft_RR, spectral_attention_RR, axis=-1)

        # get condensed features
        features_fft_HR = self.max_pooling(features_fft_HR)
        features_fft_RR = self.max_pooling(features_fft_RR)
        # features_fft_HR = self.max_pooling(self.avg_pooling(features_fft_HR))
        # features_fft_RR = self.max_pooling(self.avg_pooling(features_fft_RR))

        # flatten
        features_fft_HR = features_fft_HR.reshape(features_fft_HR.size(0),-1)
        features_fft_RR = features_fft_RR.reshape(features_fft_RR.size(0),-1)

        # combine deep features with vital signs
        if ('HR_patch' in self.auxillary_tasks) & ('RR_cosmed' in self.auxillary_tasks):
            merged_deep_features = torch.cat([features_fft_HR, features_fft_RR, output['merged-HR_patch'], output['merged-RR_cosmed']], axis=-1)

        elif ('HR_patch' in self.auxillary_tasks) & ('RR_cosmed' not in self.auxillary_tasks):
            merged_deep_features = torch.cat([features_fft_HR, output['merged-HR_patch']], axis=-1)
            del output['merged-RR_cosmed']
        
        elif ('HR_patch' not in self.auxillary_tasks) & ('RR_cosmed' in self.auxillary_tasks):
            merged_deep_features = torch.cat([features_fft_RR, output['merged-RR_cosmed']], axis=-1)
            del output['merged-HR_patch']

        else:
            merged_deep_features = self.empty_tensor
            del output['merged-HR_patch']
            del output['merged-RR_cosmed']

        # regression for target variable
        output[self.main_task], concat_feature = self.regressors[self.main_task](merged_deep_features, feature)

        attention_dict = {
            'spectral_attention_HR': spectral_attention_HR.data.detach().cpu().numpy(),
            'spectral_attention_RR': spectral_attention_RR.data.detach().cpu().numpy(),
            'channel_attention_HR': channel_attention_HR.data.detach().cpu().numpy(),
            'channel_attention_RR': channel_attention_RR.data.detach().cpu().numpy(),
            'features_fft_HR_raw': features_fft_HR_raw.data.detach().cpu().numpy(),
            'features_fft_RR_raw': features_fft_RR_raw.data.detach().cpu().numpy(),
        }
        return output, deep_feature, concat_feature, attention_dict


    
class CardioRespRegression(nn.Module):    
    """
    This function makes final regression on the deep features extracted.
    """
    def __init__(self, training_params, num_classes=10, input_dim=10):
        super(CardioRespRegression, self).__init__()
        self.vanilla_regressor = vanilla_regression(training_params, num_classes=num_classes, input_dim=input_dim, hidden_dim=training_params['hidden_dim'], n_layers=3)

    def forward(self, merged_deep_features, feature):
        concat_feature = torch.cat([merged_deep_features, feature[:,:3]], axis=-1)
        out = self.vanilla_regressor(concat_feature)

        return out, concat_feature
    

    
    
class fusion_module(nn.Module):
    """
    This function fuse multi-modal data.
    """

    def __init__(self, training_params, xf_dict):
        super(fusion_module, self).__init__()
        
        self.xf_dict = xf_dict
        self.spectral_mask = torch.from_numpy(self.xf_dict['mask'])
        self.fft_layer = FFT_layer()
        
        self.get_SNR_attention = SNR_attention_module()
        self.get_spectral_attention = spectral_attention_module(xf_dict)

        # self.avg_pool = MyAvgPool1dPadSame(kernel_size=2, stride=1)
        # self.max_pool = nn.AdaptiveMaxPool1d((1))
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x_concat):
        # x_concat dim: (B, C, T)

        # first, compute the fft of the deep features
        # (B,2C,f)
        features_fft = self.fft_layer(x_concat)
        features_fft = features_fft[:,:,self.spectral_mask]
        
        # compute channel attention and fuse features_fft
        # sum((B,2C,f) x (B,2C,1), axis=1) => (B,1,f)
        # print(self.get_SNR_attention(features_fft).size(), features_fft.size())
        channel_attention = self.get_SNR_attention(features_fft)
        feature_fft = torch.sum(channel_attention.expand(list(features_fft.size()))*features_fft, axis=1, keepdim=True)

        # compute spectral attention
        # print( self.get_spectral_attention(feature_fft).size(), feature_fft.size())
        spectral_attention = self.get_spectral_attention(feature_fft).expand(list(feature_fft.size()))
        # apply spectral attention (remove irrelevant info)
        # (B,1,f) x (B,1,f) => (B,1,f)
        feature_fft = spectral_attention*feature_fft
        return features_fft, feature_fft, channel_attention, spectral_attention


class SNR_attention_module(nn.Module):
    """
    This function comptues the SNR (0~1) for each channel of deep spectral feature
    
    Reference | 
    [1] G. de Haan and V. Jeanne, “Robust Pulse Rate From Chrominance-Based rPPG,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 10, pp. 2878–2886, Oct. 2013, doi: 10.1109/TBME.2013.2266196.
    """

    def __init__(self):
        super(SNR_attention_module, self).__init__()

        self.avg_pool = MyAvgPool1dPadSame(kernel_size=2, stride=1)
        self.max_pool = nn.AdaptiveMaxPool1d((1))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, sig_fft):
        # print('SNR attention_module')
        # sig dim: (N_batch, 1, N_spectral)
        spec = self.avg_pool( sig_fft )
        # SNR dim: (N_batch, N_ch, 1)
        SNR = self.max_pool(spec) / torch.sum(spec, axis=-1, keepdim=True)
        SNR = torch.log(SNR)/ torch.log(torch.tensor(10))
        SNR_attention = self.softmax(SNR)

        return SNR_attention


class spectral_attention_module(nn.Module):
    """
    This function create a mask in the spectral domain (spectral hard attention)
    """
    def __init__(self, xf_dict):
        super(spectral_attention_module, self).__init__()
        
        # if xf_dict is None:
        #     self.xf_dict = training_params['xf_dict']
        # else:
        self.xf_dict = xf_dict
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])
        
        # freq_smooth_dur = 5 # bpm
        # freq_smooth_win = round(freq_smooth_dur / np.mean(np.diff(self.xf_masked.data.numpy())))
        self.avg_pool = MyAvgPool1dPadSame(kernel_size=5, stride=1)
        
    def forward(self, sig_fft):
        
        # sig_fft dim: (N_batch, 1, N_spectral)

        # print(sig_fft.size())
        attention = self.avg_pool(sig_fft)
        
        # print(attention.size())

        # attention dim: (N_batch, 1, N_spectral)
        attention = attention / torch.sum(attention, axis=-1, keepdim=True)
        # print(attention.size())
        
        # attention = torch.log(attention)/ torch.log(torch.tensor(10))

        # create the mask
        attention = attention > torch.mean(attention, axis=-1, keepdim=True)*0.5
        # print(attention.size())
        
        # sys.exit()
        return attention
    

class get_SNR(nn.Module):
    """
    This function comptues the SNR for deep spectral features.
    Equivalent to channel attention.
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
    
    
    
class SingleLayer_regression(nn.Module):
    def __init__(self, input_dim=10, output_dim=50, p_dropout=0.5):
        super(SingleLayer_regression, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_dim, output_dim)
        # self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        # out = self.relu(self.bn(self.fc(x)))
        # out = self.fc(self.drop(self.relu(self.bn(x))))
        # out = self.fc(self.bn(self.drop(self.relu(x))))
        out = self.fc(self.relu(self.bn(x)))
        # out = self.fc(self.drop(self.relu(self.bn(x))))
        return out

class vanilla_regression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, hidden_dim=100, n_layers=5):
        super(vanilla_regression, self).__init__()
        
        self.basiclayer_list = nn.ModuleList()
        for i_block in range(n_layers-1):
            self.basiclayer_list.append(SingleLayer_regression(input_dim=input_dim, output_dim=hidden_dim))
            input_dim = hidden_dim
            
        # self.basiclayer_list.append(nn.Linear(input_dim, num_classes))
        self.basiclayer_list.append(SingleLayer_regression(input_dim=input_dim, output_dim=num_classes))

    def forward(self, feature, attention_sig=0):
        
        out = feature.reshape(feature.size(0), -1)
        
        for i_layer in range(len(self.basiclayer_list)):
            net = self.basiclayer_list[i_layer]
            out = net(out)
            
        return out



class FFT_layer(nn.Module):
    """
    FFT layer to transform 1D signal timeseries to spectral domain
    TODO: remove all 
    Referece: M. Chan et al., "Estimating Heart Rate from Seismocardiogram Signal using a Novel Deep Dominant Frequency Regressor and Domain Adversarial Training," 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), Taipei, Taiwan, 2022, pp. 158-162, doi: 10.1109/BioCAS54905.2022.9948650.
    """
    def __init__(self):
        super(FFT_layer, self).__init__()
        
    def forward(self, x):
        # FFT layer, mask it, reshape it, normalize feature to 1
        # out has the same dimension as xf_masked
        out = torch.fft.fft(x) # compute fft over the last axis
        NFFT = x.size()[-1]
        out = 2.0/NFFT  * (torch.abs(out[:,:,:NFFT //2])**2) # normalize based on number of spectral feature

        return out

class DominantFreqRegression(nn.Module):
    """
    The famous Dominant frequency Regressor developed by Chan et al. This model takes spectral features and compute the expectation within the xf_mask frequency range. This regression replaces the fully connected layers that aren't prone to overfitting. Note that there are NO learnable parameters in this layer.
    
    Referece: M. Chan et al., "Estimating Heart Rate from Seismocardiogram Signal using a Novel Deep Dominant Frequency Regressor and Domain Adversarial Training," 2022 IEEE Biomedical Circuits and Systems Conference (BioCAS), Taipei, Taiwan, 2022, pp. 158-162, doi: 10.1109/BioCAS54905.2022.9948650.
    """
    def __init__(self, training_params, xf_dict):
        super(DominantFreqRegression, self).__init__()
        
        self.xf_dict = xf_dict
        print(training_params['device'])
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float().to(training_params['device'])
        self.mask = torch.from_numpy(self.xf_dict['mask'])

        self.fft_layer = FFT_layer()
        
    def apply_attention(self, data, attention, axis=-1):
        out = attention.expand(list(data.size()))*data
        out_avg = torch.sum(out, axis=axis, keepdim=True)
        return out, out_avg

    def forward(self, x, spectral_attention=None, channel_attention=None):
        # x is in time domain

        # # FFT layer, mask it, reshape it, normalize feature to 1       
        out = self.fft_layer(x)

        out = out[:,:,self.mask]
        
        # out dim: (N_batch, N_ch, N_feature)
        out = out / torch.sum(out, axis=-1, keepdim=True)
        
        if spectral_attention is not None:
            out, _ = apply_attention(out, spectral_attention, axis=-1)

        if channel_attention is not None:
            _, out = apply_attention(data, channel_attention, axis=1)

        # TODO: make sure I am computing the expectation correctly
        # xf_masked_torch dim: (1, 1, N_spectral)
        xf_masked_torch = self.xf_masked[None,None,:]
        
        # expand xf_masked_torch so it has the same size of out
        # xf_masked_torch dim:  (N_batch, N_ch, N_spectral)
        xf_masked_torch = xf_masked_torch.expand(list(out.size()))

        # comptue expectation at the last dimension, assuming out dimension is sum-normalized
        # out dim: (N_batch, N_ch, 1)
        out = torch.sum(out * xf_masked_torch, axis=-1)

        out = torch.mean(out, axis=1, keepdim=True)
        # sys.exit()
        # out = torch.sum(out * self.xf_masked.to(x.device), axis=1)[:,None]
        return out
    
    


class MultiTaskLoss(nn.Module):
    def __init__(self, training_params):
        super(MultiTaskLoss, self).__init__()
        self.regression_names = training_params['regression_names']
        # self.output_names = training_params['output_names']
        self.device = training_params['device']

        self.loss_weights  = {}
        
        # main_task = self.output_names[0]
        # self.main_task = training_params['output_names'][0]
        
        self.main_task = training_params['main_task'][0]
        
        self.loss_weights = {}
        # self.loss_weights['VO2_cosmed'] = training_params['loss_weights']['main_task']
        
        for task_name in self.regression_names:
            if self.main_task in task_name:
                self.loss_weights[task_name] = training_params['main_loss_weight']
            else:
                if training_params['auxillary_loss_weight']!=0:
                    self.loss_weights[task_name] = training_params['auxillary_loss_weight']

        self.criterions = {}

        for regression_name in self.loss_weights.keys():
            self.criterions[regression_name] = torch.nn.MSELoss()

        self.use_awl = False
        if training_params['adaptive_loss_name']=='awl':
            self.awl = AutomaticWeightedLoss(len(self.criterions.keys()))
            self.use_awl = True

    def forward(self, out, label):
        
        # print('out, out.keys()', out, out.keys())
        # print('self.regression_names', self.regression_names)
        # for out_name in out.keys():
        #     print(self.regression_names.index( out_name ))

        label = {out_name: label[:, [self.regression_names.index( out_name )]] for out_name in self.criterions.keys()}
                # label = {out_name: label[:, [out_name.split('-')[-1]]] for out_name in out.keys()}

        # print('label, label.keys()', label, label.keys())
        # print('out, out.keys()', out, out.keys())

        losses = {}
        # for out_name in out.keys():
        for out_name in self.criterions.keys():
            # print(out_name, out[out_name].squeeze(), label[out_name])
            losses[out_name] = self.criterions[ out_name ](out[out_name].squeeze(), label[out_name].to(device=self.device, dtype=torch.float).squeeze())

        
        if self.use_awl:
            # losses['total'] = self.awl(losses)
            losses['total'] = self.awl(losses, self.loss_weights)
        else:
            losses['total'] = torch.sum(torch.stack([self.loss_weights[ out_name ] * losses[out_name] for out_name in  self.criterions.keys()]))
        
        return losses
    

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    Reference:
        https://github.com/Mikoto10032/AutomaticWeightedLoss
        https://arxiv.org/pdf/1805.06334.pdf
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, x, weights=None):
        loss_sum = 0
        
        for i, key in enumerate(x):
            loss = x[key]
            # print(key, loss)

            if weights is None:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            else:
                loss_sum += 0.5 / ((self.params[i]) ** 2) * loss * weights[key] + torch.log(1 + (self.params[i]) ** 2)

            # print(key, 0.5 / (self.params[i] ** 2))
        return loss_sum
    
    
# class CR_fusion_module(nn.Module):
#     def __init__(self, channel, reduction=2):
#         super(CR_fusion_module, self).__init__()
#         self.SE = SELayer(channel=2) # cardiac and respiratory
        
#     def forward(self, HR_deep_features, RR_deep_features):        
#         # HR_deep_features and RR_deep_features dim: (B,C)
#         # deep_features dim: (B,2,C)
#         deep_features = torch.cat([HR_deep_features[:,None,:],RR_deep_features[:,None,:]], axis=1)
#         # deep_features = self.SE(deep_features)
        
#         # merged_deep_features dim: (B,C)
#         deep_features = torch.mean(deep_features, axis=1)
#         return deep_features
    
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=2):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # print(x.size())
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         # print(y.size())
#         # print(self.fc(y).size(), b, c)
#         y = self.fc(y).view(b, c, 1)
        
#         # print(y.size())

#         return x * y.expand_as(x)