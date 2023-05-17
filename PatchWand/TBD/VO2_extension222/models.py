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
from VO2_extension222.models_CNNlight import *



class cardioresp_multiverse_backup(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(cardioresp_multiverse_backup, self).__init__()
        
        # input_dim = training_params['data_dimensions'][1]
        self.input_names = training_params['input_names']
        # input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        self.n_classes = 1
        
        feature_extractor = training_params['feature_extractor']
        regressor = training_params['regressor']
        auxillary_regressor = training_params['auxillary_regressor']
        self.auxillary_tasks = training_params['auxillary_tasks']
        self.main_task = training_params['main_task'][0]

        # self.output_sig_name = training_params['output_sig_name']
        
        self.N_features = len(training_params['feature_names'])
        
        self.feature_extractors = nn.ModuleDict()
        self.feature_extractors['ECG-HR_patch'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['ECG']), output_channel=1, n_block=3, xf_dict=training_params['HR_xf_dict'])
        self.feature_extractors['SCG-HR_patch'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['SCG']), output_channel=1, n_block=3, xf_dict=training_params['HR_xf_dict'])


        self.feature_extractors['PPG-RR_cosmed'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['PPG']), output_channel=1, n_block=4, xf_dict=training_params['RR_xf_dict'])
        self.feature_extractors['SCG-RR_cosmed'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['SCG']), output_channel=1, n_block=4, xf_dict=training_params['RR_xf_dict'])


        input_name = list(self.feature_extractors.keys())[0]
        feature_out_dim = self.feature_extractors[input_name].feature_out_dim
        
        # print(self.auxillary_tasks)
        # auxillary_task = self.auxillary_tasks[0]
        
        self.auxillary_regressors = nn.ModuleDict()
        self.auxillary_regressors['ECG-HR_patch'] = DominantFreqRegression(training_params, training_params['HR_xf_dict'])
        self.auxillary_regressors['SCG-HR_patch'] = DominantFreqRegression(training_params, training_params['HR_xf_dict'])
        
        self.auxillary_regressors['PPG-RR_cosmed'] = DominantFreqRegression(training_params, training_params['RR_xf_dict'])
        self.auxillary_regressors['SCG-RR_cosmed'] = DominantFreqRegression(training_params, training_params['RR_xf_dict'])
        
        
        self.get_CRQI = CRQI_module()
        self.softmax = nn.Softmax(dim=-1)

        self.get_attentio_HR = attention_module(training_params, training_params['HR_xf_dict'])
        self.get_attentio_RR = attention_module(training_params, training_params['RR_xf_dict'])

        
        self.regressors = nn.ModuleDict()
        self.regressors[self.main_task] = CardioRespRegression(training_params=training_params, num_classes=self.n_classes, input_dim=2, feature_dim=self.N_features)

        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        
    def forward(self, ecg, scg, ppg, feature):

        output = {}
        deep_feature = {}
        
        
        deep_feature['ECG-HR_patch'] = self.feature_extractors['ECG-HR_patch'](ecg)
        deep_feature['SCG-HR_patch'] = self.feature_extractors['SCG-HR_patch'](scg)
        
        deep_feature['PPG-RR_cosmed'] = self.feature_extractors['PPG-RR_cosmed'](ppg)
        deep_feature['SCG-RR_cosmed'] = self.feature_extractors['SCG-RR_cosmed'](scg)


        output['ECG-HR_patch'] = self.auxillary_regressors['ECG-HR_patch'](deep_feature['ECG-HR_patch'])
        output['SCG-HR_patch'] = self.auxillary_regressors['SCG-HR_patch'](deep_feature['SCG-HR_patch'], attention_sig=deep_feature['ECG-HR_patch'])
        output['SCG-HR_patch'] = torch.mean(output['SCG-HR_patch'], -1, keepdim=True)

        output['PPG-RR_cosmed'] = self.auxillary_regressors['PPG-RR_cosmed'](deep_feature['PPG-RR_cosmed'])
        output['SCG-RR_cosmed'] = self.auxillary_regressors['SCG-RR_cosmed'](deep_feature['SCG-RR_cosmed'], attention_sig=deep_feature['PPG-RR_cosmed'])
        output['SCG-RR_cosmed'] = torch.mean(output['SCG-RR_cosmed'], -1, keepdim=True)


        # print(deep_feature['SCG-HR_patch'].size())
        # print(output)
        HR_weight = self.get_CRQI(deep_feature['SCG-HR_patch'])
        # sys.exit()

        RR_weight = self.get_CRQI(deep_feature['SCG-RR_cosmed'])
        
        weights = torch.cat([HR_weight, RR_weight])
        weights = self.softmax(weights)


        output[self.main_task], concat_feature = self.regressors[self.main_task](deep_feature, feature)

        return output, deep_feature, concat_feature
        
        
        
        
        
        

class cardioresp_multiverse(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(cardioresp_multiverse, self).__init__()
        
        self.input_names = training_params['input_names']
        channel_n = training_params['channel_n']
        self.n_classes = 1
        
        feature_extractor = training_params['feature_extractor']
        regressor = training_params['regressor']
        auxillary_regressor = training_params['auxillary_regressor']
        self.auxillary_tasks = training_params['auxillary_tasks']
        self.main_task = training_params['main_task'][0]
        # self.attention_sig_name = training_params['attention_sig_name']
        # self.output_sig_name = training_params['output_sig_name']
        self.N_features = len(training_params['feature_names'])
        
        self.feature_extractors = nn.ModuleDict()
        self.feature_extractors['ECG-HR_patch'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['ECG']), output_channel=channel_n, n_block=3, xf_dict=training_params['HR_xf_dict'])
        self.feature_extractors['SCG-HR_patch'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['SCG']), output_channel=channel_n, n_block=3, xf_dict=training_params['HR_xf_dict'])


        self.feature_extractors['PPG-RR_cosmed'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['PPG']), output_channel=channel_n, n_block=4, xf_dict=training_params['RR_xf_dict'])
        self.feature_extractors['SCG-RR_cosmed'] = feature_extractor(training_params=training_params, input_channel=len(self.input_names['SCG']), output_channel=channel_n, n_block=4, xf_dict=training_params['RR_xf_dict'])


        input_name = list(self.feature_extractors.keys())[0]
        feature_out_dim = self.feature_extractors[input_name].feature_out_dim

        
        
        
        self.auxillary_regressors = nn.ModuleDict()
        self.auxillary_regressors['ECG-HR_patch'] = DominantFreqRegression(training_params, training_params['HR_xf_dict'])
        self.auxillary_regressors['SCG-HR_patch'] = DominantFreqRegression(training_params, training_params['HR_xf_dict'])
        
        self.auxillary_regressors['PPG-RR_cosmed'] = DominantFreqRegression(training_params, training_params['RR_xf_dict'])
        self.auxillary_regressors['SCG-RR_cosmed'] = DominantFreqRegression(training_params, training_params['RR_xf_dict'])
        
        self.auxillary_regressors['merged-HR_patch'] = DominantFreqRegression(training_params, training_params['HR_xf_dict'])
        self.auxillary_regressors['merged-RR_cosmed'] = DominantFreqRegression(training_params, training_params['RR_xf_dict'])

        # self.get_attentio_HR = attention_module(training_params, training_params['HR_xf_dict'])
        # self.get_attentio_RR = attention_module(training_params, training_params['RR_xf_dict'])
        self.HR_fusion = fusion_module(training_params, training_params['HR_xf_dict'])
        self.RR_fusion = fusion_module(training_params, training_params['RR_xf_dict'])

        
        
        
        hidden_dim = 100
        self.HR_mapping = nn.Sequential(
            SingleLayer_regression(input_dim=channel_n*1+1, output_dim=hidden_dim),
            SingleLayer_regression(input_dim=hidden_dim, output_dim=hidden_dim)
        )
        self.RR_mapping = nn.Sequential(
            SingleLayer_regression(input_dim=channel_n*1+1, output_dim=hidden_dim),
            SingleLayer_regression(input_dim=hidden_dim, output_dim=hidden_dim)
        )

        # self.HR_mapping = nn.ModuleList()
        # for i_block in range(2):
        #     self.HR_mapping.append(SingleLayer_regression(input_dim=channel_n*2, output_dim=channel_n*2))
        
        # self.RR_mapping = nn.ModuleList()
        # for i_block in range(2):
        #     self.RR_mapping.append(SingleLayer_regression(input_dim=channel_n*2, output_dim=channel_n*2))
        
        
        
        self.regressors = nn.ModuleDict()
        
        self.CR_fusion = CR_fusion_module(training_params)
        

        # channel_n = training_params['channel_n']
        concat_dim = channel_n*2*2 + 2 + 3
        # concat_dim = hidden_dim + 3
        self.regressors[self.main_task] = CardioRespRegression2(training_params=training_params, num_classes=self.n_classes, input_dim=concat_dim)
        
        # self.regressors['cardiac-'+self.main_task] = CardioRespRegression2(training_params=training_params, num_classes=self.n_classes, input_dim=concat_dim)
        # self.regressors['resp-'+self.main_task] = CardioRespRegression2(training_params=training_params, num_classes=self.n_classes, input_dim=concat_dim)
        
        # self.regressors['cardiac-'+self.main_task] = vanilla_regression(training_params, num_classes=self.n_classes, input_dim=concat_dim, hidden_dim=100, n_layers=2)
        # self.regressors['resp-'+self.main_task] = vanilla_regression(training_params, num_classes=self.n_classes, input_dim=concat_dim, hidden_dim=100, n_layers=2)
        
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        
    def forward(self, ecg, scg, ppg, feature):

        output = {}
        deep_feature = {}
        
        # output dim = (N_batch, N_ch, N_spectral_HR)
        deep_feature['ECG-HR_patch'] = self.feature_extractors['ECG-HR_patch'](ecg)
        deep_feature['SCG-HR_patch'] = self.feature_extractors['SCG-HR_patch'](scg)

        # output dim = (N_batch, N_ch, N_spectral_RR)
        deep_feature['PPG-RR_cosmed'] = self.feature_extractors['PPG-RR_cosmed'](ppg)
        deep_feature['SCG-RR_cosmed'] = self.feature_extractors['SCG-RR_cosmed'](scg)

        
        # concat multi-modal data
        merged_HR_deep_features = torch.cat([deep_feature['ECG-HR_patch'], deep_feature['SCG-HR_patch']], axis=1)
        features_fft_HR, fused_fft_HR, channel_attention_HR, spectral_attention_HR = self.HR_fusion(merged_HR_deep_features)
        features_fft_HR_SCG, _, _, _ = self.HR_fusion(deep_feature['SCG-HR_patch'])
        
        output['ECG-HR_patch'] = self.auxillary_regressors['ECG-HR_patch'](deep_feature['ECG-HR_patch'], spectral_attention_HR)
        output['SCG-HR_patch'] = self.auxillary_regressors['SCG-HR_patch'](deep_feature['SCG-HR_patch'], spectral_attention_HR)


        fused_HR_deep_features = torch.sum(channel_attention_HR.expand(list(merged_HR_deep_features.size()))*merged_HR_deep_features, axis=1, keepdim=True)
        output['merged-HR_patch'] = self.auxillary_regressors['merged-HR_patch'](fused_HR_deep_features)
        # compute channel attention and spectral attention

        # compute fused HR deep features (in time domain)
        # fused_HR_deep_features dim: torch.Size([5, 1, 375])
        # fused_HR_deep_features = torch.sum(channel_attention_HR.expand(list(merged_HR_deep_features.size()))*merged_HR_deep_features, axis=1, keepdim=True)
        # output['merged-HR_patch'] = self.auxillary_regressors['merged-HR_patch'](fused_HR_deep_features)
        # spectral_attention_HR = get_attentio_HR(merged_HR_deep_feature)

        # print('merged_HR_deep_features', merged_HR_deep_features.size())
        # print('features_fft_HR', features_fft_HR.size())
        # print('fused_fft_HR', fused_fft_HR.size())
        # print('channel_attention_HR', channel_attention_HR.size())
        # print('spectral_attention_HR', spectral_attention_HR.size())
        # print('fused_HR_deep_features', fused_HR_deep_features.size())
        # print('output[merged-HR_patch]', output['merged-HR_patch'].size())

        # output['ECG-HR_patch'] = self.auxillary_regressors['ECG-HR_patch'](deep_feature['ECG-HR_patch'])
        # output['SCG-HR_patch'] = self.auxillary_regressors['SCG-HR_patch'](deep_feature['SCG-HR_patch'])
        # output['PPG-HR_patch'] = self.auxillary_regressors['PPG-HR_patch'](deep_feature['PPG-HR_patch'])

        
#         # first, compute the fft of the deep features
#         # (B,2C,f)
#         features_fft = self.fft_layer(output['merged-HR_patch'])
#         features_fft = feature_fft[:,:,self.HR_mask]
        
#         # (B,1,f)
#         feature_fft = torch.sum(get_SNR_attention(features_fft).expand(list(features_fft.size()))*features_fft, axis=1, keepdim=True)

#         # spectral_attention_HR = get_attentio_HR(feature_fft)
#         # weighted average (B,2C,f) x (B,1,f) -> (B,2C,f)
#         # feature_fft dim: (B,1,f)
#         feature_fft = torch.sum(get_attentio_HR(feature_fft).expand(list(feature_fft.size()))*feature_fft, axis=-1, keepdim=True)
#         output['merged-HR_patch'] = self.auxillary_regressors['merged-HR_patch'](feature_fft)

        
        merged_RR_deep_features = torch.cat([deep_feature['SCG-RR_cosmed'], deep_feature['PPG-RR_cosmed']], axis=1)
        # output['merged-RR_cosmed'] = self.auxillary_regressors['merged-RR_cosmed'](merged_RR_deep_features)

        features_fft_RR, fused_fft_RR, channel_attention_RR, spectral_attention_RR = self.RR_fusion(merged_RR_deep_features)
        features_fft_RR_SCG, _, _, _ = self.RR_fusion(deep_feature['SCG-RR_cosmed'])

        # # compute fused RR deep features (in time domain)
        # # fused_RR_deep_features dim: torch.Size([5, 1, 187])
        # fused_RR_deep_features = torch.sum(channel_attention_RR.expand(list(merged_RR_deep_features.size()))*merged_RR_deep_features, axis=1, keepdim=True)
        output['SCG-RR_cosmed'] = self.auxillary_regressors['SCG-RR_cosmed'](deep_feature['SCG-RR_cosmed'], spectral_attention_RR)
        output['PPG-RR_cosmed'] = self.auxillary_regressors['PPG-RR_cosmed'](deep_feature['PPG-RR_cosmed'], spectral_attention_RR)

        fused_RR_deep_features = torch.sum(channel_attention_RR.expand(list(merged_RR_deep_features.size()))*merged_RR_deep_features, axis=1, keepdim=True)
        output['merged-RR_cosmed'] = self.auxillary_regressors['merged-RR_cosmed'](fused_RR_deep_features)
        # print('merged_RR_deep_features', merged_RR_deep_features.size())
        # print('features_fft_RR', features_fft_RR.size())
        # print('fused_fft_RR', fused_fft_RR.size())
        # print('channel_attention_RR', channel_attention_RR.size())
        # print('spectral_attention_RR', spectral_attention_RR.size())
        # print('fused_RR_deep_features', fused_RR_deep_features.size())
        # print('output[merged-RR_cosmed]', output['merged-RR_cosmed'].size())

        
        
        # band-pass filter
        # features_fft_HR dim: (B,2C,f)
        features_fft_HR = spectral_attention_HR.expand(list(features_fft_HR.size())) * features_fft_HR
        # features_fft_RR dim: (B,2C,f)
        features_fft_RR = spectral_attention_RR.expand(list(features_fft_RR.size())) * features_fft_RR

        # # features_fft_HR dim: (B,2C,f)
        # features_fft_HR = spectral_attention_HR.expand(list(features_fft_HR.size())) * features_fft_HR
        # # features_fft_RR dim: (B,2C,f)
        # features_fft_RR = spectral_attention_RR.expand(list(features_fft_RR.size())) * features_fft_RR

        
        # compute dominant energy for each channel
        # features_fft_HR dim: (B,2C)
        features_fft_HR = torch.mean(features_fft_HR, axis=-1)
        # print('features_fft_HR', features_fft_HR.size())
        # features_fft_RR dim: (B,2C)
        features_fft_RR = torch.mean(features_fft_RR, axis=-1)
        
        # output['merged-HR_patch'] dim: (B,1)
        # features_fft_HR dim: (B,C+1)
        features_fft_HR = torch.cat([features_fft_HR, output['merged-HR_patch']], axis=1)
        # print('features_fft_HR', features_fft_HR.size())
        
        # print('features_fft_RR', features_fft_RR.size())
        
        # output['merged-RR_cosmed'] dim: (B,1)
        # features_fft_RR dim: (B,C+1)
        features_fft_RR = torch.cat([features_fft_RR, output['merged-RR_cosmed']], axis=1)
        # print('features_fft_RR', features_fft_RR.size())
            
#         # map cardiac features to VO2 domain
#         # features_fft_HR dim: (B,2C+2)
#         # merged_HR_deep_features dim: (B,2C)
        # merged_HR_deep_features = self.HR_mapping(features_fft_HR)
        
        
#         # map respiratory features to VO2 domain
#         # features_fft_RR dim: (B,2C+2)
#         # merged_RR_deep_features dim: (B,2C)
        # merged_RR_deep_features = self.RR_mapping(features_fft_RR)
        


        # adaptively fuse cardiac and respiratory features
        # merged_deep_features dim: (B,2C)
        # merged_deep_features = self.CR_fusion(merged_HR_deep_features, merged_RR_deep_features)
        merged_deep_features = torch.cat([features_fft_HR, features_fft_RR], axis=-1)
        # print('merged_deep_features', merged_deep_features.size())

        # sys.exit()
        # output['cardiac-'+self.main_task], concat_feature = self.regressors[self.main_task](merged_HR_deep_features, feature)
        # output['resp-'+self.main_task], concat_feature = self.regressors[self.main_task](merged_RR_deep_features, feature)
        output[self.main_task], concat_feature = self.regressors[self.main_task](merged_deep_features, feature)
        # print('output,  deep_feature, concat_feature', output,  deep_feature, concat_feature)
        
        # sys.exit()
        return output, deep_feature, concat_feature

    
    
class CR_fusion_module(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CR_fusion_module, self).__init__()
        self.SE = SELayer(channel=2) # cardiac and respiratory
        
    def forward(self, HR_deep_features, RR_deep_features):        
        # HR_deep_features and RR_deep_features dim: (B,C)
        # deep_features dim: (B,2,C)
        deep_features = torch.cat([HR_deep_features[:,None,:],RR_deep_features[:,None,:]], axis=1)
        # deep_features = self.SE(deep_features)
        
        # merged_deep_features dim: (B,C)
        deep_features = torch.mean(deep_features, axis=1)
        return deep_features
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.size())
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # print(y.size())
        # print(self.fc(y).size(), b, c)
        y = self.fc(y).view(b, c, 1)
        
        # print(y.size())

        return x * y.expand_as(x)

    
class CardioRespRegression2(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=10):
        super(CardioRespRegression2, self).__init__()
        # self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # concat_dim = 2*training_params['channel_n']+feature_dim
        # concat_dim = 2+feature_dim
        

        # self.SE = SELayer(channel=2) # cardiac and respiratory

        self.vanilla_regressor = vanilla_regression(training_params, num_classes=num_classes, input_dim=input_dim, hidden_dim=100, n_layers=5)
        
        
#         self.bn1 = nn.BatchNorm1d(channel_n)
#         self.relu = nn.ReLU()
#         self.fc1 = nn.Linear(channel_n, channel_n)
        
#         self.bn2 = nn.BatchNorm1d(channel_n)
#         self.fc2 = nn.Linear(channel_n, channel_n)
        
#         self.fft_layer = FFT_layer()
        

#         self.HR_xf_dict = training_params['HR_xf_dict']
#         self.HR_xf_masked = torch.from_numpy(self.HR_xf_dict['xf_masked']).float()
#         # print(self.HR_xf_masked['mask'])
#         self.HR_mask = torch.from_numpy(self.HR_xf_dict['mask'])
#         self.get_attention_HR = attention_module(training_params, self.HR_xf_dict)
        
#         self.RR_xf_dict = training_params['RR_xf_dict']
#         self.RR_xf_masked = torch.from_numpy(self.RR_xf_dict['xf_masked']).float()
#         self.RR_mask = torch.from_numpy(self.RR_xf_dict['mask'])
#         self.get_attention_RR = attention_module(training_params, self.RR_xf_dict)
        
    def forward(self, merged_deep_features, feature):
        # feature = feature.reshape(feature.size(0), -1)

        # out_HR = self.vanilla_regressor(torch.cat((merged_deep_features[:,0,:], feature[:,:3]), 1))
        # out_RR = self.vanilla_regressor(torch.cat((merged_deep_features[:,1,:], feature[:,:3]), 1)

        
        # print('merged_deep_features', merged_deep_features.size())
        # merged_deep_features dim: torch.Size([5, 2, 16])
        # merged_deep_features = self.SE(merged_deep_features)
        # print(merged_deep_features.size(), feature.size())
        # concat_feature = torch.cat((torch.sum(merged_deep_features, axis=1), feature[:,:3]), 1)
        concat_feature = torch.cat([merged_deep_features, feature[:,:3]], axis=-1)
        # print(concat_feature.size())

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
        self.avg_pool = MyAvgPool1dPadSame(kernel_size=2, stride=1)
        
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
        attention = attention > torch.mean(attention, axis=-1, keepdim=True)
        # print(attention.size())
        
        # sys.exit()
        return attention


class CardioRespRegression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=1, feature_dim=0):
        super(CardioRespRegression, self).__init__()
        # self.Maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # concat_dim = 2*training_params['channel_n']+feature_dim
        # concat_dim = 2+feature_dim
        concat_dim = 2 + 3
        
        self.vanilla_regressor = vanilla_regression(training_params, num_classes=num_classes, input_dim=concat_dim, hidden_dim=100, n_layers=5)
        
        self.fft_layer = FFT_layer()
        

        self.HR_xf_dict = training_params['HR_xf_dict']
        self.HR_xf_masked = torch.from_numpy(self.HR_xf_dict['xf_masked']).float()
        # print(self.HR_xf_masked['mask'])
        self.HR_mask = torch.from_numpy(self.HR_xf_dict['mask'])
        self.get_attention_HR = attention_module(training_params, self.HR_xf_dict)
        
        self.RR_xf_dict = training_params['RR_xf_dict']
        self.RR_xf_masked = torch.from_numpy(self.RR_xf_dict['xf_masked']).float()
        self.RR_mask = torch.from_numpy(self.RR_xf_dict['mask'])
        self.get_attention_RR = attention_module(training_params, self.RR_xf_dict)
        
    def forward(self, deep_feature, feature):

        # print(deep_feature['SCG-HR_patch'].size())
        
        
        x_fft = self.fft_layer(deep_feature['SCG-HR_patch'])
        x_fft = x_fft[:,:,self.HR_mask]
        attention = self.get_attention_HR(deep_feature['ECG-HR_patch'])            
        x_fft = x_fft * attention
        x_fft_HR = torch.sum(x_fft, axis=-1)

        x_fft = self.fft_layer(deep_feature['SCG-RR_cosmed'])
        x_fft = x_fft[:,:,self.RR_mask]
        attention = self.get_attention_RR(deep_feature['PPG-RR_cosmed'])            
        x_fft = x_fft * attention
        x_fft_RR = torch.sum(x_fft, axis=-1)
        # sys.exit()
#         if attention_sig is not None:
#             attention = self.get_attention(attention_sig)            
#             # attention = attention > torch.mean(attention, axis=-1, keepdim=True)

#             # TODO: check if x and attention have the same size
#             x_fft = x_fft * attention
            
        # x_fft = torch.sum(x_fft, axis=-1)
        # print(x_fft[:5,:])
        # print('feature.size()', feature.size())
        feature = feature.reshape(feature.size(0), -1)
        # print(feature.size())
        concat_feature = torch.cat((x_fft_HR, x_fft_RR, feature[:,:3]), 1)
        # concat_feature = torch.cat((x_fft_HR, x_fft_RR), 1)
        # print('out', out.size(), out)
        # print('feature', feature.size(), feature)
        # print(concat_feature.size())
        out = self.vanilla_regressor(concat_feature)
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
    
    
    
class SingleLayer_regression(nn.Module):
    def __init__(self, input_dim=10, output_dim=50, p_dropout=0.5):
        super(SingleLayer_regression, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_dim, output_dim)
        # self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        # out = self.relu(self.bn(self.fc(x)))
        # out = self.drop(self.fc(self.relu(self.bn(x))))
        out = self.fc(self.relu(self.bn(x)))
        return out

class vanilla_regression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, hidden_dim=100, n_layers=5):
        super(vanilla_regression, self).__init__()
        
        self.basiclayer_list = nn.ModuleList()
        for i_block in range(n_layers-1):
            self.basiclayer_list.append(SingleLayer_regression(input_dim=input_dim, output_dim=hidden_dim))
            input_dim = hidden_dim
            
        self.basiclayer_list.append(nn.Linear(input_dim, num_classes))

    def forward(self, feature, attention_sig=0):
        
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
        
        
        


# class spectral_attention_module(nn.Module):
#     """
#     This function create a mask in the spectral domain (spectral hard attention)
#     """
#     def __init__(self, training_params, xf_dict=None):
#         super(spectral_attention_module, self).__init__()
#         self.fft_layer = FFT_layer()
        
#         if xf_dict is None:
#             self.xf_dict = training_params['xf_dict']
#         else:
#             self.xf_dict = xf_dict

#         self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
#         self.mask = torch.from_numpy(self.xf_dict['mask'])
        
#         freq_smooth_dur = 5 # bpm
        
#         # print(np.diff(self.xf_masked.data.numpy()))
#         freq_smooth_win = round(freq_smooth_dur / np.mean(np.diff(self.xf_masked.data.numpy())))
#         self.avg_pool = MyAvgPool1dPadSame(kernel_size=freq_smooth_win, stride=1)

#     def forward(self, sig):
#         # print('attention_module')
#         # attention dim: (N_batch, 1, N_spectral)
#         attention = self.fft_layer(sig)
#         attention = attention[:,:,self.mask]
        
#         attention = self.avg_pool(attention)
        
#         # attention dim: (N_batch, 1, N_spectral)
#         attention = attention / torch.sum(attention, axis=-1)[:,None]
        
#         # create the mask
#         attention = attention > torch.mean(attention, axis=-1, keepdim=True)
        
#         # sys.exit()
#         return attention


        
        
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
    def __init__(self, training_params, xf_dict):
        super(DominantFreqRegression, self).__init__()
        
        self.xf_dict = xf_dict
        self.xf_masked = torch.from_numpy(self.xf_dict['xf_masked']).float()
        self.mask = torch.from_numpy(self.xf_dict['mask'])

        self.fft_layer = FFT_layer()

    def forward(self, x, spectral_attention=None):
        # x is in time domain

        # # FFT layer, mask it, reshape it, normalize feature to 1       
        # print(x)
        # print(x.size())
        out = self.fft_layer(x)
        # print(out.size())
        # print(self.mask.size())
        out = out[:,:,self.mask]
        
        # if attention_sig is not None:
        #     attention = self.get_attention(attention_sig)            
        #     # TODO: check if x and attention have the same size
        #     out = out * attention

        # out dim: (N_batch, N_ch, N_feature)
        out = out / torch.sum(out, axis=-1, keepdim=True)
        
        if spectral_attention is not None:
            out = spectral_attention.expand(list(out.size())) * out

        
        
        # TODO: make sure I am computing the expectation correctly
        # xf_masked_torch dim: (1, 1, N_spectral)
        xf_masked_torch = self.xf_masked.to(x.device)[None,None,:]
        
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
    
    

class CRQI_module(nn.Module):
    """get cardiopulmonary quality index without learnable parameters"""
    def __init__(self,):
        super(CRQI_module, self).__init__()

        self.avg_pool = MyAvgPool1dPadSame(kernel_size=2, stride=1)
        self.max_pool = nn.AdaptiveMaxPool1d((1,))
        # self.activation = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x is the fft feature of a modalityt
        # print('x', x.size())
        CRQI = self.avg_pool( x )   
        # print('CRQI', CRQI.size())

        CRQI = self.max_pool(CRQI) / CRQI.sum(dim=-1,keepdim=True)
        return CRQI

        
#         outputs = []
#         for i_ch in range(x.size(1)): # the 1st dim stores the number of channels
            
#             # (N, 2, 58) -> (N, 1, 58) -> (N, 1, 58) -> (N, 1, 58)
#             interm = self.avg_pool( x[:,[i_ch],:] )   
            
#             # normalization: (N, 1, 58) -> (N, 1, 58) -> (N, 1, 1)/(N, 1, 1) -> (N, 1, 1) 
#             interm = self.max_pool(interm) / interm.sum(dim=-1,keepdim=True)

#             interm = interm[:,0,0]

#             outputs.append(interm)

#         outputs = torch.stack(outputs).T
#         weights = self.softmax(outputs)

#         out = torch.sum(x * weights[:,:,None].expand_as(x), dim=1,keepdim=True)

        # return out, weights

    

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
            
        
        
#         self.loss_weights['VO2_cosmed'] = training_params['main_loss_weight']
#         self.loss_weights['cardiac-VO2_cosmed'] = training_params['main_loss_weight']
#         self.loss_weights['resp-VO2_cosmed'] = training_params['main_loss_weight']
        
#         if training_params['auxillary_loss_weight']!=0:
#             # self.loss_weights['ECG-HR_patch'] = training_params['auxillary_loss_weight']
#             # self.loss_weights['SCG-HR_patch'] = training_params['auxillary_loss_weight']
#             # self.loss_weights['PPG-RR_cosmed'] = training_params['auxillary_loss_weight']
#             # self.loss_weights['SCG-RR_cosmed'] = training_params['auxillary_loss_weight']
#             self.loss_weights['merged-HR_patch'] = training_params['auxillary_loss_weight']
#             self.loss_weights['merged-RR_cosmed'] = training_params['auxillary_loss_weight']

        # self.regression_names = ['merged-HR_patch', 'SCG-HR_patch', 'merged-RR_cosmed', 'SCG-RR_cosmed', 'VO2_cosmed']
        
        self.criterions = {}

        for regression_name in self.loss_weights.keys():
            self.criterions[regression_name] = torch.nn.MSELoss()
        #     if self.main_task in regression_name:
        #         self.loss_weights[regression_name] = training_params['loss_weights']['main_task']
        #     else:
        #         N_aux_tasks = len(self.regression_names) - 1
        #         if N_aux_tasks==0:
        #             self.loss_weights[regression_name] = 0
        #         else:
        #             self.loss_weights[regression_name] = training_params['loss_weights']['auxillary_task']/N_aux_tasks

    def forward(self, out, label):
        
        # print('out, out.keys()', out, out.keys())
        # print('self.regression_names', self.regression_names)
        # for out_name in out.keys():
        #     print(self.regression_names.index( out_name ))

        label = {out_name: label[:, [self.regression_names.index( out_name )]] for out_name in out.keys()}
                # label = {out_name: label[:, [out_name.split('-')[-1]]] for out_name in out.keys()}

        # print('label, label.keys()', label, label.keys())
        # print('out, out.keys()', out, out.keys())

        losses = {}
        # for out_name in out.keys():
        for out_name in self.criterions.keys():
            # print(out_name, out[out_name].squeeze(), label[out_name])
            losses[out_name] = self.criterions[ out_name ](out[out_name].squeeze(), label[out_name].to(device=self.device, dtype=torch.float).squeeze())
        # losses = {out_name: self.criterions[ out_name ](out[out_name].squeeze(), label[out_name].to(device=self.device, dtype=torch.float).squeeze()) for out_name in out.keys()}
            # print(out_name, out[out_name].squeeze(), label[out_name])
        
        # print(torch.stack([self.loss_weights[ out_name ] * losses[out_name] for out_name in out.keys()]))
        # print(out.keys())
        # sys.exit()

        losses['total'] = torch.sum(torch.stack([self.loss_weights[ out_name ] * losses[out_name] for out_name in  self.criterions.keys()]))
        
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
    