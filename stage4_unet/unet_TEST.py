#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from tqdm import tqdm
from tqdm import trange
import time

import pandas as pd

import math
from math import sin
from icecream import ic
import json
import argparse

# checklist 1: uncomment matplotlib.use('Agg')
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
import matplotlib.gridspec as gridspec

# plt.style.use('dark_background')

from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

i_seed = 0

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from torchsummary import summary

import PIL

import sys
import sys
sys.path.append('../') # add this line so Data and data are visible in this file
sys.path.append('../../') # add this line so Data and data are visible in this file
sys.path.append('../PatchWand/') # add this line so Data and data are visible in this file

from plotting_tools import *
from preprocessing import *
from setting import *
from surrogate_extraction import *
from dataIO import *
from filters import *
from spectral_module import *
from resp_module import *
from stage4_regression import *
from unet_extension.dataset_util import *
from unet_extension.evaluation_util import *
from unet_extension.models import *
from unet_extension.training_util import *


import datetime
import time

# checklist 2: comment out all magic command
from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:





# In[ ]:





# In[24]:


parser = argparse.ArgumentParser(description='RR_estimate')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--training_params_file', metavar='training_params_file', help='training_params_file',
                    default='training_params_list.json')
# parser.add_argument('--task_ids_train', metavar='task_ids_train', help='task_ids_train',
#                     default='')
# parser.add_argument('--task_ids_val', metavar='task_ids_val', help='task_ids_val',
#                     default='')
parser.add_argument('--TF_type', metavar='TF_type', help='TF_type',
                    default='target')
parser.add_argument('--variant', metavar='variant', help='variant',
                    default='baseline')
parser.add_argument('--grad_clip', metavar='grad_clip', help='grad_clip',
                    default=False)
parser.add_argument('--input_names', metavar='input_names', help='input_names',
                    default='ECG_SR')
# parser.add_argument('--model_name', metavar='model_name', help='model_name',
#                     default='UNet')
# parser.add_argument('--training_scheme', metavar='training_scheme', help='training_scheme',
#                     default='LOSO')


# checklist 3: comment first line, uncomment second line

# args = parser.parse_args(['--input_folder', '../../../covid/results/stage3/win60_overlap95_seq20_ECG_AMpt+ECG_AMr+ECG_AMbi+ECG_SR_norm_script16/', 
# args = parser.parse_args(['--input_folder', '../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/', 
# args = parser.parse_args(['--input_folder', '../../data/stage3-1_windowing/GT_dataset/win60_overlap95_seq20_norm/', 
# #                           '--output_folder', '../../../covid/results/stage4/unet_test/win60_overlap95_seq20_ECGAMnorm+SCGBWnorm/',
#                           '--output_folder', '../../data/stage4_UNet_test/',
# #                           '--task_ids_train', '0,1,2,3,4,5',
# #                           '--task_ids_val', '0,1,2,3,4,5,6,7,9,10',
#                           '--variant', 'AT_block',
# #                           '--variant', 'baseline',
# #                           '--variant', 'Late_UNet',
# #                           '--variant', 'Attention_UNet',
#                           '--TF_type', 'prepare',
#                           '--grad_clip', 'true',
#                           '--input_names', 'ECG_SR+ECG_SR',
# #                           '--input_names', 'PPG',
#                           '--training_params_file', 'training_params_baseline_test.json',])
#                           '--model_name', 'UNet'])
                          
args = parser.parse_args()
print(args)


# In[25]:


# TODO

## add late fusion
## SE_block fusion
## 
## 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # TF_type
# # source data only
# # 1. source
# # include target data
# # 2. FT_top # Fine-tune top layer
# # 3. FT_top2 # Fine-tune top 2 layers
# # 4. FT_all # Fine-tune all layers
# # 5. target # Train from scratch
# # 6. prepare # train a model use all data in `domain` and `input_names`

# In[ ]:





# In[26]:


inputdir = args.input_folder
outputdir = args.output_folder

# if len(args.task_ids_train)==0:
#     task_ids_train = None
# else:
#     task_ids_train = [int(item) for item in args.task_ids_train.split(',')]

# if len(args.task_ids_val)==0:
#     task_ids_val = None
# else:
#     task_ids_val = [int(item) for item in args.task_ids_val.split(',')]

TF_type = args.TF_type
variant = args.variant
grad_clip = args.grad_clip

input_names = args.input_names
input_names = input_names.split('+')

if grad_clip=='true':
    grad_clip = True
else:
    grad_clip = False

domain = inputdir.split('/')[-3]
training_params_file = args.training_params_file

# modality = training_params_file.split('_')[-1].split('.')[0]
modality = '+'.join(input_names).replace('_', '-')

# # outputdir += '{}/{}/{}/{}/'.format(domain, modality, inputdir.split('/')[-2], TF_type)
    
outputdir += '{}/{}/{}/{}/'.format(domain, modality, variant, TF_type)

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
print('will export data to', outputdir)


# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


# training_scheme


# In[28]:


torch.cuda.is_available()


# In[29]:


def get_pretrain_dir(training_params):

    if training_params['domain']=='CDC_dataset':
        source_domain = 'GT_dataset'
    else:
        source_domain = 'CDC_dataset'
        
    variant = training_params['variant']

    # by default, use this dir
    # TODO: check if model channel dim is the same as data channel dim
    pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECG-SR+ECG-SR/{}/prepare/'.format(source_domain, variant)

    input_N_channel = len(training_params['input_names'])
    if input_N_channel==1:
        pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECG/{}/prepare/'.format(source_domain, variant)
#       pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/SCG/win60_overlap95_seq20_norm/prepare/'.format(source_domain)

    elif input_N_channel==2:
        pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECG-SR+ECG-SR/{}/prepare/'.format(source_domain, variant)
#         first_list = training_params['input_names']
#         sec_list = ['ECG_SR', 'SCG_AMpt']

#         # use ECGSCG if they are also 'ECG_SR' and 'SCG_AMpt'
#         if all(map(lambda x, y: x == y, first_list, sec_list)):
# #                     pretrain_dir = os.path.expanduser('~')+'/Estimation_RR/covid/results/stage4/unet_test/win60_overlap95_seq20_ECG_SR+SCG_AMpt_norm_script18/'
#             pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECGSCG/win60_overlap95_seq20_norm/prepare/'.format(source_domain)

#         else:
#             pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECGdouble/win60_overlap95_seq20_norm/prepare/'.format(source_domain)
    elif input_N_channel==3:
        pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECGtriple/{}/prepare/'.format(source_domain, variant)

    elif input_N_channel==4:
        pretrain_dir = os.path.expanduser('~')+'/Estimation_EE/data/stage4_UNet/{}/ECGquadruple/{}/prepare/'.format(source_domain, variant)


    return pretrain_dir


# In[30]:


# # training_params['model_name'] = 'variant'
# xarr = np.array([1, 1, 2, 2.1, 3, 4, 5])
# y = np.array([1,2])
# mask_task = np.in1d(xarr, y)
# # Out[25]: array([ True,  True, False, False, False], dtype=bool)

# domain


# In[ ]:



with open(training_params_file) as json_file:
    training_params_list = json.load(json_file)

for training_params in training_params_list:
    
#     training_params['training_scheme'] = training_scheme
    # include device in training_params
    training_params['TF_type'] = TF_type
    training_params['domain'] = domain
    training_params['variant'] = variant
    training_params['outputdir'] = outputdir
    training_params['grad_clip'] = grad_clip
    training_params['input_names'] = input_names
    
    
    if domain=='CDC_dataset':
        if len(training_params['task_ids_train'])==0:
            training_params['task_ids_train'] = [0,1,2,3,4,5]
        else:
            training_params['task_ids_train'] = [int(item) for item in training_params['task_ids_train'].split(',')]

        if len(training_params['task_ids_val'])==0:
            training_params['task_ids_val'] = [0,1,2,3,4,5,6,10]
        else:
            training_params['task_ids_val'] = [int(item) for item in training_params['task_ids_val'].split(',')]
    elif domain=='GT_dataset':
        training_params['task_ids_train'] = [101]
        training_params['task_ids_val'] = [101]


    
    training_params['select_channel'] = True
    
#     if 'variant' not in training_params:
#         training_params['variant'] = 'baseline'

    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
#     device = torch.device('cpu')
    training_params['device'] = device

    stage3_dict = data_loader('stage3_dict', inputdir).item()
    freq = stage3_dict['freq']
    
    training_params['xf_masked'] =  torch.from_numpy(freq)*60
    
    # include freq_dict in training_params
#     freq = data_loader('freq', inputdir)
    freq_dict = dict(zip(np.arange(freq.shape[0]), freq))
    training_params['freq_dict'] = freq_dict
    training_params['subject_ids'] = stage3_dict['subject_ids']    
    
#     if 'ordered_subject_ids' in training_params:
#         training_params['subject_ids'] = np.asarray(training_params['ordered_subject_ids'])
#       "ordered_subject_ids": [116, 118, 113, 105, 212, 117, 114, 110, 101, 103, 104, 106, 107, 108, 111, 115, 119, 120, 121],
# 16, 10, 13, 9, 11,
    if domain=='GT_dataset':
        training_params['subject_ids'] = np.asarray([11, 10, 9,1, 18, 23, 13, 16, 5,  3, 4, 6, 7, 12, 15, 17, 22])
    
    training_params['surrogate_names'] = stage3_dict['surrogate_names']
    
    
    training_params['meta_names'] = stage3_dict['meta_names']
    
    training_params['i_HR'] = training_params['meta_names'].index('HR_cosmed')
    training_params['i_VT'] = training_params['meta_names'].index('VT_cosmed')
    training_params['i_RR'] = training_params['meta_names'].index('RR_cosmed')
    # include data_dimensions in training_params
    
    
    if domain=='GT_dataset':
        task_id = [101]
    elif domain=='CDC_dataset':
#         task_id = [0, 1, 2, 3, 4, 5]
#         task_id = [0, 1, 2, 5]
#         task_id = [0, 1, 2, 3, 5, 6]
        task_id = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]


    if domain=='GT_dataset':
        training_params['CV_config'] = {
            'subject_id': 104,
            'task_id': task_id,
            'reject_subject_id': []
        }
    elif domain=='CDC_dataset':
        training_params['CV_config'] = {
            'subject_id': 104,
            'task_id': task_id,
            'reject_subject_id': [101, 102, 103]
        }
    
    # reject bad subject
    training_params['subject_ids'] = training_params['subject_ids'][~np.isin(training_params['subject_ids'], training_params['CV_config']['reject_subject_id'])]
#     sys.exit()
    
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    data_dimensions = dataloaders['train'].dataset.__getitem__(0)[0].size()
    print('data_dimensions:', data_dimensions)
    training_params['data_dimensions'] = list(data_dimensions)
    del dataloaders

    
    training_params['label_range'] = 'label+estimated'


    

    pretrain_dir =  get_pretrain_dir(training_params)
    training_params['pretrain_dir'] = pretrain_dir
    
    
    if 'regressor' not in training_params:
#     training_params['regressor'] = 'DominantFreq_regressor'
        training_params['regressor'] = None

    
#     training_params['model_type'] = model_dict[training_params['model_name']]
    training_params['model_type'] = model_dict[training_params['variant']]


# In[ ]:





# In[12]:


stage3_dict['surrogate_names']


# In[ ]:





# In[ ]:





# In[13]:


print(torch.version.cuda)


# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


# # outputdir += '{}/{}/{}/{}/'.format(domain, modality, inputdir.split('/')[-2], TF_type)
# # training_params

# df_ppg_hr = pd.read_feather('../../data/stage3/cardiac/df_ppg_hr.feather')
# df_ppg_hr[(df_ppg_hr['subject_id']=='106') & (df_ppg_hr['task_name']=='6MWT')]


# In[15]:


# outputdir_df_ppg_selected =  '../../data/stage3/cardiac/df_ppg_selected.feather'
# df_ppg_selected = pd.read_feather(outputdir_df_ppg_selected)
# df_ppg_selected[df_ppg_selected['task_name']=='6MWT']


# In[ ]:





# In[ ]:





# In[ ]:





# # retrieve signal and labels

# In[ ]:





# In[ ]:





# In[16]:


debug_sAttentLayer = False

if debug_sAttentLayer:

    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    data =dataloaders['train'].dataset.data
    meta =dataloaders['train'].dataset.meta
    
#     mask = meta[:,0]==4
#     data = data[mask,:,:,:]
#     meta = meta[mask,:]

    i_sample = -15
    i_sample = -1

    fig,axes=plt.subplots(3,1,figsize=(10,8))
    axes[0].imshow(data[i_sample,0,:,:].T)
    axes[1].imshow(data[i_sample,1,:,:].T)
    axes[2].imshow(data[i_sample,:,:,:].mean(axis=0).T)
    plt.show()

    atten_block = sAttentLayer2(training_params, N_freq=58, channel=1, reduction=2).to(training_params['device']).float()
    data = torch.from_numpy(data).to(training_params['device']).float()

    x = {}
    
    x = data

    
#     x['ECG'] = torch.concat([data[:,[0],:,:], data[:,[0],:,:]],dim=1)
#     x['SCG'] = torch.concat([data[:,[1],:,:], data[:,[1],:,:]],dim=1)

#     out, x, weights, outputs = atten_block(x)
    out, weights = atten_block(x)
    print(weights[i_sample, :])

    fig,ax=plt.subplots(figsize=(4,2))

    ax.plot(weights.detach().cpu().numpy()[:,0])
    # plt.plot(meta[:,0]/200+0.5)
    plt.show()


    fig,axes=plt.subplots(3,1,figsize=(10,8))
    axes[0].imshow(x[i_sample,0,:,:].detach().cpu().numpy().T )
    axes[1].imshow(x[i_sample,1,:,:].detach().cpu().numpy().T )

#     out_sum = (out['ECG'][i_sample,1,:,:].detach().cpu().numpy() + out['SCG'][i_sample,1,:,:].detach().cpu().numpy() ) / 2
#     axes[2].imshow(out_sum)
    axes[2].imshow(out[i_sample,0,:,:].detach().cpu().numpy().T)

    plt.show()


# In[ ]:





# In[17]:


# m(aaa).size()


# In[ ]:





# In[18]:


# KERNEL_SIZE = 3
# groups = 1
# channel = 1
# N_freq = 58

# conv = nn.Sequential(
#     nn.Conv2d(channel, channel, kernel_size=KERNEL_SIZE, stride=1, padding='same', bias=True, groups=groups),
#     nn.BatchNorm2d(channel),
#     nn.ReLU(inplace=True),
#     nn.AvgPool2d(kernel_size=KERNEL_SIZE, stride=(KERNEL_SIZE,1), padding=(0,1)),
#     nn.Conv2d(channel, channel, kernel_size=KERNEL_SIZE, stride=1, padding='same', bias=True, groups=groups),
#     nn.BatchNorm2d(channel),
#     nn.ReLU(inplace=True),
#     nn.AvgPool2d(kernel_size=KERNEL_SIZE, stride=(KERNEL_SIZE,1), padding=(0,1)),
# ).to(training_params['device']).float()

# avg_pool = nn.AdaptiveAvgPool2d((1, N_freq))
# max_pool = nn.AdaptiveMaxPool2d((1, 1))
# softmax = nn.Softmax(dim=-1)
# fc = nn.Linear(N_freq,1).to(training_params['device']).float()


# aaa = avg_pool(conv(x[:,[0],:,:]))
# bbb = avg_pool(conv(x[:,[1],:,:]))

# aaa = aaa / aaa.sum(dim=-1,keepdim=True)
# bbb = bbb / bbb.sum(dim=-1,keepdim=True)

# aaa_logit = fc(aaa.squeeze())
# bbb_logit = fc(bbb.squeeze())

# # aaa_max = max_pool(aaa).squeeze()
# # bbb_max = max_pool(bbb).squeeze()



# aaa.size()

# aaa_logit.size()

# weights = torch.stack([aaa_logit, bbb_logit]).T
# weights = softmax(weights)

# weights.size()

# weights

# self.softmax = nn.Softmax(dim=-1)


# fig,axes=plt.subplots(2,1,figsize=(10,8))
# # axes[0].imshow(aaa.detach().cpu().numpy()[i_sample,0,:,:].T)
# # axes[1].imshow(bbb.detach().cpu().numpy()[i_sample,0,:,:].T)
# axes[0].plot(aaa.detach().cpu().numpy()[i_sample,0,:,:].squeeze())
# axes[0].plot(bbb.detach().cpu().numpy()[i_sample,0,:,:].squeeze())

# print(aaa_max.squeeze()[i_sample], bbb_max.squeeze()[i_sample])

# #     axes[2].imshow(data[i_sample,:,:,:].mean(axis=0).T)
# plt.show()


# In[19]:


training_params['model_type']


# In[ ]:





# In[20]:


def test_model(training_params):
    
    model_def = training_params['model_type']
    device = training_params['device']
#     model= model_def(in_ch=1, out_ch=2, n1=training_params['N_channels']).to(device).float()
#     if training_params['model_name']=='UNet':
#         model= model_def(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
#     elif training_params['model_name']=='Late_UNet':
#         model= model_def(training_params=training_params)
    model= model_def(training_params=training_params)

    data_dimensions = training_params['data_dimensions'].copy()
    
    if training_params['flipper']:
        data_dimensions[-1] = data_dimensions[-1] * 2 

    print(model)
    
#     summary(model, input_size=tuple(data_dimensions), batch_size=16, device='cpu')
    
test_model(training_params)


# In[ ]:





# # unit test: check if model saved and loaded are the same

# In[21]:


import unittest

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
        return True
    else:
        return False

class TestModelFunctions(unittest.TestCase):
    """Tests for a pileline output"""
    
    @classmethod
    def setUpClass(cls):
        print('outputdir:', outputdir)
        
    def test_save_load(self):
        
        model_def = training_params['model_type']

#         if training_params['model_name']=='UNet':
#             model1 = model_def(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
#         elif training_params['model_name']=='Late_UNet':
#             model1 = model_def(training_params=training_params)
        model1 = model_def(training_params=training_params)

#         model1= training_params['model_type'](in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)

        torch.save(model1.state_dict(), outputdir+'model_weights.pth')


#         if training_params['model_name']=='UNet':
#             model2 = model_def(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
#         elif training_params['model_name']=='Late_UNet':
#             model2 = model_def(training_params=training_params)
#         model2= training_params['model_type'](in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
        model2 = model_def(training_params=training_params)

        model2.load_state_dict(torch.load(outputdir+'model_weights.pth'))

        self.assertTrue(compare_models(model1, model2))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # get outputdir

# In[22]:


def get_outputdirs(training_params):

#     outputdir = training_params['outputdir']
#     sweep_folder = get_sweep_folder(training_params)
#     outputdir_sweep = outputdir+'{}/'.format(sweep_folder)

    outputdir_sweep = outputdir
    
    outputdir_numeric = outputdir_sweep + 'numeric_results/'
    if outputdir_numeric is not None:
        if not os.path.exists(outputdir_numeric):
            os.makedirs(outputdir_numeric)

    outputdir_modelout = outputdir_sweep + 'model_output/'
    if outputdir_modelout is not None:
        if not os.path.exists(outputdir_modelout):
            os.makedirs(outputdir_modelout)

    outputdir_activation = outputdir_sweep + 'activation_layers/'
    if outputdir_activation is not None:
        if not os.path.exists(outputdir_activation):
            os.makedirs(outputdir_activation)
            
    outputdir_featuremap = outputdir_sweep + 'feature_maps/'
    if outputdir_featuremap is not None:
        if not os.path.exists(outputdir_featuremap):
            os.makedirs(outputdir_featuremap)

#     outputdir_feature = outputdir_sweep + 'feature_visualization/'
#     if outputdir_feature is not None:
#         if not os.path.exists(outputdir_feature):
#             os.makedirs(outputdir_feature)

    training_params['outputdir_sweep'] = outputdir_sweep
    training_params['outputdir_numeric'] = outputdir_numeric
    training_params['outputdir_modelout'] = outputdir_modelout
    training_params['outputdir_activation'] = outputdir_activation
    training_params['outputdir_featuremap'] = outputdir_featuremap

#     training_params['outputdir_feature'] = outputdir_feature

    return training_params


# In[23]:


task = 'RR_cosmed'

df_performance_train = {}
df_performance_val = {}
df_performance_val_input = {}
df_performance_val_exp = {}

df_outputlabel_train = {}
df_outputlabel_val = {}
df_outputlabel_val_input = {}

df_performance_train[task] = pd.DataFrame()
df_performance_val[task] = pd.DataFrame()
df_performance_val_exp[task] = pd.DataFrame()
df_performance_val_input[task] = pd.DataFrame()

df_outputlabel_train[task] = pd.DataFrame()
df_outputlabel_val[task] = pd.DataFrame()
df_outputlabel_val_input[task] = pd.DataFrame()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # TODO: need to change pretrain dir so GT_dataset load model from CDC_dataset dir, and CDC_dataset load model from GT_dataset dir

# In[ ]:





# In[ ]:





# In[ ]:



print('\n================================ TRAIN / TEST ================================')
print('begining soon...')
start_time = time.time()

# use_pretrained = True

# Determining which pretrain model to use is a bit involved. Please read get_pretrain_dir carefully
pretrain_dir = training_params['pretrain_dir']
print('\tusing pretrain_dir:', pretrain_dir)

device = training_params['device']

training_params = get_outputdirs(training_params) # could be tricky since it changes several keys
model_def = training_params['model_type']



vis_act = False


vis_filt = True
vis_feat = True

# for i_CV, subject_id in enumerate(ordered_subject_ids):
for subject_id in training_params['subject_ids']:
    
#     if subject_id!=22:
#         continue

    # stop wasting your time on these reject_subject_id!
    if subject_id in training_params['CV_config']['reject_subject_id']:
        continue

    # subject_id = -1 implies using all of the data for training
#     if (training_params['TF_type']=='pretrain') or (training_params['TF_type']=='prepare'):
    if (training_params['TF_type']=='prepare'):
        subject_id = -1

    # set CV_config-subject_id to subject_id so get_loaders can get the right loader for ya
    training_params['CV_config']['subject_id'] = subject_id
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    
    # LOSO obviously
    print('[sub {}] LOSO'.format( int(subject_id) ) )
    
    # just timing it, may remove it later
    get_loaders_time = time.time()


    # trained with the source (GT dataset), test on the target (CDC dataset)
    # need to import the pretrained model if 'source' or 'FT_' are in training_params['TF_type']
    if ('source' in training_params['TF_type']) or ('FT_' in training_params['TF_type']) or ('pretrain' in training_params['TF_type']):
        print('\tload a model trained using the GT dataset (dont initialize a model from scratch)')
        total_loss_train, total_loss_val = 0, 0

        # load the training_params for the pretrain model
        # TODO: write some tests to make sure it is parsing the right model
        # if pretrain model and input # don't match, maybe replace the first conv layer?
        training_params_pretrain = data_loader('training_params', pretrain_dir).item()
        pretrain_channel = len(training_params_pretrain['input_names'])
        
        # get the pretrained model
#         model= U_Net(in_ch=pretrain_channel, out_ch=2, n1 = training_params['N_channels'], training_params=training_params)        
        model= model_def(training_params=training_params)        
        model.load_state_dict(torch.load(pretrain_dir+'model_weights.pth'))
        
        # TBD: make sure pretrain model has the same channel dimension as the data channel #
        # for if pretrain model and input # don't match, replace the first conv layer
#         if pretrain_channel!=training_params['data_dimensions'][0]:
#             print('\tpretrain model has different input dimension compared to data. Replace the first conv layer.')
#             model_template = U_Net(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
#             model.Conv1.conv[0] = model_template.Conv1.conv[0]
#             del model_template

        # FT_top and FT_top2 may not be used anymore since it doesn't fit the research objective (6/29)
        # for if FT_top, replace the first layer
        if 'FT_top' in training_params['TF_type']:
            print('\tfine-tune the top layer')
#             model_template = U_Net(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
            model_template= model_def(training_params=training_params)
            model.Conv1.conv[0] = model_template.Conv1.conv[0]
            del model_template

        # for if FT_top2, replace the first two layers
        elif 'FT_top2' in training_params['TF_type']:
            print('\tfine-tune the top 2 layers')
#             model_template = U_Net(in_ch=training_params['data_dimensions'][0], out_ch=2, n1=training_params['N_channels'], training_params=training_params)
            model_template= model_def(training_params=training_params)
            model.Conv1.conv[0] = model_template.Conv1.conv[0]
            model.Conv1.conv[3] = model_template.Conv1.conv[3]
            del model_template

    # if training_params['TF_type']=='target' or 'pretrain', will train from scratch
    else:
        print('\tinitialize a new model, train from scratch')
#         model= U_Net(in_ch=training_params['data_dimensions'][0], out_ch=2, n1 = training_params['N_channels'], training_params=training_params)
        model= model_def(training_params=training_params)


    
    model= model.to(device).float()

    # if training_params['TF_type']=='source', will not train (simply feed data into model trained using the GT dataset)
    if training_params['TF_type']!='source':
        print('\ttraining the model...')
        model, total_loss_train, total_loss_val = train_model(model, dataloaders, training_params)
    else:
        print('\tdont train the model...')

    
#     sys.exit()

    train_model_time = time.time()


    plot_loss(total_loss_train, total_loss_val, outputdir=training_params['outputdir_sweep']+'TRAIN-TEST/')

    # (~0.3sec)
    print('\tgetting performance for TRAIN and TEST...')
    performance_dict_TRAIN, model_out_dict_TRAIN = get_performance(model, dataloaders['train_eval'], training_params)
    performance_dict_TEST, model_out_dict_TEST = get_performance(model, dataloaders['val'], training_params)

    
#     # model out is based on interpolated labels
#     plot_unet_results(model_out_dict_TRAIN, training_params, title_str='TRAIN', outputdir=outputdir+'TRAIN-TEST/')
    plot_unet_results(model_out_dict_TEST, training_params, title_str='TEST', outputdir=training_params['outputdir_sweep']+'TRAIN-TEST/')

    #             print( CV_dict['performance_dict_val']['out_dict'].keys(), CV_dict['performance_dict_val']['label_dict'].keys(), task)
    label_est_val = model_out_dict_TEST['RR_model'].squeeze()*60
#     label_expectation_val = model_out_dict_TEST['RR_expectation'].squeeze()
    label_val = model_out_dict_TEST['RR_label'].squeeze()*60
    label_input_val = model_out_dict_TEST['RR_input_spectral'].squeeze()*60
    
    RQI_fft_val = get_RQI_fft(model_out_dict_TEST['out_concat'][:,0,:])
    RQI_kurtosis_val = get_RQI_kurtosis(model_out_dict_TEST['out_concat'][:,0,:])

    # WIP
    VT_val = model_out_dict_TEST['meta'][:, training_params['i_VT']]
    HR_val = model_out_dict_TEST['meta'][:, training_params['i_HR']]

    label_est_train = model_out_dict_TRAIN['RR_model'].squeeze()*60
#     label_expectation_train = model_out_dict_TRAIN['RR_expectation'].squeeze()
    label_train = model_out_dict_TRAIN['RR_label'].squeeze()*60
    
    RQI_fft_train = get_RQI_fft(model_out_dict_TRAIN['out_concat'][:,0,:])
    RQI_kurtosis_train = get_RQI_kurtosis(model_out_dict_TRAIN['out_concat'][:,0,:])
    
    ts_train = model_out_dict_TRAIN['ts']
    ts_val = model_out_dict_TEST['ts']

    # WIP
    VT_train = model_out_dict_TRAIN['meta'][:, training_params['i_VT']]
    HR_train = model_out_dict_TRAIN['meta'][:, training_params['i_HR']]

#     kurtosis_train = get_kurtosis(model_out_dict_TRAIN['out_concat'][:,0,:].T)

#     label_train = model_out_dict_TRAIN['RR_input_spectral'].squeeze()*60

    # get performance df for training and testing dataset
    df_performance_train[task] = df_performance_train[task].append( get_df_performance(label_train, label_est_train, subject_id, task), ignore_index=True )
    df_performance_train[task].to_csv(training_params['outputdir_numeric'] + 'df_performance_train_{}.csv'.format(task), index=False)
    

    df_outputlabel_train[task] = df_outputlabel_train[task].append(
        pd.DataFrame( {
        'label_est': label_est_train,
        'label': label_train,
        'task': [task]*label_train.shape[0],
        'CV': model_out_dict_TRAIN['meta'][:,0],
        'activity': model_out_dict_TRAIN['meta'][:,1],
        'RQI_fft': RQI_fft_train,
        'RQI_kurtosis': RQI_kurtosis_train,

        'VT_train': VT_train,
        'HR_train': HR_train,
            
        'ts_train': ts_train,
        }), ignore_index=True )

    df_outputlabel_train[task].to_csv(training_params['outputdir_numeric'] + 'df_outputlabel_train_{}.csv'.format(task), index=False)


    # RR estimated from model for val dataset
    df_performance_val[task] = df_performance_val[task].append( get_df_performance(label_val, label_est_val, subject_id, task), ignore_index=True )
    df_performance_val[task].to_csv(training_params['outputdir_numeric'] + 'df_performance_val_{}.csv'.format(task), index=False)
    # RR expectation from model for val dataset
#     df_performance_val_exp[task] = df_performance_val_exp[task].append( get_df_performance(label_val, label_expectation_val, subject_id, task), ignore_index=True )
#     df_performance_val_exp[task].to_csv(training_params['outputdir_numeric'] + 'df_performance_val_exp_{}.csv'.format(task), index=False)

    df_outputlabel_val[task] = df_outputlabel_val[task].append(
        pd.DataFrame( {
        'label_est': label_est_val,
#         'label_expectation_val': label_expectation_val,
        'label': label_val,
        'task': [task]*label_val.shape[0],
        'CV': model_out_dict_TEST['meta'][:,0],
        'activity': model_out_dict_TEST['meta'][:,1],
        'RQI_fft': RQI_fft_val,
        'RQI_kurtosis': RQI_kurtosis_val,
            
        'VT_val': VT_val,
        'HR_val': HR_val,
            
        'ts_val': ts_val,
        }), ignore_index=True )

    df_outputlabel_val[task].to_csv(training_params['outputdir_numeric'] + 'df_outputlabel_val_{}.csv'.format(task), index=False)


#     plot_regression(df_outputlabel_val[task], df_performance_val[task], task, training_params, fig_name='regression_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'] )
    plot_regression(df_outputlabel_val[task], task, training_params, fig_name='regression_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'] )


    
    # RR estimated from input for val dataset
    df_performance_val_input[task] = df_performance_val_input[task].append( get_df_performance(label_val, label_input_val, subject_id, task), ignore_index=True )
    df_performance_val_input[task].to_csv(training_params['outputdir_numeric'] + 'df_performance_val_input_{}.csv'.format(task), index=False)

    df_outputlabel_val_input[task] = df_outputlabel_val_input[task].append(
        pd.DataFrame( {
        'label_est': label_input_val,
        'label': label_val,
        'task': [task]*label_val.shape[0],
        'CV': model_out_dict_TEST['meta'][:,0],
        'activity': model_out_dict_TEST['meta'][:,1],
        'RQI_fft': RQI_fft_val,
        'RQI_kurtosis': RQI_kurtosis_val,
            
        'VT_val': VT_val,
        'HR_val': HR_val,
        
        'ts_val': ts_val
        }), ignore_index=True )

    df_outputlabel_val_input[task].to_csv(training_params['outputdir_numeric'] + 'df_outputlabel_val_input_{}.csv'.format(task), index=False)

    plot_regression(df_outputlabel_val_input[task], task, training_params, fig_name='regression_val_input_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'] )
    plot_regression(df_outputlabel_val_input[task], task, training_params, fig_name='regression_val_input_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'] )

    #             plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

    

#     sys.exit()

    print('\tvisualizing results...')
    # TODO: implement this
    # plot the input
    # plot ouput after each maxpool
    # plot attention map (the last y in SE block)
    if vis_act:
        check_attention(model, dataloaders, training_params, mode='best', fig_name = 'IO_weights_{}'.format(int(subject_id)), outputdir=training_params['outputdir_activation']+'best/', show_plot=False)
        check_attention(model, dataloaders, training_params, mode='worst', fig_name = 'IO_weights_{}'.format(int(subject_id)), outputdir=training_params['outputdir_activation']+'worst/', show_plot=False)
        check_attention(model, dataloaders, training_params, mode='random', fig_name = 'IO_weights_{}'.format(int(subject_id)), outputdir=training_params['outputdir_activation']+'random/', show_plot=False)

    if vis_filt:
        
#         if training_params['model_name'] == 'UNet':
        if training_params['variant'] == 'baseline':
            input_name = None
            check_filters(model, training_params, input_choice=input_name, outputdir=training_params['outputdir_featuremap'])

        elif training_params['model_name'] == 'Late_UNet':
            for input_name in training_params['input_names']:
                check_filters(model, training_params, input_choice=input_name, outputdir=training_params['outputdir_featuremap'])

    if vis_feat:
#         if training_params['model_name'] == 'UNet':
        if training_params['variant'] == 'baseline':
            input_name = None
            check_featuremap(model, dataloaders, training_params, mode='best', input_choice=input_name, fig_name = 'featuremap_{}_{}'.format(input_name, int(subject_id)), outputdir=training_params['outputdir_featuremap']+'best/', show_plot=False)
            check_featuremap(model, dataloaders, training_params, mode='worst', input_choice=input_name, fig_name = 'featuremap_{}_{}'.format(input_name, int(subject_id)), outputdir=training_params['outputdir_featuremap']+'worst/', show_plot=False)
            check_featuremap(model, dataloaders, training_params, mode='random', input_choice=input_name, fig_name = 'featuremap_{}_{}'.format(input_name, int(subject_id)), outputdir=training_params['outputdir_featuremap']+'random/', show_plot=False)

            if training_params['variant']=='AT_block':
                check_weights(model, dataloaders, training_params, fig_name = 'weights_{}'.format(int(subject_id)), outputdir=training_params['outputdir_featuremap'], show_plot=False)

#         elif training_params['model_name'] == 'Late_UNet':
        elif training_params['variant'] == 'Late_UNet':
            for input_name in training_params['input_names']:
                check_featuremap(model, dataloaders, training_params, mode='best', input_choice=input_name, fig_name = 'featuremap_{}_{}'.format(input_name, int(subject_id)), outputdir=training_params['outputdir_featuremap']+'best/', show_plot=False)
                check_featuremap(model, dataloaders, training_params, mode='worst', input_choice=input_name, fig_name = 'featuremap_{}_{}'.format(input_name, int(subject_id)), outputdir=training_params['outputdir_featuremap']+'worst/', show_plot=False)


    
    # TODO: implement this
    # plot the conv folters
#     plot_filters(model, fig_name = 'filters_{}'.format(subject_id), outputdir=training_params['outputdir_activation']+'filters/', show_plot=False)

    torch.save(model.state_dict(), training_params['outputdir_sweep'] + 'model_weights.pth')

    del model
    torch.cuda.empty_cache()
    

#     if (training_params['TF_type']=='pretrain') or (training_params['TF_type']=='prepare'):
#         break
    if (training_params['TF_type']=='prepare'):
        break

    
# def plot_regression_all_agg(df_outputlabel, df_performance, training_params, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):

# plot_regression_all_agg(df_outputlabel_train[task], df_performance_train[task], training_params, fig_name='LinearR_agg_train_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
plot_regression_all_agg(df_outputlabel_train[task], training_params, fig_name='LinearR_agg_train_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])

# plot_regression_all_agg(df_outputlabel_val[task], df_performance_val[task], training_params, fig_name='LinearR_agg_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
plot_regression_all_agg(df_outputlabel_val[task], training_params, fig_name='LinearR_agg_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])

plot_output(df_outputlabel_val[task], task, fig_name = 'outputINtime_val_{}'.format(task),  show_plot=False, outputdir=training_params['outputdir_modelout'])

    
print('\nDONE!')


# In[ ]:


# model_out_dict_TRAIN['ts']


# In[ ]:





# In[ ]:


data_saver(training_params, 'training_params', outputdir)

sys.exit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# def check_featuremap(model, dataloaders, training_params, mode='random', fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
#     # mode='worst'
#     data, out, label, RR_model, RR_label, activation = get_data_activation(model, dataloaders, training_params, verbose=True)

#     b, c, h, w = data.shape    
#     error_abs = np.abs(RR_model - RR_label).squeeze()

#     if mode=='worst':
#         i_sample = np.argmax(error_abs)
#     if mode=='best':
#         i_sample = np.argmin(error_abs)
#     if mode=='random':
#         # check one sample only
#         i_sample = np.random.randint(b)

#     fig = plt.figure(figsize=(15,30), constrained_layout=True, dpi=100)
#     gs = gridspec.GridSpec(16, 8)

#     ts = np.arange(h)*3 # 1Hz
#     RR_range = label_range_dict['RR']

#     extent = [ts[0], ts[-1], RR_range[0], RR_range[-1]]

#     # 1. plot the input
#     vmin = data[i_sample,:,:,:].min()
#     vmax = data[i_sample,:,:,:].max()
#     for i_input, input_name in enumerate(training_params['input_names']):
#         ax = fig.add_subplot(gs[i_input, 0])
#         ax.imshow(data[i_sample,i_input,:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title(input_name)

#     #     'Conv1', 'Conv2', 'Conv3', 'Up_conv3', 'Up_conv2'
#     # 2. plot the 1st average pooling output in SE block
#     key = 'Conv1'
#     vmin = activation[key][i_sample,:,:,:].min()
#     vmax = activation[key][i_sample,:,:,:].max()
#     for i_ch in range(activation[key].shape[1]):
#         ax = fig.add_subplot(gs[i_ch, 1])
#         ax.imshow(activation[key][i_sample,i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('i_ch: {}\n{}'.format(i_ch, key))


#     key = 'Conv2'
#     vmin = activation[key][i_sample,:,:,:].min()
#     vmax = activation[key][i_sample,:,:,:].max()
#     for i_ch in range(activation[key].shape[1]):
#         ax = fig.add_subplot(gs[i_ch, 2])
#         ax.imshow(activation[key][i_sample,i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('i_ch: {}\n{}'.format(i_ch, key))


#     key = 'Conv3'
#     vmin = activation[key][i_sample,:,:,:].min()
#     vmax = activation[key][i_sample,:,:,:].max()
#     for i_ch in range(activation[key].shape[1]):
#         ax = fig.add_subplot(gs[i_ch, 3])
#         ax.imshow(activation[key][i_sample,i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('i_ch: {}\n{}'.format(i_ch, key))

#     key = 'Up_conv3'
#     vmin = activation[key][i_sample,:,:,:].min()
#     vmax = activation[key][i_sample,:,:,:].max()
#     for i_ch in range(activation[key].shape[1]):
#         ax = fig.add_subplot(gs[i_ch, 4])
#         ax.imshow(activation[key][i_sample,i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('i_ch: {}\n{}'.format(i_ch, key))

#     key = 'Up_conv2'
#     vmin = activation[key][i_sample,:,:,:].min()
#     vmax = activation[key][i_sample,:,:,:].max()
#     for i_ch in range(activation[key].shape[1]):
#         ax = fig.add_subplot(gs[i_ch, 5])
#         ax.imshow(activation[key][i_sample,i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('i_ch: {}\n{}'.format(i_ch, key))

#     # 6. plot output
#     vmin = out[i_sample,:,:,:].min()
#     vmax = out[i_sample,:,:,:].max()
#     #     print(vmin, vmax)
#     for i_out in range(out.shape[1]):
#         ax = fig.add_subplot(gs[i_out, 6])
#         ax.imshow(out[i_sample, i_out,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('out ch: {}'.format(i_out) )

#     vmin = label[i_sample,:2,:,:].min()
#     vmax = label[i_sample,:2,:,:].max()
#     for i_label in range(label.shape[1]-1):
#         ax = fig.add_subplot(gs[i_label, 7])
#         ax.imshow(label[i_sample, i_label,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('label ch: {}'.format(i_label) )

#     fig.subplots_adjust(wspace=0.2, hspace=0.8)

# #     if fig_name is None:
# #         fig_name = 'featuremap'

# #     fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample), fontsize=20)

#     # if log_wandb:
#     #     wandb.log({fig_name: wandb.Image(fig)})

#     # if outputdir is not None:
#     #     if not os.path.exists(outputdir):
#     #         os.makedirs(outputdir)

#     #     fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

#     # if show_plot == False:
#     #     plt.close(fig)
#     #     pyplot.close(fig)
#     #     plt.close('all')

#     plt.show()
    
#     fig.subplots_adjust(wspace=0.2, hspace=0.8)

#     if fig_name is None:
#         fig_name = 'featuremap'

#     fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample) + '\nRRout: {:.3f} RRlabel: {:.3f} AE: {:.3f} [BPM]'.format(
#         RR_model[i_sample].squeeze(), RR_label[i_sample].squeeze(), error_abs[i_sample]), fontsize=15)
    
#     if log_wandb:
#         wandb.log({fig_name: wandb.Image(fig)})

#     if outputdir is not None:
#         if not os.path.exists(outputdir):
#             os.makedirs(outputdir)

#         fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

#     if show_plot == False:
#         plt.close(fig)
#         pyplot.close(fig)
#         plt.close('all')

#     plt.show()


# In[ ]:


# def check_activation(model, dataloaders, training_params, mode='random', fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
#     data, out, label, RR_model, RR_label, activation = get_data_activation(model, dataloaders, training_params)
#     b, c, h, w = data.shape    
#     error_abs = np.abs(RR_model - RR_label).squeeze()
    
    
#     if mode=='worst':
#         i_sample = np.argmax(error_abs)
#     if mode=='best':
#         i_sample = np.argmin(error_abs)
#     if mode=='random':
#         # check one sample only
# #         np.random.seed(0)
#         i_sample = np.random.randint(b)

# #     print(error_abs)
# #     print(error_abs[i_sample])
# #     print(error_abs.min(), error_abs.max())



#     fig = plt.figure(figsize=(15,8), constrained_layout=True, dpi=100)
#     gs = gridspec.GridSpec(4, 8)

#     # gs.update(wspace = 0.5, hspace = 0.8)
#     # i_sample = 50


#     ts = np.arange(h)*3 # 1Hz
#     RR_range = label_range_dict['RR']

#     extent = [ts[0], ts[-1], RR_range[0], RR_range[-1]]

#     # 1. plot the input
#     vmin = data[i_sample,:,:,:].min()
#     vmax = data[i_sample,:,:,:].max()
#     for i_input, input_name in enumerate(training_params['input_names']):
#         ax = fig.add_subplot(gs[i_input, 0])
#         ax.imshow(data[i_sample,i_input,:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title(input_name)

#     # 2. plot the 1st average pooling output in SE block
#     vmin = activation['SE1_conv1'][i_sample,:,:,:].min()
#     vmax = activation['SE1_conv1'][i_sample,:,:,:].max()
#     for i_input, input_name in enumerate(training_params['input_names']):
#         ax = fig.add_subplot(gs[i_input, 1])
#         ax.imshow(activation['SE1_conv1'][i_sample,i_input,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title(input_name+'\nSE1_conv1')
        
#     # 2. plot the 1st average pooling output in SE block
#     vmin = activation['SE1_conv2'][i_sample,:,:,:].min()
#     vmax = activation['SE1_conv2'][i_sample,:,:,:].max()
#     for i_input, input_name in enumerate(training_params['input_names']):
#         ax = fig.add_subplot(gs[i_input, 2])
#         ax.imshow(activation['SE1_conv2'][i_sample,i_input,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title(input_name+'\nSE1_conv2')

#     # 5. plot the weights for each input
#     ax = fig.add_subplot(gs[:, 3])
#     ax.barh(training_params['input_names'][::-1], activation['SE1_avg_pool'].squeeze()[i_sample,:][::-1], height=0.1)
#     ax.set_title('SE1_avg_pool')
#     ax_no_top_right(ax)
    
# #     5. plot the weights for each input
#     ax = fig.add_subplot(gs[:, 4])
# #     ax.barh(training_params['input_names'][::-1], activation['SE1_fc1'][i_sample,:][::-1], height=0.1)
#     ax.barh(np.arange(activation['SE1_fc1'].shape[-1]), activation['SE1_fc1'][i_sample,:][::-1], height=0.1)
#     ax.set_title('SE1_fc1')
#     ax_no_top_right(ax)
    
#     # 5. plot the weights for each input
#     ax = fig.add_subplot(gs[:, 5])
#     ax.barh(training_params['input_names'][::-1], activation['SE1_fc2'][i_sample,:][::-1], height=0.1)
#     ax.set_title('SE1_fc2')
#     ax_no_top_right(ax)

#     # 6. plot the weights for each input
#     ax = fig.add_subplot(gs[:, 6])
#     ax.barh(training_params['input_names'][::-1], activation['SE1_weight'][i_sample,:][::-1], height=0.1)
#     ax.set_title('SE1_weight')
#     ax_no_top_right(ax)
    
#     if verbose:
# #         print('SE1_conv1', vmin, vmax)

# #         print('input', vmin, vmax)

# #         print('SE1_conv2', vmin, vmax)

#         print('SE1_avg_pool', activation['SE1_avg_pool'][i_sample,:])

#         print('SE1_fc1', activation['SE1_fc1'][i_sample,:])

#         print('SE1_fc2', activation['SE1_fc2'][i_sample,:])

#         print('SE1_weight', activation['SE1_weight'][i_sample,:])
    
# #     # 6. plot the weights for each input
# #     ax = fig.add_subplot(gs[:, 6])
# #     ax.barh(training_params['input_names'][::-1], activation['SE1_conv'][i_sample,:][::-1], height=0.1)
# #     ax.set_title('SE1_conv')
# #     ax_no_top_right(ax)
# #     print('SE1_conv', activation['SE1_conv'][i_sample,:])


#     # 6. plot output
#     vmin = out[i_sample,:,:,:].min()
#     vmax = out[i_sample,:,:,:].max()
# #     print(vmin, vmax)
#     for i_out in range(out.shape[1]):
#         ax = fig.add_subplot(gs[i_out, -1])
#         ax.imshow(out[i_sample, i_out,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#         ax.set_title('out ch: {}'.format(i_out) )
        

# #     print(label.shape)
#     vmin = label[i_sample,-1,:,:].min()
#     vmax = label[i_sample,-1,:,:].max()
# #     for i_label in range(label.shape[1]):
#     ax = fig.add_subplot(gs[i_out+2, -1])
#     ax.imshow(label[i_sample, -1,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#     ax.set_title('label ch: {}'.format(-1) )
        

#     fig.subplots_adjust(wspace=0.2, hspace=0.8)


#     if fig_name is None:
#         fig_name = 'IO_weights'

#     fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample), fontsize=20)

#     if log_wandb:
#         wandb.log({fig_name: wandb.Image(fig)})

#     if outputdir is not None:
#         if not os.path.exists(outputdir):
#             os.makedirs(outputdir)

#         fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

#     if show_plot == False:
#         plt.close(fig)
#         pyplot.close(fig)
#         plt.close('all')

#     plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# model = nn.ModuleDict()
# model['i'] = U_Net(in_ch=1, out_ch=2, n1 = training_params['N_channels'], training_params=training_params)

# U_Net(in_ch=1, out_ch=2, n1 = training_params['N_channels'], training_params=training_params)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# for task in training_params['regressor_names']:
#     if task!=main_task:
#         continue

#     plot_BA(df_outputlabel_val[main_task], main_task, fig_name='BA_val_{}'.format(main_task), show_plot=False, outputdir=outputdir+'model_output/', log_wandb=training_params['wandb'])
#     plot_regression_all_agg(df_outputlabel_val[main_task], df_performance_val[main_task], outputdir=outputdir+'model_output/', show_plot=False, log_wandb=training_params['wandb'])

# # log metrices on wnadb
# if training_params['wandb']==True:
#     # W&B
#     label = df_outputlabel_val[main_task]['label'].values
#     label_est = df_outputlabel_val[main_task]['label_est'].values
# #         print(label.shape, label)
# #         print(label_est.shape, label_est)

#     PCC = get_PCC(label, label_est)
#     Rsquared = get_CoeffDeterm(label, label_est)
#     MAE, _ = get_MAE(label, label_est)
#     RMSE = get_RMSE(label, label_est)
#     MAPE, _ = get_MAPE(label, label_est)

#     wandb.log(
#         {
#             'val_MAE': MAE,
#             'val_RMSE': RMSE,
#             'val_MAPE': MAPE,
#             'val_PCC': PCC,
#             'val_Rsquared': Rsquared,
#         })


# In[ ]:



# # performance_dict_TESTc, model_out_dict_TESTc = get_performance(model, dataloaders['valc'], training_params)

# if plot_results:
#     # model out is based on interpolated labels
#     plot_unet_results(model_out_dict_TRAIN, training_params, title_str='TRAIN', outputdir=outputdir+'TRAIN-TEST/')
#     plot_unet_results(model_out_dict_TEST, training_params, title_str='TEST', outputdir=outputdir+'TRAIN-TEST/')

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# label_train, label_est_train


# In[ ]:


# df_performance_val[task]


# In[ ]:



# get_PCC(label_train.squeeze(), label_est_train.squeeze())


# In[ ]:





# # print results for Reviewer's responses

# In[ ]:


# RR_input


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# model_out_dict_TRAIN['RR_input']
# model_out_dict_TRAIN['RR_label']


# In[ ]:


# model_out_dict_TRAIN['meta'][:,0]==subject_id


# In[ ]:


# plot_results


# In[ ]:


# model_out_dict_TESTc['RR_model'].shape, model_out_dict_TESTc['ts'].shape
# model_out_dict_TESTc.keys()


# In[ ]:


# MAE_mean_model_TEST = performance_dict_TEST['MAE_mean_model']
# MAPE_mean_model_TEST = performance_dict_TEST['MAPE_mean_model']
# MAE_mean_input_TEST = performance_dict_TEST['MAE_mean_input']
# MAPE_mean_input_TEST = performance_dict_TEST['MAPE_mean_input']
# MAE_std_model_TEST = performance_dict_TEST['MAE_std_model']
# MAPE_std_model_TEST = performance_dict_TEST['MAPE_std_model']
# MAE_std_input_TEST = performance_dict_TEST['MAE_std_input']
# MAPE_std_input_TEST = performance_dict_TEST['MAPE_std_input']

# if plot_results:
#     # scatter is based on cosmed datapoints
#     plot_unet_scatter_compare(model_out_dict_TEST,  outputdir+'TRAIN-TEST/', show_plot=True)
#     plot_unet_scatter_single(model_out_dict_TEST,  outputdir+'TRAIN-TEST/', show_plot=True)

# plt.show()
# # get_performance_model_time = time.time()
# # time_elapsed = time.time() - start_time

# # predictions_dict = {'MAE_mean_model_val': [MAE_mean_model_TEST*60],
# #                     'MAE_std_model_val': [MAE_std_model_TEST*60],
# #                     'MAPE_mean_model_val': [MAPE_mean_model_TEST*100],
# #                     'MAPE_std_model_val': [MAPE_std_model_TEST*100],
# #                     'MAE_mean_input_val': [MAE_mean_input_TEST*60],
# #                     'MAE_std_input_val': [MAE_std_input_TEST*60],
# #                     'MAPE_mean_input_val': [MAPE_mean_input_TEST*100],
# #                     'MAPE_std_input_val': [MAPE_std_input_TEST*100],
# #                     'time_elapsed': [time_elapsed]}

# # df_performance_TEST = df_performance.append(pd.DataFrame.from_dict(predictions_dict), ignore_index=True)
# # df_performance_TEST.to_csv(outputdir+'TRAIN-TEST/df_performance_TEST.csv', index=False)



# # TEST_dict = {
# #     'MAE input': ['{:.2f} ± {:.2f} bpm'.format(MAE_mean_input_TEST*60, MAE_std_input_TEST*60)],
# #     'MAE model': ['{:.2f} ± {:.2f} bpm'.format(MAE_mean_model_TEST*60, MAE_std_model_TEST*60)],
# #     'MAPE input': ['{:.2f} ± {:.2f} %'.format(MAPE_mean_input_TEST*100, MAPE_std_input_TEST*100)],
# #     'MAPE model': ['{:.2f} ± {:.2f} %'.format(MAPE_mean_model_TEST*100, MAPE_std_model_TEST*100)]
# # }
# # df_pretty_performance = df_pretty_performance.append(pd.DataFrame.from_dict(TEST_dict), ignore_index=True)
# # df_pretty_performance.to_csv(outputdir+'TRAIN-TEST/df_pretty_performance.csv', index=False)



# # RR_label = model_out_dict_TESTc['RR_label']
# # RR_input = model_out_dict_TESTc['RR_input']
# # RR_model = model_out_dict_TESTc['RR_model']
# # meta = model_out_dict_TESTc['meta']
# # hr_TEST = dataloaders['valc'].dataset.hr
# # ts = model_out_dict_TESTc['ts']

# # i_shift = hr_TEST.shape[1]//2
# # hr = hr_TEST[:,i_shift]


# # eval_matrix = np.c_[RR_label, RR_input, RR_model, meta, hr, ts]

# # # column_names = 'RR_label'+ 'RR_input'+ 'RR_model'+ 'meta'
# # column_names = ['RR_label']

# # for i in range(RR_input.shape[1]):
# #     column_names.append('RR_input{}'.format(i))
    
# # column_names.append('RR_model')
# # column_names.append('meta')
# # column_names.append('hr')
# # column_names.append('ts')


# # df_RR_TEST = pd.DataFrame(eval_matrix, columns=column_names)
# # df_RR_TEST.to_csv(outputdir+'TRAIN-TEST/df_RR_TEST.csv', index=False)

# # print('\nDONE!')


# In[ ]:


# # plt.plot(ts, hr)

# plt.plot(ts, RR_label)
# # plt.plot(ts, RR_input[:,1])
# plt.plot(ts, RR_model)


# In[ ]:





# In[ ]:





# In[ ]:





# # aggregate results

# In[ ]:


RR_label = model_out_dict_TEST['RR_label']
RR_input = model_out_dict_TEST['RR_input']
RR_model = model_out_dict_TEST['RR_model']
meta = model_out_dict_TEST['meta']


print('subject id\tinput prediction\t\tmodel prediction')
print('============================================================================')
df_sub_performance = pd.DataFrame()

print_performance = True

for subject_id in np.unique(meta[:,0]):

    indices_sub = np.where(meta==subject_id)[0]
    MAE_mean_input, MAE_std_input = get_MAE(RR_label[indices_sub], RR_input[indices_sub].mean(axis=1, keepdims=True))
    MAPE_mean_input, MAPE_std_input = get_MAPE(RR_label[indices_sub], RR_input[indices_sub].mean(axis=1, keepdims=True))

    MAE_mean_model, MAE_std_model = get_MAE(RR_label[indices_sub], RR_model[indices_sub])
    MAPE_mean_model, MAPE_std_model = get_MAPE(RR_label[indices_sub], RR_model[indices_sub])

    if print_performance == True:

        print('{}\t\tMAE: {:.2f} ± {:.2f} bpm\t\tMAE: {:.2f} ± {:.2f} bpm'.format(int(subject_id),MAE_mean_input*60, MAE_std_input*60, MAE_mean_model*60, MAE_std_model*60))


    sub_dict = {
        'subject_id': ['{}'.format(int(subject_id))],
        'MAE input': ['{:.2f} ± {:.2f} bpm'.format(MAE_mean_input*60, MAE_std_input*60)],
        'MAE model': ['{:.2f} ± {:.2f} bpm'.format(MAE_mean_model*60, MAE_std_model*60)],
        'MAPE input': ['{:.2f} ± {:.2f} %'.format(MAPE_mean_input*100, MAPE_std_input*100)],
        'MAPE model': ['{:.2f} ± {:.2f} %'.format(MAPE_mean_model*100, MAPE_std_model*100)]
    }
    df_sub_performance = df_sub_performance.append(pd.DataFrame.from_dict(sub_dict), ignore_index=True)

df_sub_performance.to_csv(outputdir+'TRAIN-TEST/df_sub_performance.csv', index=False)


# In[ ]:


df_sub_performance


# In[ ]:


plt.scatter()


# In[ ]:





# In[ ]:





# # grave

# In[ ]:


# def bland_altman_plot(m1, m2,
#                       sd_limit=1.96,
#                       ax=None,
#                       scatter_kwds=None,
#                       mean_line_kwds=None,
#                       limit_lines_kwds=None):
#     """
#     Bland-Altman Plot.
#     A Bland-Altman plot is a graphical method to analyze the differences
#     between two methods of measurement. The mean of the measures is plotted
#     against their difference.
#     Parameters
#     ----------
#     m1, m2: pandas Series or array-like
#     sd_limit : float, default 1.96
#         The limit of agreements expressed in terms of the standard deviation of
#         the differences. If `md` is the mean of the differences, and `sd` is
#         the standard deviation of those differences, then the limits of
#         agreement that will be plotted will be
#                        md - sd_limit * sd, md + sd_limit * sd
#         The default of 1.96 will produce 95% confidence intervals for the means
#         of the differences.
#         If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
#         defaults to 3 standard deviatons on either side of the mean.
#     ax: matplotlib.axis, optional
#         matplotlib axis object to plot on.
#     scatter_kwargs: keywords
#         Options to to style the scatter plot. Accepts any keywords for the
#         matplotlib Axes.scatter plotting method
#     mean_line_kwds: keywords
#         Options to to style the scatter plot. Accepts any keywords for the
#         matplotlib Axes.axhline plotting method
#     limit_lines_kwds: keywords
#         Options to to style the scatter plot. Accepts any keywords for the
#         matplotlib Axes.axhline plotting method
#    Returns
#     -------
#     ax: matplotlib Axis object
#     """

#     if len(m1) != len(m2):
#         raise ValueError('m1 does not have the same length as m2.')
#     if sd_limit < 0:
#         raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))

#     means = np.mean([m1, m2], axis=0)
#     diffs = m1 - m2
#     mean_diff = np.mean(diffs)
#     std_diff = np.std(diffs, axis=0)

#     if ax is None:
#         ax = plt.gca()

#     scatter_kwds = scatter_kwds or {}
#     if 's' not in scatter_kwds:
#         scatter_kwds['s'] = 20
#     mean_line_kwds = mean_line_kwds or {}
#     limit_lines_kwds = limit_lines_kwds or {}
#     for kwds in [mean_line_kwds, limit_lines_kwds]:
#         if 'color' not in kwds:
#             kwds['color'] = 'gray'
#         if 'linewidth' not in kwds:
#             kwds['linewidth'] = 1
#     if 'linestyle' not in mean_line_kwds:
#         kwds['linestyle'] = '--'
#     if 'linestyle' not in limit_lines_kwds:
#         kwds['linestyle'] = ':'

#     ax.scatter(means, diffs, **scatter_kwds)
#     ax.axhline(mean_diff, **mean_line_kwds)  # draw mean line.

#     # Annotate mean line with mean difference.
#     ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
#                 xy=(0.99, 0.5),
#                 horizontalalignment='right',
#                 verticalalignment='center',
#                 fontsize=14,
#                 xycoords='axes fraction')

#     if sd_limit > 0:
#         half_ylim = (1.5 * sd_limit) * std_diff
#         ax.set_ylim(mean_diff - half_ylim,
#                     mean_diff + half_ylim)

#         limit_of_agreement = sd_limit * std_diff
#         lower = mean_diff - limit_of_agreement
#         upper = mean_diff + limit_of_agreement
#         for j, lim in enumerate([lower, upper]):
#             ax.axhline(lim, **limit_lines_kwds)
#         ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
#                     xy=(0.99, 0.07),
#                     horizontalalignment='right',
#                     verticalalignment='bottom',
#                     fontsize=14,
#                     xycoords='axes fraction')
#         ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
#                     xy=(0.99, 0.92),
#                     horizontalalignment='right',
#                     fontsize=14,
#                     xycoords='axes fraction')

#     elif sd_limit == 0:
#         half_ylim = 3 * std_diff
#         ax.set_ylim(mean_diff - half_ylim,
#                     mean_diff + half_ylim)

#     ax.set_ylabel('Difference', fontsize=15)
#     ax.set_xlabel('Means', fontsize=15)
#     ax.tick_params(labelsize=13)
#     plt.tight_layout()
#     return ax

# RR_label = model_out_dict_TEST['RR_label']*60
# RR_model = model_out_dict_TEST['RR_model']*60

# fig, ax = plt.subplots(1, figsize = (8,5), dpi=100)

# bland_altman_plot(RR_label, RR_model, ax=ax)


# In[ ]:





# In[ ]:


# TBD

# def plot_unet_scatter_sub(model_out_dict_train, model_out_dict_val, training_params, outputdir=None, show_plot=False):

#     fig = plt.figure(figsize=(21,7), dpi=100)

# #     model_out_dict_train = get_model_out(model, dataloader['train_eval'], training_params)
#     RR_label_train = model_out_dict_train['RR_label'][:,0]
# #     RR_input_train = model_out_dict_train['RR_input']
#     RR_model_train = model_out_dict_train['RR_model'][:,0]
#     meta_train = model_out_dict_train['meta']
#     N_subjects_train = np.unique(meta_train).shape[0]
    
# #     model_out_dict_val = get_model_out(model, dataloader['val'], training_params)
#     RR_label_val = model_out_dict_val['RR_label'][:,0]
#     RR_input_val = model_out_dict_val['RR_input'][:,0]
#     RR_model_val = model_out_dict_val['RR_model'][:,0]
#     meta_val = model_out_dict_val['meta']
#     N_subjects_val = np.unique(meta_val).shape[0]
    
#     RR_arr = np.r_[RR_label_train, RR_model_train, RR_label_val, RR_model_val]*60
# #     RR_range = [RR_arr.min(), RR_arr.max()]
    
#     RR_range = label_range_dict['br']
    
#     fontsize = 20
#     ax1 = fig.add_subplot(1, 3, 1)
#     colornames = list(color_dict.keys())
#     markernames = list(marker_dict.keys())

#     i = 0
#     for subject_id in np.unique(model_out_dict_train['meta']):
#         indices_sub = np.where(model_out_dict_train['meta']==subject_id)[0]
#         ax1.scatter(RR_label_train[indices_sub]*60, RR_model_train[indices_sub]*60, label='sub {}'.format(int(subject_id)), color='steelblue', marker=marker_dict[markernames[i%len(markernames)]], s=80, alpha=0.3)
#         i += 1
# #     ax1.scatter(RR_label_train*60, RR_model_train*60, label='model RR', color='steelblue', s=80, alpha=0.3)

#     ax1.plot( RR_range,RR_range , '--', color='gray', alpha=0.8)
    
#     # these are matplotlib.patch.Patch properties
#     props = dict(boxstyle='round,pad=0.7', facecolor='none', edgecolor='black', alpha=0.5)
# # bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
    
    
#     textstr = r'$r^{2}$' + '= {:.2f}'.format(r2_score(RR_label_train, RR_model_train)) + '\n' + \
#     'N = {} subejcts'.format(N_subjects_train) + '\n' + \
#     'MAE = {:.2f} bpm'.format(get_MAE(RR_label_train, RR_model_train)[0]*60) + '\n' + \
#     'MAPE = {:.2f} %'.format(get_MAPE(RR_label_train, RR_model_train)[0]*100)
    
#     # place a text box in upper left in axes coords
#     ax1.text(0.7, 0.08, textstr, transform=ax1.transAxes, fontsize=fontsize-5,
#     verticalalignment='bottom', horizontalalignment='left', bbox=props)
#     ax1.legend(loc='upper right', frameon=True)

#     ax1.set_title('Training RR', fontsize=fontsize*1.3)
#     ax1.set_xlabel('RR reference (bpm)', fontsize=fontsize)
#     ax1.set_ylabel('RR estimation (bpm)', fontsize=fontsize)
    
#     ax1.set_xlim(RR_range)
#     ax1.set_ylim(RR_range)
    


#     ax2 = fig.add_subplot(1, 3, 2)
#     for subject_id in np.unique(model_out_dict_val['meta']):
#         indices_sub = np.where(model_out_dict_val['meta']==subject_id)[0]
#         ax2.scatter(RR_label_val[indices_sub]*60, RR_model_val[indices_sub]*60, label='sub {}'.format(int(subject_id)), color='firebrick', marker=marker_dict[markernames[i%len(markernames)]], s=80, alpha=0.3)
#         i += 1

#     ax2.plot( RR_range,RR_range , '--', color='gray', alpha=0.8)

#     ax2.set_title('Validation RR', fontsize=fontsize*1.3)
#     ax2.set_xlabel('RR reference (bpm)', fontsize=fontsize)
#     ax2.set_ylabel('RR estimation (bpm)', fontsize=fontsize)
#     ax2.set_xlim(RR_range)
#     ax2.set_ylim(RR_range)
    
    
#     # these are matplotlib.patch.Patch properties
#     textstr = r'$r^{2}$' + '= {:.2f}'.format(r2_score(RR_label_val, RR_model_val)) + '\n' + \
#     'N = {} subejcts'.format(N_subjects_val) + '\n' + \
#     'MAE = {:.2f} bpm'.format(get_MAE(RR_label_val, RR_model_val)[0]*60) + '\n' + \
#     'MAPE = {:.2f} %'.format(get_MAPE(RR_label_val, RR_model_val)[0]*100)
    
#     # place a text box in upper left in axes coords
#     ax2.text(0.7, 0.08, textstr, transform=ax2.transAxes, fontsize=fontsize-5,
#     verticalalignment='bottom', horizontalalignment='left', bbox=props)
#     ax2.legend(loc='upper right', frameon=True)

    
    
#     ax3 = fig.add_subplot(1, 3, 3)

#     i -= np.unique(model_out_dict_val['meta']).shape[0]
#     for subject_id in np.unique(model_out_dict_val['meta']):
#         indices_sub = np.where(model_out_dict_val['meta']==subject_id)[0]
#         ax3.scatter(RR_label_val[indices_sub]*60, RR_input_val[indices_sub]*60, label='sub {}'.format(int(subject_id)), color='seagreen', marker=marker_dict[markernames[i%len(markernames)]], s=80, alpha=0.3)
#         i += 1
    

#     ax3.plot( RR_range,RR_range , '--', color='gray', alpha=0.8)

#     ax3.set_title('Validation RR (DSP)', fontsize=fontsize*1.3)
#     ax3.set_xlabel('RR reference (bpm)', fontsize=fontsize)
#     ax3.set_ylabel('RR estimation (bpm)', fontsize=fontsize)
#     ax3.set_xlim(RR_range)
#     ax3.set_ylim(RR_range)
#     ax3.legend(loc='upper right', frameon=True)
    

#     # these are matplotlib.patch.Patch properties
#     textstr = r'$r^{2}$' + '= {:.2f}'.format(r2_score(RR_label_val, RR_input_val)) + '\n' + \
#     'N = {} subejcts'.format(N_subjects_val) + '\n' + \
#     'MAE = {:.2f} bpm'.format(get_MAE(RR_label_val, RR_input_val)[0]*60) + '\n' + \
#     'MAPE = {:.2f} %'.format(get_MAPE(RR_label_val, RR_input_val)[0]*100)
    
#     # place a text box in upper left in axes coords
#     ax3.text(0.7, 0.08, textstr, transform=ax3.transAxes, fontsize=fontsize-5,
#     verticalalignment='bottom', horizontalalignment='left', bbox=props)
    
    
#     fig.tight_layout()
    
#     if outputdir is not None:
#         if not os.path.exists(outputdir):
#             os.makedirs(outputdir)
#         fig.savefig(outputdir + 'model_scatter.png', facecolor=fig.get_facecolor())
#     if show_plot == False:
#         plt.close(fig)
#         pyplot.close(fig)
#         plt.close('all')

        
# # plot_unet_scatter_sub(model_out_dict_TRAIN, model_out_dict_TEST, training_params, outputdir+'TRAIN-TEST/')


# In[ ]:




