#!/usr/bin/env python
# coding: utf-8

# # TODO: work on pred_dann so it adapts to the new architecture

# In[1]:


import numpy as np
np.random.seed(0) 

import argparse

import os
import math
from math import sin

import json

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import wandb

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets, models
from torchsummary import summary
torch.manual_seed(0)

i_seed = 0

from datetime import datetime
import pytz
import pprint

import sys
sys.path.append('../') # add this line so Data and data are visible in this file
sys.path.append('../../') # add this line so Data and data are visible in this file
sys.path.append('../PatchWand/') # add this line so Data and data are visible in this file

# from PatchWand import *
from plotting_tools import *
from handy_tools import *
from setting import *
from evaluate import *

from stage3_preprocess import *
from stage4_regression import *
from dataIO import *
from FL_extension.training_util import *
from FL_extension.dataset_util import *
from FL_extension.evaluation_util import *
from FL_extension.models import *
from FL_extension.models_CNNlight import *
# from unet_extension.training_util import *

from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:





# In[2]:


print(torch.version.cuda)


# In[ ]:





# In[3]:


parser = argparse.ArgumentParser(description='SpO2_estimate')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--subject_id', metavar='subject_id', help='subject_id',
                    default='101')
parser.add_argument('--training_params_file', metavar='training_params_file', help='training_params_file',
                    default='training_params_list.json')


# checklist 3: comment first line, uncomment second line
# args = parser.parse_args(['--input_folder', '../../data/stage3_FL/win8_overlap75', 
#                           '--output_folder', '../../data/stage4_FL/',
# #                           '--training_params_file', 'training_params_baseline.json',
#                           '--training_params_file', 'training_params_dummy.json',
#                          ])
# args = parser.parse_args(['--input_folder', '../../data/stage3/win60_overlap90/', 
#                           '--output_folder', '../../data/stage4_FL/',
#                           '--training_params_file', 'training_params_baseline.json',
# #                           '--training_params_file', 'training_params_STFT.json',
# #                           '--training_params_file', 'training_params_RespiratoryRegression.json',
#                          ])
args = parser.parse_args()
print(args)


# In[4]:


# if 1==1 :
#     print('hi')


# In[ ]:





# # start timing

# In[5]:



tz_NY = pytz.timezone('America/New_York') 
datetime_start = datetime.now(tz_NY)
print("start time:", datetime_start.strftime("%Y-%b-%d %H:%M:%S"))


# In[6]:


inputdir = args.input_folder
outputdir = args.output_folder
training_params_file = args.training_params_file

outputdir = outputdir + '{}/'.format(training_params_file.split('.json')[0].split('_')[-1])


if not os.path.exists(outputdir):
    os.makedirs(outputdir)


# In[7]:


# inputdir


# # get training params and dataloaders

# In[8]:


torch.cuda.is_available()


# In[9]:


def get_model_out_names(training_params):
    model_out_names = []

#     for output_name in training_params['output_names']:
    for output_name in training_params['output_names']+['domain']:
        for input_name in training_params['input_names']:
            model_out_names.append(output_name+'-{}'.format(input_name))
    return model_out_names


# In[10]:


def get_modality_dict(training_params):
    label_encoder = LabelEncoder()
    modality_encoded = label_encoder.fit_transform(training_params['input_names'])
#     modality_encoded # array([0, 1])

    modality_dict = {}
    for i_modality in modality_encoded:
        modality_dict[training_params['input_names'][i_modality]] = i_modality
#     training_params['modality_dict'] = modality_dict
    return modality_dict


# In[ ]:





# In[11]:


training_params_file


# In[12]:


with open(training_params_file) as json_file:
    training_params_list = json.load(json_file)

for training_params in [training_params_list[0]]:
    
    # 1. store device info
    # include device in training_params
    if training_params['cuda_i']==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
    training_params['device'] = device
    
    # 2. check training mode (subject_ind [default] vs. subject_specific)
    if 'training_mode' in training_params:
        training_mode = training_params['training_mode']
    else:
        training_params = 'subject_ind'
    
    # 3. transfer some info from stage3_dict to training_params
    stage3_dict = data_loader('stage3_dict', inputdir).item()
    training_params['list_signal'] = stage3_dict['list_signal']
    training_params['list_feature'] = stage3_dict['list_feature']
    training_params['list_output'] = stage3_dict['list_output']
    training_params['list_meta'] = stage3_dict['list_meta']
    training_params['FS_RESAMPLE_DL'] = stage3_dict['FS_RESAMPLE_DL']
    training_params['subject_ids'] = stage3_dict['subject_ids']
    training_params['task_ids'] = stage3_dict['task_ids']
    training_params['sequence'] = stage3_dict['sequence']
    
    # 4. get data loaders (TODO: load data into script once to save IO time)
    # [change it] first change CV_config so get_loaders prepare the data correctly though
    training_params['CV_config'] = {
        'subject_id': 113,
#         'task_id': 5,
    }
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    print('data dimensions are:', dataloaders['train'].dataset.data.shape)

    data_dimensions = dataloaders['train'].dataset.__getitem__(0)[0].size()
    training_params['data_dimensions'] = list(data_dimensions)
    del dataloaders
    
    # 5. get sweep name (is this necessary?)
    sweep_name = training_params['sweep_name'] 
    
    # 6. get the right feature extractor function
    if training_params['model_name'] == 'FeatureExtractor_CNN':
        training_params['feature_extractor'] = FeatureExtractor_CNN
    elif training_params['model_name'] == 'ResNet1D':
        training_params['feature_extractor'] = ResNet1D
    elif training_params['model_name'] == 'FeatureExtractor_CNN2':
        training_params['feature_extractor'] = FeatureExtractor_CNN2
    elif 'CNNlight' in training_params['model_name']: # designed for HR estimation
        training_params['feature_extractor'] = FeatureExtractor_CNNlight
        

    # 6. get the right feature extractor function
    if training_params['regressor_type'] == 'DominantFreqRegression':
        training_params['regressor'] = DominantFreqRegression
    elif training_params['regressor_type'] == 'FFTRegression':
        training_params['regressor'] = FFTRegression
    elif training_params['regressor_type'] == 'RespiratoryRegression':
        training_params['regressor'] = RespiratoryRegression

    # 7. store the ordered_subject_ids, inputdir, and outputdir
    training_params['ordered_subject_ids'] = np.asarray(training_params['ordered_subject_ids'])
#       "ordered_subject_ids": [101, 103, 104, 106, 107, 110, 111, 115, 116, 117, 118, 119, 120, 121, 113],
    # [101, 103, 104, 106, 111, 115, 116, 117, 118, 119, 120, 121, 113],
    
    training_params['inputdir'] = inputdir
    training_params['outputdir'] = outputdir
    
    # 8. get regressor_names ('domain-ECG_filt', 'HR_patch-ppg_ir_2_cardiac', etc.)
    # [change it]
    training_params = get_regressor_names(training_params)

    # 9. compute output_dim if it's LSTM
    # [change it]
    if 'LSTM' in training_params['model_name']:
        training_params = change_output_dim(training_params)

    # 10. get model output data keys ('domain-ECG_filt', 'HR_patch-ppg_ir_2_cardiac', etc., which tells me what data is used for computing the output)
    # [change it]
    training_params['model_out_names'] = get_model_out_names(training_params)
    
    # 11. encoder the modality to integers (the first is 0, the second is 1, etc.)
    # store the code in modality_dict
    training_params['modality_dict'] = get_modality_dict(training_params)

    # 12. get freq meta (what is the frequency for each spectral feature, mask it used label_range_dict['HR_DL'])
    training_params = update_freq_meta(training_params)

# training_params = training_params_list[0]


# In[21]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


training_params['model_out_names']


# In[14]:


training_params['regressor_names']


# In[ ]:





# In[15]:


# outputdir = outputdir+training_params['model_name']+'/'
# if not os.path.exists(outputdir):
#     os.makedirs(outputdir)
    


# # define trainer, evaler, preder functions

# In[16]:


trainer = train_dann
evaler = eval_dann
preder = pred_dann


# In[ ]:





# In[ ]:





# # some random param
# ## kernel_size = 5 ~ 16*100/300 (16 for 300 Hz sampling rate, 5 for 100 sampling rate)
# 

# In[ ]:





# In[17]:


if training_params['model_name'] == 'ResNet1D':
# reference: https://github.com/hsd1503/resnet1d/blob/master/test_physionet.py
    training_params['base_filters'] = training_params['channel_n'] # [64] 
    training_params['in_channels'] = training_params['data_dimensions']
#     training_params['increasefilter_gap'] = [training_params['downsample_gap'][0] * 2]
    training_params['n_block'] = training_params['n_block_macro'] * training_params['downsample_gap']
    training_params['increasefilter_gap'] = training_params['downsample_gap']
    


# In[ ]:





# In[18]:


# training_params['data_dimensions']


# In[ ]:





# # test the model

# In[ ]:





# In[20]:


def test_model_lstm(training_params):
    print('test_model_lstm')

    print('using model ', training_params['model_name'])

    # prepare model
    model = resp_multiverse(training_params=training_params)
    model = model.to(device).float()

    # prepare data
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    data = dataloaders['val'].dataset.data[:5,:,:]
    data = torch.from_numpy(data)

    feature = dataloaders['val'].dataset.feature[:5,:]
    feature = torch.from_numpy(feature)

    label = dataloaders['val'].dataset.label[:5,:]
    label = torch.from_numpy(label)

    data = data.to(device=device, dtype=torch.float)
    feature = feature.to(device=device, dtype=torch.float)
    label = label.to(device=device, dtype=torch.float)

    # model inference
    out = model(data, feature)

    # compute loss
    criterion = MultiTaskLoss(training_params)
    losses = criterion(out, label)

    # check losses
    print(losses)
    del model

    
def test_model(training_params):
    print('test_model')

    print('using model ', training_params['model_name'])

    model = resp_multiverse(training_params=training_params)
    summary(model, input_size=[tuple(training_params['data_dimensions']), (model.N_features,1)], device='cpu')
    print(model)
    del model
    
def test_model_dann(training_params):
    print('test_model_dann')
    print('using model ', training_params['model_name'])

    model = resp_DANN(training_params=training_params)
    print(model)

    summary(model, input_size=[tuple(training_params['data_dimensions']), (model.N_features,1)], device='cpu')
    
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable pytorch_total_params:', pytorch_total_params)
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('all pytorch_total_params:', pytorch_total_params)

    del model



debug_model = True
if debug_model==True:
    if 'LSTM' in training_params['model_name']:
        test_model_lstm(training_params)
    elif 'DANN' not in training_params['model_name']:
        test_model(training_params)
    else:
        test_model_dann(training_params)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # TODO
# ## make sure data can pass through the model

# In[20]:


check_data_flow = False

if check_data_flow:

    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    data = dataloaders['train'].dataset.data[:5,:,:]
    # data = torch
    data = torch.from_numpy(data)
    data = data.to(device=device, dtype=torch.float)

    feature = dataloaders['train'].dataset.feature[:5,:]
    # data = torch
    feature = torch.from_numpy(feature)
    feature = feature.to(device=device, dtype=torch.float)

    label = dataloaders['train'].dataset.label[:5,:]
    # data = torch
    label = torch.from_numpy(label)
    label = label.to(device=device, dtype=torch.float)

    model = resp_DANN(training_params=training_params)
    model = model.to(device).float()
    output = model(data, feature)

    print(data.size(), feature.size(), label.size(), output['domain-ECG_filt'].size())


# In[ ]:





# In[ ]:





# In[ ]:





# ## train, val, eval model

# In[21]:


if training_params['wandb']:
    wandb.login()
    os.environ["WANDB_DIR"] = os.path.abspath(outputdir)
    os.environ["WANDB_NOTEBOOK_NAME"] = 'feature_learning'


# In[ ]:





# In[ ]:





# In[22]:


def get_sweep_folder(training_params):
    n_block = training_params['n_block']
    inputs_combined = '+'.join([ i_name.split('_')[0] for i_name in training_params['input_names']])
    auxillary_weight = training_params['loss_weights']['auxillary_task']
    adversarial_weight = training_params['adversarial_weight']
    channel_n = training_params['channel_n']
    
    list_act = '+'.join( [str(int) for int in training_params['activity_names']] )
    
    sweep_folder = '{}blocks-{}-weight{}+{}-{}ch-act{}-{}'.format(n_block, inputs_combined, auxillary_weight, adversarial_weight, channel_n, list_act, training_params['regressor_type'])
    
    return sweep_folder


# In[ ]:





# In[ ]:





# In[23]:


def get_outputdirs(training_params):

    outputdir = training_params['outputdir']
    sweep_folder = get_sweep_folder(training_params)
    outputdir_sweep = outputdir+'{}/'.format(sweep_folder)

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

    outputdir_feature = outputdir_sweep + 'feature_visualization/'
    if outputdir_feature is not None:
        if not os.path.exists(outputdir_feature):
            os.makedirs(outputdir_feature)

    training_params['outputdir_sweep'] = outputdir_sweep
    training_params['outputdir_numeric'] = outputdir_numeric
    training_params['outputdir_modelout'] = outputdir_modelout
    training_params['outputdir_activation'] = outputdir_activation
    training_params['outputdir_feature'] = outputdir_feature

    return training_params


# In[24]:


training_params['data_dimensions']


# In[ ]:





# In[25]:


debug_auxillary = False

def train_master(training_params):
    
    # TODO: change all to training_params['xxx'] = get_xxx(training_params)
    training_params = get_outputdirs(training_params) # could be tricky since it changes several keys
    training_params = get_regressor_names(training_params) # may not need this in this task
    training_params['model_out_names'] = get_model_out_names(training_params)
    training_params['modality_dict'] = get_modality_dict(training_params)
    training_params = update_freq_meta(training_params) # could be tricky since it changes several keys

    
#     pprint.pprint(training_params)

    if len(training_params['input_names'])==1:
#         if  training_params['loss_weights']['auxillary_task']!=0:
        if training_params['adversarial_weight']!=0 or training_params['loss_weights']['auxillary_task']!=0:
            print('ony one signal, no need to try all DA weights')
            return
        

    
    df_performance_train = {}
    df_performance_val = {}

    df_outputlabel_train = {}
    df_outputlabel_val = {}

#     for task in training_params['tasks']:
#     for task in training_params['regressor_names']:
    for task in training_params['model_out_names']:

        df_performance_train[task] = pd.DataFrame()
        df_performance_val[task] = pd.DataFrame()

        df_outputlabel_train[task] = pd.DataFrame()
        df_outputlabel_val[task] = pd.DataFrame()

#     ordered_subject_ids = np.asarray([115, 107, 113, 110, 101, 104, 106, 121, 212, 102, 103, 111, 114, 116, 118, 119, 120])
#     ordered_subject_ids = np.asarray([107, 113, 110, 101, 104, 115, 106, 121, 212, 102, 103, 111, 114, 116, 118, 119, 120])
#         ordered_subject_ids = np.asarray([101, 110, 113, 119, 115, 107, 104, 106, 121, 212, 102, 103, 111, 114, 116, 118, 120])
#     ordered_subject_ids = np.asarray([101, 110, 113, 119, 115])

    ordered_subject_ids = training_params['ordered_subject_ids']
    main_task = training_params['output_names'][0].split('-')[0]
    i_activity = training_params['list_meta'].index('task') #  i_activity th column has the activity info

    # look at the interesting subjects first so I can quickly debug
    for i_CV, subject_id in enumerate(ordered_subject_ids):
        
#         if subject_id !=110:
#             continue
        
        if 'CV_max' in training_params:
            if i_CV >= training_params['CV_max']:
                continue

        training_params['CV_config']['subject_id'] = subject_id

        device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
        print('using device', device)

        
        # need to load the data for each LOSO CV
        dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
        print('data dimensions are:', dataloaders['train'].dataset.data.shape)
        print('dataset_sizes: ', dataset_sizes)

        # update the dimension so the model is created correctly
        data_dimensions = dataloaders['train'].dataset.__getitem__(0)[0].size()
        training_params['data_dimensions'] = list(data_dimensions)

        print('using model ', training_params['model_name'])

#         training_params = get_regressor_names(training_params)
#         model = resp_multiverse(training_params=training_params)
        model = resp_DANN(training_params=training_params)

#         print(model)
        
        model = model.to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=0.01)

#         criterion = MultiTaskLoss(training_params)
        criterion = AdversarialLoss(training_params)

        
        training_params['criterion'] = criterion
        training_params['optimizer'] = optimizer
        training_params['inputdir'] = inputdir
        
        
#         print( training_params['regressor_names'])
        CV_dict = train_model(model, training_params, dataloaders, trainer, evaler, preder)

#         print(CV_dict['performance_dict_val']['out_dict'])
#         sys.exit()

        plot_losses(CV_dict, outputdir=training_params['outputdir_sweep'], show_plot=False)
        

#         TODO: fix this code
#         print(training_params['regressor_names'])
        
#         for task in CV_dict['performance_dict_train']['out_dict'].keys():
#         for task in training_params['regressor_names']:
        for task in training_params['model_out_names']:
            if 'domain' in task:
                continue
        
            label_est_val = CV_dict['performance_dict_val']['out_dict'][task]
            label_val = CV_dict['performance_dict_val']['label_dict'][task]
            activity_val = CV_dict['performance_dict_val']['meta_arr'][:,i_activity]
            

            label_est_train = CV_dict['performance_dict_train']['out_dict'][task]
            label_train = CV_dict['performance_dict_train']['label_dict'][task]
            activity_train = CV_dict['performance_dict_train']['meta_arr'][:,i_activity]

#             if 'domain' in task:
#                 np.argmax(a, axis=1)
            
            
            # rescale the label after making estimations
            if 'perc' in training_params['output_names'][0]:
                i_meta = training_params['meta_names'].index('EEavg_est')
#                 print(CV_dict['performance_dict_train']['meta_arr'], CV_dict['performance_dict_train']['meta_arr'].shape)
                meta_train = CV_dict['performance_dict_train']['meta_arr'][:, i_meta]
                meta_val = CV_dict['performance_dict_val']['meta_arr'][:, i_meta]

                label_train = label_train*meta_train
                label_val = label_val*meta_val
                label_est_train = label_est_train*meta_train
                label_est_val = label_est_val*meta_val
            elif 'weighted' in training_params['output_names'][0]:
                i_meta = training_params['meta_names'].index('weight')
                meta_train = CV_dict['performance_dict_train']['meta_arr'][:, i_meta]
                meta_val = CV_dict['performance_dict_val']['meta_arr'][:, i_meta]

                label_train = label_train*meta_train
                label_val = label_val*meta_val
                label_est_train = label_est_train*meta_train
                label_est_val = label_est_val*meta_val

            
            # get performance df for training and testing dataset
            df_performance_train[task] = df_performance_train[task].append( get_df_performance(label_train, label_est_train, subject_id, task), ignore_index=True )

            df_performance_train[task].to_csv(training_params['outputdir_numeric'] + 'df_performance_train_{}.csv'.format(task), index=False)

            
#             print(task , label_train, label_train.shape )
            
            
            df_outputlabel_train[task] = df_outputlabel_train[task].append(
                pd.DataFrame( {
                'label_est': label_est_train,
                'label': label_train,
                'CV': [subject_id]*label_train.shape[0],
                'task': [task]*label_train.shape[0],
                'activity': np.vectorize(tasks_dict_reversed.get)(activity_train)
                }), ignore_index=True )

            df_outputlabel_train[task].to_csv(training_params['outputdir_numeric'] + 'df_outputlabel_train_{}.csv'.format(task), index=False)

            df_performance_val[task] = df_performance_val[task].append( get_df_performance(label_val, label_est_val, subject_id, task), ignore_index=True )
            df_performance_val[task].to_csv(training_params['outputdir_numeric'] + 'df_performance_val_{}.csv'.format(task), index=False)

            df_outputlabel_val[task] = df_outputlabel_val[task].append(
                pd.DataFrame( {
                'label_est': label_est_val,
                'label': label_val,
                'CV': [subject_id]*label_val.shape[0],
                'task': [task]*label_val.shape[0],
                'activity': np.vectorize(tasks_dict_reversed.get)(activity_val)
                }), ignore_index=True )

            df_outputlabel_val[task].to_csv(training_params['outputdir_numeric'] + 'df_outputlabel_val_{}.csv'.format(task), index=False)

            # plot performance training and testing dataset
            if (main_task not in task) and (debug_auxillary==False):
                continue
            
#             plot_regression(df_outputlabel_train[task], df_performance_train[task], task, fig_name='regression_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')
#             plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

            plot_regression(df_outputlabel_val[task], df_performance_val[task], task, training_params, fig_name='regression_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
#             plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

#             plot_output(df_outputlabel_train[task], task, fig_name = 'outputINtime_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout)
        
        if training_params['regressor_type']=='DominantFreqRegression':
            check_featuremap(model, training_params, mode='worst', fig_name = 'DL_activation_{}_'.format(subject_id), outputdir=training_params['outputdir_activation']+'worst/{}/'.format(subject_id), show_plot=False)
            check_featuremap(model, training_params, mode='best', fig_name = 'DL_activation_{}_'.format(subject_id), outputdir=training_params['outputdir_activation']+'best/{}/'.format(subject_id), show_plot=False)
        
        del model
        torch.cuda.empty_cache()


    for task in training_params['model_out_names']:
        if main_task not in task:
            continue
#         if task!=main_task:
#             continue
#         plot_regression_all_agg(df_outputlabel_train[task], df_performance_train[task], training_params, fig_name='LinearR_agg_train_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'])
        plot_regression(df_outputlabel_train[task], df_performance_train[task], task, training_params, fig_name='regression_train_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'])
        plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'])

#         plot_regression_all_agg(df_outputlabel_val[task], df_performance_val[task], training_params, fig_name='LinearR_agg_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
        plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])

        plot_output(df_outputlabel_val[task], task, fig_name = 'outputINtime_val_{}'.format(task),  show_plot=False, outputdir=training_params['outputdir_modelout'])

#     plot_BA(df_outputlabel_val[main_task], main_task, fig_name='BA_val_{}'.format(main_task), show_plot=False, outputdir=outputdir+'model_output/', log_wandb=training_params['wandb'])
#     plot_regression_all_agg(df_outputlabel_val[main_task], df_performance_val[main_task], outputdir=outputdir+'model_output/', show_plot=False, log_wandb=training_params['wandb'])

    # log metrices on wnadb
    if training_params['wandb']==True:
        main_task = training_params['model_out_names'][0]

        for task in training_params['model_out_names']:
            if 'domain' in task:
                continue
            
            label = df_outputlabel_val[task]['label'].values
            label_est = df_outputlabel_val[task]['label_est'].values

            PCC = get_PCC(label, label_est)
            Rsquared = get_CoeffDeterm(label, label_est)
            MAE, _ = get_MAE(label, label_est)
            RMSE = get_RMSE(label, label_est)
            MAPE, _ = get_MAPE(label, label_est)
            wandb.log(
                {
                    '{}_MAE'.format(task): MAE,
#                     '{}_RMSE'.format(task): RMSE,
#                     '{}_MAPE'.format(task): MAPE,
                    '{}_PCC'.format(task): PCC,
                    '{}_Rsquared'.format(task): Rsquared,
                })
            
            if task == main_task:
                wandb.log(
                    {
                        'val_MAE'.format(task): MAE,
    #                     '{}_RMSE'.format(task): RMSE,
    #                     '{}_MAPE'.format(task): MAPE,
                        'val_PCC'.format(task): PCC,
                        'val_Rsquared'.format(task): Rsquared,
                    })


        
    data_saver(training_params, 'training_params', training_params['outputdir_sweep'])
    # training_params = data_loader('training_params', outputdir).item()
        
    print('done!')


# In[26]:


# training_params['model_out_names']


# In[27]:


# training_params['model_out_names'][0]


# In[ ]:





# In[28]:


# import torch

# x = torch.randn( 3, 4)
# y = torch.softmax(x, dim=-1)


# In[29]:


# x
# y


# In[30]:


# y.sum(axis=-1)


# In[ ]:





# In[31]:


def train_sweep(config=None):

#     with wandb.init(config=config, entity='inanlab', project="[TL] stage2_cnn", reinit=True, dir=outputdir):
    with wandb.init(config=config, reinit=True, dir=outputdir):

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
#         print(config)
        pprint.pprint(config)

        # init the model
        
        # things that need to be change when using a new config
        for key in config.keys():
            if key=='loss_weights':
#                 training_params[key]['RR_cosmed'] = config[key]
                training_params[key]['auxillary_task'] = config[key]
            else:
                training_params[key] = config[key]

        train_master(training_params)


# In[ ]:





# In[ ]:





# # TODO: use training_params['regressor_names'] for looping

# In[ ]:





# In[32]:


if training_params['wandb']:
    print('sweeping for:', sweep_name)
    sweep_config = training_params['sweep_config']    
#     with wandb.init(config=config, entity='inanlab', project="[TL] stage2_cnn", reinit=True, dir=outputdir):
    sweep_id = wandb.sweep(sweep_config, entity='inanlab', project='[FL] stage4_'+training_params['sweep_name'])

#     sweep_id = wandb.sweep(sweep_config, project=sweep_name)
    wandb.agent(sweep_id, train_sweep)
    
else:
    train_master(training_params)


# In[ ]:





# In[ ]:


# if 'label_range' in training_params:
#     if training_params['label_range'] == 'label+estimated':
#         print('hji')


# In[ ]:


if training_params['wandb']:
    wandb.finish()


# In[ ]:





# # TODO: think where to place this

# In[ ]:





# In[ ]:


datetime_end = datetime.now(tz_NY)
print("end time:", datetime_end.strftime("%Y-%b-%d %H:%M:%S"))

duration = datetime_end-datetime_start
duration_in_s = duration.total_seconds()
days    = divmod(duration_in_s, 86400)        # Get days (without [0]!)
hours   = divmod(days[1], 3600)               # Use remainder of days to calc hours
minutes = divmod(hours[1], 60)                # Use remainder of hours to calc minutes
seconds = divmod(minutes[1], 1)               # Use remainder of minutes to calc seconds
print("Time between dates: %d days, %d hours, %d minutes and %d seconds" % (days[0], hours[0], minutes[0], seconds[0]))


# In[ ]:


sys.exit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def get_label_mapped(training_params):
    
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    label = dataloaders['val'].dataset.label
    # this function is written to map one vector to another (find the closest value in a dictionary vector)
    xf = np.linspace(0.0, 1.0/2.0*training_params['FS_Extracted'] , 375//2)*60
    mask = (xf>=label_range_dict['HR_DL'][0]) & (xf<=label_range_dict['HR_DL'][1])

    diff_matrix = np.abs(xf[None,:] - label.squeeze()[:,None]) # label_dim (subject data) x reference_dim (ordered)
    indices = np.argmin(diff_matrix, axis=1) # label_dim 
    print('indices', indices.shape)
    xf_repeated = np.tile(xf, (indices.shape[0], 1)).T
    print('xf_repeated', xf_repeated.shape)

    label_mapped = xf_repeated[ indices, range(xf_repeated.shape[1])]


# In[ ]:


get_label_mapped(training_params)


# In[ ]:


# x size torch.Size([2, 89])
# index_dominant torch.Size([2])
# self.xf_masked torch.Size([89])
# xf_repeated torch.Size([89, 2])


# In[ ]:





# In[ ]:


training_params['model_name']


# In[ ]:


# # def get_activation(name):
# #     activation = {}
# #     def hook(model, input, output):
# #         activation[name] = output.detach()
# #     return hook
    
# def model_hooking(model, training_params, get_activation):

#     model_name = training_params['model_name']
    
#     if model_name=='FeatureExtractor_CNN2':
#         key = list(model.feature_extractors.keys())[0]
#         model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
#         model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
#         model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
#         model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
#         model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
#         model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))
        
# #         layer_names = ['layer1', ]
        
#     elif model_name=='FeatureExtractor_CNN':
#         key = list(model.feature_extractors.keys())[0]
#         model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
#         model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
#         model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
#         model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
#         model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
#         model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))
#     if model_name=='ResNet1D':
#         pass
# #         key = list(model.feature_extractors.keys())[0]
# #         model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
# #         model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
# #         model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
# #         model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
# #         model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
# #         model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))


# In[ ]:


# training_params['model_name']


# In[ ]:





# In[ ]:





# # TODO: improve this block

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


performance_dict_train = preder(model, dataloaders['train'], training_params)
performance_dict_val = preder(model, dataloaders['val'], training_params)

unit = unit_dict[task.split('_')[0]]

for task in training_params['tasks']:
    
    
    print('evaluating task:', task)
    MAE, std_AE = get_MAE(performance_dict_train['out_dict'][task], performance_dict_train['label_dict'][task])
    print('\ttrainin: {:.2f}±{:.2f} {}'.format(MAE, std_AE, unit))

    MAE, std_AE = get_MAE(performance_dict_val['out_dict'][task], performance_dict_val['label_dict'][task])
    print('\tval: {:.2f}±{:.2f} {}'.format(MAE, std_AE, unit))


# In[ ]:





# In[ ]:





# In[ ]:





# # produce output figures
# # TODO: implement the plotting functions below

# # TODO!!!!
# 
# # for task in tasks
# #     plot_loss vs epoch (train and val)
# #     plot_MAE, RMSE, vs epoch (train and val)
# #     plot scatter plots (show PCC, BD, std, ect.) (just val)
# #     plot output vs. label (just val)

# In[ ]:





# In[ ]:


output_name = training_params['output_names'][0].split('_')[0]
unit_dict[output_name]


# In[ ]:


# kcalpmin2watt = 69.7333333

# sub_weight = 76


# In[ ]:


# performance_dict_train['label_dict']


# In[ ]:


training_params['tasks']


# In[ ]:


for task in training_params['tasks']:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=100)
    fontsize = 15
    data_min = np.min(np.r_[performance_dict_train['out_dict'][task], performance_dict_train['label_dict'][task]])
    data_max = np.max(np.r_[performance_dict_train['out_dict'][task], performance_dict_train['label_dict'][task]])
    ax1.scatter(performance_dict_train['out_dict'][task], performance_dict_train['label_dict'][task], alpha=0.3)
    ax1.set_xlim(data_min, data_max)
    ax1.set_ylim(data_min, data_max)
    ax1.plot( [data_min, data_max],[data_min, data_max], '--', color='gray', alpha=0.8)

    ax1.set_xlabel('estimated {} ({})'.format(task.split('_')[0], unit_dict[task.split('_')[0]]), fontsize=fontsize)
    ax1.set_ylabel('true {} ({})'.format(task.split('_')[0], unit_dict[task.split('_')[0]]), fontsize=fontsize)
#     ax.set_xlabel('estimated {} ({})'.format(output_name, 'W'), fontsize=fontsize)
#     ax.set_ylabel('true {} ({})'.format(output_name, 'W'), fontsize=fontsize)

    ax1.set_title('training', fontsize=fontsize)


#     fig, ax = plt.subplots(figsize=(5,5))
#     fontsize = 15
    data_min = np.min(np.r_[performance_dict_val['out_dict'][task], performance_dict_val['label_dict'][task]])
    data_max = np.max(np.r_[performance_dict_val['out_dict'][task], performance_dict_val['label_dict'][task]])
    ax2.scatter(performance_dict_val['out_dict'][task], performance_dict_val['label_dict'][task], alpha=0.3)
    ax2.set_xlim(data_min, data_max)
    ax2.set_ylim(data_min, data_max)
    ax2.plot( [data_min, data_max],[data_min, data_max], '--', color='gray', alpha=0.8)

    ax2.set_xlabel('estimated {} ({})'.format(task.split('_')[0], unit_dict[task.split('_')[0]]), fontsize=fontsize)
    ax2.set_ylabel('true {} ({})'.format(task.split('_')[0], unit_dict[task.split('_')[0]]), fontsize=fontsize)
    
    ax2.set_title('testing', fontsize=fontsize)

#     ax.set_xlabel('estimated {} ({})'.format(output_name, 'W'), fontsize=fontsize)
#     ax.set_ylabel('true {} ({})'.format(output_name, 'W'), fontsize=fontsize)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




