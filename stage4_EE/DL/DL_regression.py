#!/usr/bin/env python
# coding: utf-8

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
from setting import *
from models import *
# from models_CNN import *
# from models_CNN2 import *
from models_resnet import *
from evaluate import *

from stage3_preprocess import *
from stage4_regression import *
from training_util import *
from dataset_util import *
from dataIO import *

from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


# loss = nn.MSELoss()
# input = torch.ones(3, 5, requires_grad=True)
# target = torch.ones(3, 5)*10
# output = loss(input, target)
# output


# In[3]:


# aaa = np.random.rand(10,2,4)
# aaa.mean(axis=-1).shape


# In[4]:


print(torch.version.cuda)


# In[ ]:





# In[10]:


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
# args = parser.parse_args(['--input_folder', '../../data/stage3_norun/', 
#                           '--output_folder', '../../data/stage4_TEST/',
# #                           '--training_params_file', 'training_params_baseline.json',
#                           '--training_params_file', 'training_params_dummy.json',
#                          ])
args = parser.parse_args()
print(args)


# # start timing

# In[11]:



tz_NY = pytz.timezone('America/New_York') 
datetime_start = datetime.now(tz_NY)
print("start time:", datetime_start.strftime("%Y-%b-%d %H:%M:%S"))


# In[ ]:





# In[ ]:





# In[12]:


inputdir = args.input_folder
outputdir = args.output_folder
training_params_file = args.training_params_file

if not os.path.exists(outputdir):
    os.makedirs(outputdir)


# # get training params and dataloaders

# In[13]:


torch.cuda.is_available()


# In[ ]:





# In[14]:


with open(training_params_file) as json_file:
    training_params_list = json.load(json_file)

for training_params in [training_params_list[0]]:
    # include device in training_params
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
    training_params['device'] = device
    
    if 'training_mode' in training_params:
        training_mode = training_params['training_mode']
    else:
        training_params = 'subject_ind'

    training_params['CV_config'] = {
        'subject_id': 113,
        'task_id': 5,
    }
    stage3_dict = data_loader('stage3_dict', inputdir).item()
    training_params['list_signal'] = stage3_dict['list_signal']
    training_params['list_feature'] = stage3_dict['list_feature']
    training_params['list_output'] = stage3_dict['list_output']
    training_params['list_meta'] = stage3_dict['list_meta']
    training_params['FS_RESAMPLE_DL'] = stage3_dict['FS_RESAMPLE_DL']
    training_params['subject_ids'] = stage3_dict['subject_ids']
    training_params['task_ids'] = stage3_dict['task_ids']
    training_params['sequence'] = stage3_dict['sequence']
    
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    print('data dimensions are:', dataloaders['train'].dataset.data.shape)

    data_dimensions = dataloaders['train'].dataset.__getitem__(0)[0].size()
    training_params['data_dimensions'] = list(data_dimensions)
    
    sweep_name = training_params['sweep_name'] 
    
    if training_params['model_name'] == 'FeatureExtractor_CNN':
        training_params['featrue_extractor'] = FeatureExtractor_CNN
    elif training_params['model_name'] == 'ResNet1D':
        training_params['featrue_extractor'] = ResNet1D
    elif training_params['model_name'] == 'FeatureExtractor_CNN2':
        training_params['featrue_extractor'] = FeatureExtractor_CNN2

    training_params['ordered_subject_ids'] = np.asarray(training_params['ordered_subject_ids'])

    
    training_params['inputdir'] = inputdir
    training_params['outputdir'] = outputdir
    
    training_params = get_regressor_names(training_params)

    if 'LSTM' in training_params['model_name']:
        training_params = change_output_dim(training_params)

    
    del dataloaders
# training_params = training_params_list[0]


# In[ ]:





# In[ ]:





# In[ ]:


# data_train, feature_train, label_train, meta_train = get_samples(inputdir, 'train/', training_params)
# data_train.shape, feature_train.shape, label_train.shape, meta_train.shape


# In[ ]:


outputdir = outputdir+training_params['model_name']+'/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    


# # define trainer, evaler, preder functions

# In[ ]:


trainer = train_resnet
evaler = eval_resnet
preder = pred_resnet


# In[ ]:





# In[ ]:





# # some random param
# ## kernel_size = 5 ~ 16*100/300 (16 for 300 Hz sampling rate, 5 for 100 sampling rate)
# 

# In[ ]:





# In[ ]:


# training_params['downsample_gap']

# # n_block = training_params['n_block_macro'][0] * training_params['downsample_gap'][0]
# # downsample_gap = training_params['downsample_gap'][0]
# # base_filters = training_params['base_filters'][0]
# # use_sc =  training_params['use_sc']
# # increasefilter_gap = downsample_gap * 2
# # kernel_size = training_params['kernel_size']
# # groups = training_params['base_filters'][0]


# In[ ]:





# In[ ]:





# In[ ]:


if training_params['model_name'] == 'ResNet1D':
# reference: https://github.com/hsd1503/resnet1d/blob/master/test_physionet.py
    training_params['base_filters'] = training_params['channel_n'] # [64] 
    training_params['in_channels'] = training_params['data_dimensions']
#     training_params['increasefilter_gap'] = [training_params['downsample_gap'][0] * 2]
    training_params['n_block'] = training_params['n_block_macro'] * training_params['downsample_gap']
    training_params['increasefilter_gap'] = training_params['downsample_gap']
    


# In[ ]:


# # kernel_size = 10 # ~ 16*100/300 (16 for 300 Hz sampling rate, 5 for 100 sampling rate)

# training_params['stride'] = 2
# # stride = 2
# # increasefilter_gap = 12
# # base_filters = 32
# training_params['n_classes'] = 1
# # n_classes = 1 # regression
# training_params['groups'] = 32
# # groups = 32


# In[ ]:


training_params['data_dimensions']


# In[ ]:





# # test the model

# # TODO: need to fix summary()

# In[ ]:


# model = resp_multiverse(training_params=training_params)
# model


# In[ ]:





# In[ ]:


# # model
# # print([n for n, _ in model.named_children()])
# try:
# #     model.regressors['EErq_cosmed'].ch_pooling.register_forward_hook(get_activation('aaa'+'_layer_pooling'))
#     print(model.feature_extractors['ppg_ir_1_cardiac'].final_ch_pooling)
# #     model.feature_extractors['ppg_ir_1_cardiac'].final_ch_pooling.register_forward_hook(get_activation('j'+'_layer_pool'))

# except:
#     print('hi')


# In[ ]:





# In[ ]:


def ma_test(training_params):
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


# In[ ]:


def test_model(training_params):

    print('using model ', training_params['model_name'])

    model = resp_multiverse(training_params=training_params)
    summary(model, input_size=[tuple(training_params['data_dimensions']), (model.N_features,1)], device='cpu')
    print(model)
    del model


debug_model = True
if debug_model==True:
    if 'LSTM' not in training_params['model_name']:
        test_model(training_params)
    else:
        ma_test(training_params)


# In[ ]:


training_params


# In[ ]:





# In[ ]:





# In[ ]:


# dataloaders


# In[ ]:





# In[ ]:


# dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

# # data = dataloaders['val'].dataset.data[:5,:,:]
# # data = torch.from_numpy(data)

# # feature = dataloaders['val'].dataset.feature[:5,:]
# # feature = torch.from_numpy(feature)

# label = dataloaders['val'].dataset.label


# In[ ]:


# label[:,0,:].mean(axis=-1).shape


# In[ ]:





# In[ ]:





# In[ ]:


# /6000//4//4//4//4//4


# In[ ]:





# In[ ]:


# out['EErq_cosmedperc'].size(), label.size()


# In[ ]:


# a = torch.rand(10,1)
# aaa = []
# aaa.append(a)
# aaa
# # print(a.transpose(0,3).size())
# # print(a.permute(3,2,1,0).size())


# In[ ]:


# torch.concat(aaa, -1).size()


# In[ ]:


# pytorch_total_params/(10**6)


# In[ ]:


# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[ ]:


# del model


# In[ ]:





# In[ ]:





# # define training, validating, and evaluating funcitons 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## train, val, eval model

# In[ ]:


if training_params['wandb']:
    wandb.login()
    os.environ["WANDB_DIR"] = os.path.abspath(outputdir)
    os.environ["WANDB_NOTEBOOK_NAME"] = 'DL_regression'


# In[ ]:





# In[ ]:


outputdir_numeric = outputdir + 'numeric_results/'
if outputdir_numeric is not None:
    if not os.path.exists(outputdir_numeric):
        os.makedirs(outputdir_numeric)
        
    
outputdir_modelout = outputdir + 'model_output/'
if outputdir_modelout is not None:
    if not os.path.exists(outputdir_modelout):
        os.makedirs(outputdir_modelout)
        
        


# In[ ]:





# In[ ]:


debug_auxillary = False

def train_master(training_params):
    
    training_params = get_regressor_names(training_params)

    pprint.pprint(training_params)
    
    df_performance_train = {}
    df_performance_val = {}

    df_outputlabel_train = {}
    df_outputlabel_val = {}

#     for task in training_params['tasks']:
    for task in training_params['regressor_names']:

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


    # for subject_id in training_params['subject_ids']:
    for i_CV, subject_id in enumerate(ordered_subject_ids):
        
        if 'CV_max' in training_params:
            if i_CV >= training_params['CV_max']:
                continue

        training_params['CV_config']['subject_id'] = subject_id

        device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
        print('using device', device)


        print('using model ', training_params['model_name'])

#         training_params = get_regressor_names(training_params)
        model = resp_multiverse(training_params=training_params)
#         print(model)
        
        model = model.to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=0.01)

        criterion = MultiTaskLoss(training_params)
        
        training_params['criterion'] = criterion
        training_params['optimizer'] = optimizer
        training_params['inputdir'] = inputdir
        

        
#         print( training_params['regressor_names'])
        CV_dict = train_model(model, training_params, trainer, evaler, preder)

#         print(CV_dict['performance_dict_val']['out_dict'])
#         sys.exit()

        plot_losses(CV_dict, outputdir=outputdir, show_plot=False)

        

#         TODO: fix this code
#         print(training_params['regressor_names'])
        
#         for task in CV_dict['performance_dict_train']['out_dict'].keys():
        for task in training_params['regressor_names']:
        
#             print(task, df_performance_train)

#         for task in training_params['tasks']:

#             print( CV_dict['performance_dict_val']['out_dict'].keys(), CV_dict['performance_dict_val']['label_dict'].keys(), task)
            label_est_val = CV_dict['performance_dict_val']['out_dict'][task]
            label_val = CV_dict['performance_dict_val']['label_dict'][task]

            label_est_train = CV_dict['performance_dict_train']['out_dict'][task]
            label_train = CV_dict['performance_dict_train']['label_dict'][task]
            
            
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

            df_performance_train[task].to_csv(outputdir_numeric + 'df_performance_train_{}.csv'.format(task), index=False)

            df_outputlabel_train[task] = df_outputlabel_train[task].append(
                pd.DataFrame( {
                'label_est': label_est_train,
                'label': label_train,
                'CV': [subject_id]*label_train.shape[0],
                'task': [task]*label_train.shape[0]
                }), ignore_index=True )

            df_outputlabel_train[task].to_csv(outputdir_numeric + 'df_outputlabel_train_{}.csv'.format(task), index=False)

            df_performance_val[task] = df_performance_val[task].append( get_df_performance(label_val, label_est_val, subject_id, task), ignore_index=True )
            df_performance_val[task].to_csv(outputdir_numeric + 'df_performance_val_{}.csv'.format(task), index=False)

            df_outputlabel_val[task] = df_outputlabel_val[task].append(
                pd.DataFrame( {
                'label_est': label_est_val,
                'label': label_val,
                'CV': [subject_id]*label_val.shape[0],
                'task': [task]*label_val.shape[0]
                }), ignore_index=True )

            df_outputlabel_val[task].to_csv(outputdir_numeric + 'df_outputlabel_val_{}.csv'.format(task), index=False)

            # plot performance training and testing dataset
            if (main_task not in task) and (debug_auxillary==False):
                continue
            
#             plot_regression(df_outputlabel_train[task], df_performance_train[task], task, fig_name='regression_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')
#             plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

            plot_regression(df_outputlabel_val[task], df_performance_val[task], task, fig_name='regression_val_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
#             plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

#             plot_output(df_outputlabel_train[task], task, fig_name = 'outputINtime_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout)
        
        check_featuremap(model, training_params, mode='worst', fig_name = 'DL_activation_{}_'.format(subject_id), outputdir=outputdir+'activation_layers_worst/{}/'.format(subject_id), show_plot=False)
        check_featuremap(model, training_params, mode='best', fig_name = 'DL_activation_{}_'.format(subject_id), outputdir=outputdir+'activation_layers_best/{}/'.format(subject_id), show_plot=False)
        
        del model
        torch.cuda.empty_cache()


    for task in training_params['regressor_names']:
        if task!=main_task:
            continue
        plot_regression_all_agg(df_outputlabel_train[task], df_performance_train[task], fig_name='LinearR_agg_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
        plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])

        plot_regression_all_agg(df_outputlabel_val[task], df_performance_val[task], fig_name='LinearR_agg_val_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
        plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])

        plot_output(df_outputlabel_val[task], task, fig_name = 'outputINtime_val_{}'.format(task),  show_plot=False, outputdir=outputdir_modelout)

#     plot_BA(df_outputlabel_val[main_task], main_task, fig_name='BA_val_{}'.format(main_task), show_plot=False, outputdir=outputdir+'model_output/', log_wandb=training_params['wandb'])
#     plot_regression_all_agg(df_outputlabel_val[main_task], df_performance_val[main_task], outputdir=outputdir+'model_output/', show_plot=False, log_wandb=training_params['wandb'])

    # log metrices on wnadb
    if training_params['wandb']==True:
        # W&B
        label = df_outputlabel_val[main_task]['label'].values
        label_est = df_outputlabel_val[main_task]['label_est'].values
#         print(label.shape, label)
#         print(label_est.shape, label_est)
    
        PCC = get_PCC(label, label_est)
        Rsquared = get_CoeffDeterm(label, label_est)
        MAE, _ = get_MAE(label, label_est)
        RMSE = get_RMSE(label, label_est)
        MAPE, _ = get_MAPE(label, label_est)

        wandb.log(
            {
                'val_MAE': MAE,
                'val_RMSE': RMSE,
                'val_MAPE': MAPE,
                'val_PCC': PCC,
                'val_Rsquared': Rsquared,
            })


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


def train_sweep(config=None):

#     with wandb.init(config=config, entity='inanlab', project="[TL] stage2_cnn", reinit=True, dir=outputdir):
    with wandb.init(config=config, reinit=True, dir=outputdir):

        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        print(config)
        
        # init the model
        
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





# In[ ]:


if training_params['wandb']:
    print('sweeping for:', sweep_name)
    sweep_config = training_params['sweep_config']    
#     with wandb.init(config=config, entity='inanlab', project="[TL] stage2_cnn", reinit=True, dir=outputdir):
    sweep_id = wandb.sweep(sweep_config, entity='inanlab', project='[EE] stage4_'+training_params['sweep_name'])

#     sweep_id = wandb.sweep(sweep_config, project=sweep_name)
    wandb.agent(sweep_id, train_sweep)
    
else:
    train_master(training_params)


# In[ ]:


# aaa = np.ones((5,1))
# aaa = aaa[:,:,None]
# aaa[:,:, 0]


# In[ ]:





# In[ ]:


# PCC = get_PCC(np.ones(5), np.ones(5))

# print('{:.2f}'.format(PCC))
# # print('hi')


# In[ ]:


wandb.finish()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# end timing


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




