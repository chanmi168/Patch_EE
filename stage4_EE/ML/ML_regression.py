#!/usr/bin/env python
# coding: utf-8

# # regression ultimate code

# In[1]:


import numpy as np
import argparse

import os
import math
from math import sin

import json

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

import seaborn as sns

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

import sys
sys.path.append('../') # add this line so Data and data are visible in this file
sys.path.append('../../') # add this line so Data and data are visible in this file
sys.path.append('../../PatchWand/') # add this line so Data and data are visible in this file

# from PatchWand import *
from plotting_tools import *
from setting import *
# from models import *
# from models_CNN import *
# from unet_extension.models import *
# # from unet_extension.models_CNN import *
# from unet_extension.training_util import *
# from unet_extension.dataset_util import *
# from EE_extension.models import *
# from unet_extension.models_CNN import *
# from EE_extension.training_util import *
from VO2_extension1111.dataset_util import *
from VO2_extension1111.training_util import *
from VO2_extension1111.evaluation_util import *


from evaluate import *

from stage3_preprocess import *
# from training_util import *
# from dataset_util import *
from dataIO import *
from stage4_regression import *

from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:





# In[ ]:





# In[2]:


# m = MLPRegressor()

# m.get_params()


# In[3]:


# from datetime import datetime
# import pytz

# tz_NY = pytz.timezone('America/New_York') 
# datetime_start = datetime.now(tz_NY)
# print("start time:", datetime_start.strftime("%Y-%b-%d %H:%M:%S"))



# datetime_end = datetime.now(tz_NY)
# print("end time:", datetime_end.strftime("%Y-%b-%d %H:%M:%S"))

# duration = datetime_end-datetime_start
# duration_in_s = duration.total_seconds()
# days    = divmod(duration_in_s, 86400)        # Get days (without [0]!)
# hours   = divmod(days[1], 3600)               # Use remainder of days to calc hours
# minutes = divmod(hours[1], 60)                # Use remainder of hours to calc minutes
# seconds = divmod(minutes[1], 1)               # Use remainder of minutes to calc seconds
# print("Time between dates: %d days, %d hours, %d minutes and %d seconds" % (days[0], hours[0], minutes[0], seconds[0]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


# print(torch.version.cuda)


# In[5]:


parser = argparse.ArgumentParser(description='EE_estimate')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--training_params_file', metavar='training_params_file', help='training_params_file',
                    default='training_params_list.json')


# checklist 3: comment first line, uncomment second line
# args = parser.parse_args(['--input_folder', '../../data/stage3/', 
# args = parser.parse_args(['--input_folder', '../../data/stage3_norun/', 
# args = parser.parse_args(['--input_folder', '../../data/stage3-1_windowing/CDC_dataset/win60_overlap95_seq20_norm/', 
# args = parser.parse_args(['--input_folder', '../../../data/stage3/win60_overlap90/', 
# # args = parser.parse_args(['--input_folder', '../../data/stage3_TEST/', 
# #                           '--output_folder', '../../data/stage4/ML_regression/',
#                           '--output_folder', '../../../data/stage4/ML_regression/',
#                           '--training_params_file', 'training_params_ML.json',
# #                           '--training_params_file', 'training_params_baseline.json',
#                          ])
args = parser.parse_args()
print(args)


# In[ ]:





# In[6]:


inputdir = args.input_folder
outputdir = args.output_folder
training_params_file = args.training_params_file



# # get training params and dataloaders

# In[7]:


# dataloaders, dataset_sizes = get_loaders(inputdir, training_params)


# In[8]:


# data_loader('meta', inputdir).shape


# In[9]:


stage3_dict = data_loader('stage3_dict', inputdir).item()
# stage3_dict


# In[10]:


# training_params['list_feature']


# In[ ]:





# In[11]:


with open(training_params_file) as json_file:
    training_params_list = json.load(json_file)

for training_params in training_params_list:
    # include device in training_params
#     device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
#     training_params['device'] = device

    if 'training_mode' in training_params:
        training_mode = training_params['training_mode']
    else:
        training_params = 'subject_ind'

#     training_params['CV_config'] = {
#         'subject_id': 101,
#         'task_ids': [0,1,2,3,4,5],
#     }
    
    task_id = [0, 1, 2, 3, 4, 5]   
    # task_id = [6]
    # task_id = [1, 2]

    training_params['CV_config'] = {
        'subject_id': 113,
        'task_ids': task_id,
        # 'reject_subject_id': [101, 102, 103, 104, 105, 109]
        # 'reject_subject_id': [101, 102, 103, 105, 109, 115]
        'reject_subject_id': [101, 102, 103, 109]
        # 'reject_subject_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,112,114, 115]
    }
    
    stage3_dict = data_loader('stage3_dict', inputdir).item()
    
    training_params['sequence'] = stage3_dict['sequence']
    training_params['list_signal'] = stage3_dict['list_signal']
    training_params['list_feature'] = stage3_dict['list_feature']

    print( training_params['list_feature'] )
#           "feature_names": ["HR_patch", "weight", "height", "gender", "age", "0.00Hz", "3.91Hz", "7.81Hz", "11.72Hz", "15.62Hz", "19.53Hz", "23.44Hz", "scg_std", "scg_std_perc"],

    training_params['list_output'] = stage3_dict['list_output']
    training_params['list_meta'] = stage3_dict['list_meta']
    training_params['FS_RESAMPLE_DL'] = stage3_dict['FS_RESAMPLE_DL']
    training_params['subject_ids'] = stage3_dict['subject_ids']
    training_params['task_ids'] = stage3_dict['task_ids']
    


    
#           "feature_names": ["HR_patch", "weight", "height", "gender", "age", "0.00Hz", "3.91Hz", "7.81Hz", "11.72Hz", "15.62Hz", "19.53Hz", "23.44Hz"],

#     input_CV = '../../data/stage3/113/CV2/'
#     dataloaders, dataset_sizes = get_loaders(input_CV, training_params)
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    print('data dimensions are:', dataloaders['train'].dataset.ecg.shape)

    data_dimensions = dataloaders['train'].dataset.__getitem__(0)[0].size()
    training_params['data_dimensions'] = list(data_dimensions)
    
    # sweep_name = training_params['sweep_name'] 
    
    sweep_name = training_params['model_name']
    training_params['sweep_name'] = sweep_name

training_params = training_params_list[0]


# In[12]:


stage3_dict


# In[13]:


# outputdir = outputdir + training_params['model_name'] + '/'

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
training_params['outputdir'] = outputdir


# In[ ]:





# In[14]:


# dataloaders, dataset_sizes = get_loaders(inputdir, training_params)


# In[15]:


# plt.plot(dataloaders['train'].dataset.data[0,2,:])


# In[16]:


# plt.plot(dataloaders['train'].dataset.data.ecg[50,:])


# In[17]:


# aaa= dataloaders['train'].dataset.feature[:, training_params['feature_names'].index('scg_std')]
# ccc= dataloaders['train'].dataset.feature[:, training_params['feature_names'].index('HR_patch')]


# bbb = dataloaders['train'].dataset.label[:, training_params['output_names'].index('EErq_cosmed')]


# In[ ]:





# In[18]:


# dataloaders['train'].dataset.feature.shape


# In[19]:


# dataloaders['train'].dataset.label.shape


# In[ ]:





# In[20]:


# for i, feature_name in enumerate(training_params['feature_names']):
#     fig, ax = plt.subplots(1,1,figsize=(5,5), dpi=80)
    
#     ax.scatter(dataloaders['train'].dataset.feature[:,i], dataloaders['train'].dataset.label)
#     ax.set_xlabel(feature_name)
#     ax.set_ylabel('EE')


# In[ ]:





# In[21]:


# training_params['subject_ids']


# In[22]:


# data_loader('meta', inputdir)


# In[23]:


# training_params['output_names']


# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


# task


# In[ ]:





# In[25]:


# "feature_names": ["HR_patch", "weight", "height", "gender", "age", "0.00Hz", "3.91Hz", "7.81Hz", "11.72Hz", "15.62Hz", "19.53Hz", "23.44Hz"],
# training_params['feature_names'] 


# In[26]:


# training_params['meta_names'] 


# In[ ]:





# In[ ]:





# In[ ]:





# ## HP from Mobashir's paper
# - learning rate = 0.05, 
# - max_depth=10, 
# - subsample=0.6, 
# - colsample_bytree = 0.7, 
# - n_estimators = 100, 
# - min_child_weight = 2, 
# - gamma = 0.3.

# In[ ]:





# In[ ]:





# In[ ]:





# In[27]:


# dataloaders['val'].dataset.meta


# In[ ]:





# In[28]:


# plt.plot(dataloaders['train'].dataset.label[:,3])


# In[29]:


# data_train, feature_train, label_train, meta_train = get_samples(inputdir, 'train/', training_params)
# data_val, feature_val, label_val, meta_val = get_samples(inputdir, 'val/', training_params)

# # zero mean unit variance
# feature_mean = np.mean(feature_train, axis=0)
# feature_std = np.std(feature_train, axis=0)


# # addecd feature importance plot

# In[30]:


def plot_feature_importances(feature_names, feature_importances, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):

    fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=100)
    fontsize = 12
    ax.barh(feature_names, feature_importances)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax_no_top_right(ax)

    fig.tight_layout()
    
    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if fig_name is None:
            fig_name = 'feature_importance'
        else:
            fig_name = fig_name

        fig.savefig(outputdir + fig_name, bbox_inches='tight', transparent=False)

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})
        
    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


# In[ ]:





# In[31]:


if training_params['wandb']:
    os.environ["WANDB_DIR"] = os.path.abspath(outputdir)
    os.environ["WANDB_NOTEBOOK_NAME"] = 'ML_regression'
    wandb.login()


# In[ ]:





# In[32]:


# !wandb login --relogin


# In[33]:


# outputdir_numeric = outputdir + 'numeric_results/'
# if outputdir_numeric is not None:
#     if not os.path.exists(outputdir_numeric):
#         os.makedirs(outputdir_numeric)
        
    
# outputdir_modelout = outputdir + 'model_output/'
# if outputdir_modelout is not None:
#     if not os.path.exists(outputdir_modelout):
#         os.makedirs(outputdir_modelout)
        

# outputdir_featureimportance = outputdir + 'feature_importance/'
# if outputdir_modelout is not None:
#     if not os.path.exists(outputdir_featureimportance):
#         os.makedirs(outputdir_featureimportance)


# In[ ]:





# In[ ]:





# In[34]:


def get_outputdirs(training_params):

    outputdir = training_params['outputdir']
    i_rep = training_params['i_rep']

    outputdir_sweep = outputdir+'rep{}/{}-{}feat/'.format(i_rep, training_params['model_name'], len(training_params['feature_names']))

    outputdir_numeric = outputdir_sweep + 'numeric_results/'
    if outputdir_numeric is not None:
        if not os.path.exists(outputdir_numeric):
            os.makedirs(outputdir_numeric)

    outputdir_modelout = outputdir_sweep + 'model_output/'
    if outputdir_modelout is not None:
        if not os.path.exists(outputdir_modelout):
            os.makedirs(outputdir_modelout)

    outputdir_featureimportance = outputdir_sweep + 'feature_importance/'
    if outputdir_modelout is not None:
        if not os.path.exists(outputdir_featureimportance):
            os.makedirs(outputdir_featureimportance)

    training_params['outputdir_sweep'] = outputdir_sweep
    training_params['outputdir_numeric'] = outputdir_numeric
    training_params['outputdir_modelout'] = outputdir_modelout
    training_params['outputdir_featureimportance'] = outputdir_featureimportance

    return training_params


# In[ ]:





# In[ ]:





# # train, test, store results

# In[35]:


# MLPRegressor()


# In[36]:


# training_params['CV_config']['subject_id'] = 106
# dataloaders, dataset_sizes = get_loaders(inputdir, training_params)


# In[37]:


# df_performance_train = pd.DataFrame()
# df_performance_val = pd.DataFrame()

# df_outputlabel_train = pd.DataFrame()
# df_outputlabel_val = pd.DataFrame()


# # training_params['output_names'] = ['EE_cosmed']

# for subject_id in training_params['subject_ids']:
    
# #     if subject_id!=117:
# #         continue
#     print(subject_id)
#     training_params['CV_config']['subject_id'] = subject_id

#     try:
#         dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
#     except:
#         print('An exception occurred for subject:', subject_id)
#         continue

    
# #     i_feature = training_params['feature_names'].index('HR_patch')
# #     i_label = training_params['output_names'].index('EE_cosmed')

    
#     if training_params['model_name']=='LinearRegression':
#         model = LinearRegression()
#     elif training_params['model_name']=='XGBRegressor':
# #         model = XGBRegressor(learning_rate=0.05, max_depth=10, subsample=0.6, colsample_bytree=0.7, n_estimators=100, min_child_weight=2, gamma=0.3)
#         model = XGBRegressor(learning_rate=0.05, max_depth=50, subsample=0.6, colsample_bytree=0.7, n_estimators=100, min_child_weight=2, gamma=0.001)
    
#     feature_train = dataloaders['train'].dataset.feature
#     feature_val = dataloaders['val'].dataset.feature
    
# #     sys.exit()

#     # zero mean unit variance
# #     feature_mean = np.mean(feature_train, axis=0)
# #     feature_std = np.std(feature_train, axis=0)

# #     feature_train = (feature_train-feature_mean)/feature_std
# #     feature_val = (feature_val-feature_mean)/feature_std

    
#     label_train = dataloaders['train'].dataset.label
#     label_val = dataloaders['val'].dataset.label
    
#     model = model.fit(feature_train, label_train)
#     label_est_train = model.predict(feature_train)
#     label_est_val = model.predict(feature_val)
    
#     label_train = label_train.squeeze()
#     label_val = label_val.squeeze()
#     label_est_train = label_est_train.squeeze()
#     label_est_val = label_est_val.squeeze()
    
# #     sys.exit()
#     if 'perc' in training_params['output_names'][0]:
#         i_meta = training_params['meta_names'].index('EEavg_est')
#         meta_train = dataloaders['train'].dataset.meta[:, i_meta]
#         meta_val = dataloaders['val'].dataset.meta[:, i_meta]
    
#         label_train = label_train*meta_train
#         label_val = label_val*meta_val
#         label_est_train = label_est_train*meta_train
#         label_est_val = label_est_val*meta_val
#     elif 'weighted' in training_params['output_names'][0]:
# #         print('hi')
#         i_meta = training_params['meta_names'].index('weight')
#         meta_train = dataloaders['train'].dataset.meta[:, i_meta]
#         meta_val = dataloaders['val'].dataset.meta[:, i_meta]
    
#         label_train = label_train*meta_train
#         label_val = label_val*meta_val
#         label_est_train = label_est_train*meta_train
#         label_est_val = label_est_val*meta_val
    
# #     sys.exit()
    
# #     sys.exit()
    
#     # get performance df for training and testing dataset
#     df_performance_train = df_performance_train.append( get_df_performance(label_train, label_est_train, subject_id, task), ignore_index=True )
#     df_performance_train.to_csv(outputdir_numeric+'df_performance_train.csv', index=False)

#     df_outputlabel_train = df_outputlabel_train.append(
#         pd.DataFrame( {
#         'label_est': label_est_train,
#         'label': label_train,
#         'CV': [subject_id]*label_train.shape[0],
#         'task': [task]*label_train.shape[0],
#         }), ignore_index=True )

#     df_outputlabel_train.to_csv(outputdir_numeric+'df_outputlabel_train.csv', index=False)

#     df_performance_val = df_performance_val.append( get_df_performance(label_val, label_est_val, subject_id, task), ignore_index=True )
#     df_performance_val.to_csv(outputdir_numeric+'df_performance_val.csv', index=False)

#     df_outputlabel_val = df_outputlabel_val.append(
#         pd.DataFrame( {
#         'label_est': label_est_val,
#         'label': label_val,
#         'CV': [subject_id]*label_val.shape[0],
#         'task': [task]*label_val.shape[0]
#         }), ignore_index=True )

#     df_outputlabel_val.to_csv(outputdir_numeric+'df_outputlabel_val.csv', index=False)
    
#     if training_params['model_name']=='XGBRegressor':
#         plot_feature_importances(training_params['feature_names'], model.feature_importances_, fig_name='feature_importance_'+str(subject_id), outputdir=outputdir_featureimportance, show_plot=False)


# sys.exit()


# In[38]:


# training_params['CV_config']['subject_id'] = 101
# dataloaders, dataset_sizes = get_loaders(inputdir, training_params)


# In[39]:


# dataloaders['train'].dataset.feature


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


# plot_feature_importances


# In[41]:


# training_params['output_names']


# In[42]:


# training_params['model_name']


# In[43]:


def train_master(training_params):

    # training_params = get_regressor_names(training_params)
    training_params = get_outputdirs(training_params) # could be tricky since it changes several keys

    print(training_params['outputdir_sweep'])
    df_performance_train = pd.DataFrame()
    df_performance_val = pd.DataFrame()

    df_outputlabel_train = pd.DataFrame()
    df_outputlabel_val = pd.DataFrame()


    # training_params['output_names'] = ['EE_cosmed']

    for subject_id in training_params['subject_ids']:

        if subject_id in training_params['CV_config']['reject_subject_id']:
            continue

    #     if subject_id!=117:
    #         continue
        print(subject_id)
        training_params['CV_config']['subject_id'] = subject_id

        dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

#         try:
#             dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
#         except:
#             print('An exception occurred for subject:', subject_id)
#             continue


    #     i_feature = training_params['feature_names'].index('HR_patch')
    #     i_label = training_params['output_names'].index('EE_cosmed')


        if training_params['model_name']=='LinearRegression':
            model = LinearRegression()
        elif training_params['model_name']=='XGBRegressor':
    #         model = XGBRegressor(learning_rate=0.05, max_depth=10, subsample=0.6, colsample_bytree=0.7, n_estimators=100, min_child_weight=2, gamma=0.3)
            model = XGBRegressor(learning_rate=0.05, max_depth=50, subsample=0.6, colsample_bytree=0.7, n_estimators=100, min_child_weight=2, gamma=0.001, verbosity=0, seed=training_params['i_rep'])
        elif training_params['model_name']=='MLPRegressor':
            # model = MLPRegressor(random_state=1, max_iter=1000)
            hidden_layer_sizes =tuple()
            training_params['hidden_dim'] = len(training_params['feature_names'])
            for i in range(training_params['n_layers']):
                hidden_layer_sizes = hidden_layer_sizes + (training_params['hidden_dim'],)
            
            model = MLPRegressor(max_iter=25, hidden_layer_sizes=hidden_layer_sizes, batch_size=64, random_state=training_params['i_rep'])


        feature_train = dataloaders['train'].dataset.feature
        feature_val = dataloaders['val'].dataset.feature
        
        meta_train = dataloaders['train'].dataset.meta
        meta_val = dataloaders['val'].dataset.meta
        
        print(feature_train.shape, feature_val.shape)

    #     sys.exit()

        # zero mean unit variance
    #     feature_mean = np.mean(feature_train, axis=0)
    #     feature_std = np.std(feature_train, axis=0)

    #     feature_train = (feature_train-feature_mean)/feature_std
    #     feature_val = (feature_val-feature_mean)/feature_std


        label_train = dataloaders['train'].dataset.label
        label_val = dataloaders['val'].dataset.label

        model = model.fit(feature_train, label_train)
        label_est_train = model.predict(feature_train)
        label_est_val = model.predict(feature_val)
        
        label_train = label_train.squeeze()
        label_val = label_val.squeeze()
        label_est_train = label_est_train.squeeze()
        label_est_val = label_est_val.squeeze()
        
        
        

#         if 'perc' in training_params['output_names'][0]:
#             i_meta = training_params['meta_names'].index('EEavg_est')
#             meta_train = dataloaders['train'].dataset.meta[:, i_meta]
#             meta_val = dataloaders['val'].dataset.meta[:, i_meta]

#             label_train = label_train*meta_train
#             label_val = label_val*meta_val
#             label_est_train = label_est_train*meta_train
#             label_est_val = label_est_val*meta_val
#         elif 'weighted' in training_params['output_names'][0]:
#             i_meta = training_params['meta_names'].index('weight')
#             meta_train = dataloaders['train'].dataset.meta[:, i_meta]
#             meta_val = dataloaders['val'].dataset.meta[:, i_meta]

#             label_train = label_train*meta_train
#             label_val = label_val*meta_val
#             label_est_train = label_est_train*meta_train
#             label_est_val = label_est_val*meta_val

    #     sys.exit()
        task = training_params['output_names'][0]
        task_name = task.split('_')[0]
    #     sys.exit()

        # get performance df for training and testing dataset
        df_performance_train = df_performance_train.append( get_df_performance(label_train, label_est_train, subject_id, task), ignore_index=True )
        df_performance_train.to_csv(training_params['outputdir_numeric']+'df_performance_train.csv', index=False)

        df_outputlabel_train = df_outputlabel_train.append(
            pd.DataFrame( {
            'label_est': label_est_train,
            'label': label_train,
            'CV': [subject_id]*label_train.shape[0],
            'task': [task]*label_train.shape[0],
            'activity': meta_train[:,1],
            'weight': meta_train[:,0],
            'height': meta_train[:,1],
            'VT_cosmed': meta_train[:,2],
            }), ignore_index=True )

        df_outputlabel_train.to_csv(training_params['outputdir_numeric']+'df_outputlabel_train.csv', index=False)

        df_performance_val = df_performance_val.append( get_df_performance(label_val, label_est_val, subject_id, task), ignore_index=True )
        df_performance_val.to_csv(training_params['outputdir_numeric']+'df_performance_val.csv', index=False)

        df_outputlabel_val = df_outputlabel_val.append(
            pd.DataFrame( {
            'label_est': label_est_val,
            'label': label_val,
            'CV': [subject_id]*label_val.shape[0],
            'task': [task]*label_val.shape[0],
            'activity': meta_val[:,1],
            'weight': meta_val[:,0],
            'height': meta_val[:,1],
            'VT_cosmed': meta_val[:,2],
            }), ignore_index=True )

        df_outputlabel_val.to_csv(training_params['outputdir_numeric']+'df_outputlabel_val.csv', index=False)

        if training_params['model_name']=='XGBRegressor':
            plot_feature_importances(training_params['feature_names'], model.feature_importances_, fig_name='feature_importance_'+str(subject_id), outputdir=training_params['outputdir_featureimportance'], show_plot=False, log_wandb=training_params['wandb'])




        # plot performance training and testing dataset
#         plot_BA(df_outputlabel_train, task, fig_name='BA_train', show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])

        plot_BA(df_outputlabel_val, task_name=task_name, fig_name='BA_val', show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
        # plot_regression(df_outputlabel_val, df_performance_val, task, fig_name='regression_val', show_plot=False, outputdir=outputdir_modelout, log_wandb=training_params['wandb'])
        plot_regression(df_outputlabel_train, training_params, task_name=task_name, fig_name='regression_train', show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])
        plot_regression(df_outputlabel_val,  training_params, task_name=task_name, fig_name='regression_val', show_plot=False, outputdir=training_params['outputdir_modelout'], log_wandb=training_params['wandb'])

        
        
        

    plot_output(df_outputlabel_val, task_name=task_name, fig_name = 'outputINtime_val_',  show_plot=False, outputdir=training_params['outputdir_modelout'])

    # plot_regression_all_agg(df_outputlabel_val, df_performance_val, fig_name='LinearR_agg_val', outputdir=outputdir_modelout, show_plot=False, log_wandb=training_params['wandb'])
    # plot_regression_all_agg(df_outputlabel_train, df_performance_val, fig_name='LinearR_agg_train', outputdir=outputdir_modelout, show_plot=False, log_wandb=training_params['wandb'])


    plot_regression_all_agg(df_outputlabel_val, training_params, task_name=task_name, fig_name='LinearR_agg_val', outputdir=training_params['outputdir_modelout'], show_plot=False, log_wandb=training_params['wandb'])
    plot_regression_all_agg(df_outputlabel_train, training_params, task_name=task_name, fig_name='LinearR_agg_train', outputdir=training_params['outputdir_modelout'], show_plot=False, log_wandb=training_params['wandb'])


    # log metrices on wnadb
    if training_params['wandb']==True:

        # W&B
        label = df_outputlabel_val['label'].values
        label_est = df_outputlabel_val['label_est'].values

        PCC = get_PCC(label, label_est)
        Rsquared = get_CoeffDeterm(label, label_est)
        MAE, _ = get_MAE(label, label_est)
        RMSE = get_RMSE(label, label_est)
        MAPE, _ = get_MAPE(label, label_est)
        
        print(MAE)

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





# In[44]:


import traceback

def train_sweep(config=None):   
#     with wandb.init(config=config, entity='inanlab', project="[TL] stage2_cnn", reinit=True, dir=outputdir):
    with wandb.init(config=config, reinit=True, dir=outputdir):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        
        print(config)
                
        for key in config.keys():
            training_params[key] = config[key]
            
            
        try: 
            train_master(training_params)
        except Exception:
            print(traceback.print_exc(), file=sys.stderr)


# In[45]:


# training_params


# In[46]:


if training_params['wandb']:
    print('sweeping for:', sweep_name)
    sweep_config = training_params['sweep_config']    
#     with wandb.init(config=config, entity='inanlab', project="[TL] stage2_cnn", reinit=True, dir=outputdir):
    sweep_id = wandb.sweep(sweep_config, entity='inanlab', project='[VO2] stage4_'+training_params['sweep_name'])

#     sweep_id = wandb.sweep(sweep_config, project=sweep_name)
    wandb.agent(sweep_id, train_sweep)
    
else:
    train_master(training_params)


# In[47]:


sys.exit()


# In[ ]:


# plt.plot(label_est_val)


# In[ ]:


# plt.plot(label_est_train)
# plt.plot(label_train)
# # meta_train
# # plt.plot(label_train)


# In[ ]:


# plt.plot(dataloaders['train'].dataset.label)


# In[ ]:


# plt.plot(dataloaders['val'].dataset.label)


# In[ ]:


# plt.plot(label_est_train)
# plt.show()


# In[ ]:


# model = MLPRegressor(random_state=1, max_iter=3)
# print(model)


# In[ ]:





# In[ ]:





# In[ ]:


# feature_train = dataloaders['train'].dataset.feature


# feature_mean = np.mean(feature_train, axis=0)
# feature_std = np.std(feature_train, axis=0)
# # plt.plot(feature_std)
# plt.plot(feature_mean)


# In[ ]:


# plt.plot( dataloaders['train'].dataset.label[:200])


# In[ ]:


# plt.plot(dataloaders['val'].dataset.label)


# In[ ]:


# meta_train.shape, label_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# # plot performance training and testing dataset
# plot_regression(df_outputlabel_train[task], df_performance_train[task], task, fig_name='regression_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')
# plot_BA(df_outputlabel_train[task], task, fig_name='BA_train_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

# plot_regression(df_outputlabel_val[task], df_performance_val[task], task, fig_name='regression_val_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')
# plot_BA(df_outputlabel_val[task], task, fig_name='BA_val_{}'.format(task), show_plot=False, outputdir=outputdir+'model_output/')

# plot_output(df_outputlabel_train[task], fig_name = 'outputINtime_train_', show_plot=True, outputdir=outputdir+'model_output/')
# plot_output(df_outputlabel_val[task], fig_name = 'outputINtime_val_',  show_plot=True, outputdir=outputdir+'model_output/')


# In[ ]:


# # plot performance training and testing dataset
# # plot_regression(df_outputlabel_train, df_performance_train, task, fig_name='regression_train', show_plot=False, outputdir=outputdir)
# plot_BA(df_outputlabel_train, task, fig_name='BA_train', show_plot=False, outputdir=outputdir_modelout)
# # plot_output(df_outputlabel_train, task, fig_name = 'outputINtime_train_', show_plot=False, outputdir=outputdir_modelout)

# # plot_regression(df_outputlabel_val, df_performance_val, task, fig_name='regression_val', show_plot=False, outputdir=outputdir)
# plot_BA(df_outputlabel_val, task, fig_name='BA_val', show_plot=False, outputdir=outputdir_modelout)
# plot_output(df_outputlabel_val, task, fig_name = 'outputINtime_val_',  show_plot=False, outputdir=outputdir_modelout)

# plot_regression_all_agg(df_outputlabel_val, df_performance_val, fig_name='LinearR_agg_val', outputdir=outputdir_modelout, show_plot=False, log_wandb=training_params['wandb'])
# plot_regression_all_agg(df_outputlabel_train, df_performance_val, fig_name='LinearR_agg_train', outputdir=outputdir_modelout, show_plot=False, log_wandb=training_params['wandb'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


task_name = task.split('_')[0]

label_range = [my_floor(df_outputlabel_val['label'].values.min()), my_ceil(df_outputlabel_val['label'].values.max())]

N_sub = len(df_outputlabel_val['CV'].unique())
N_samples = df_outputlabel_val.shape[0]
t_dur = N_samples*3/60


PCC = get_PCC(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
Rsquared = get_CoeffDeterm(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
MAE, MAE_std = get_MAE(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
RMSE = get_RMSE(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)
MAPE, MAPE_std = get_MAPE(df_outputlabel_val['label'].values, df_outputlabel_val['label_est'].values)

title_str = '{} range: {:.1f}-{:.1f} {}'.format(task.split('_')[0], label_range[0], label_range[1], unit_dict[task_name])
textstr = 'RMSE={:.2f} {}\nMAE={:.2f} {}\nMAPE={:.2f} {}\nPCC={:.2f}\nR2={:.2f}\nN_sub={}\nN_samples={}\nduration={:.2f} min'.format(
    RMSE, unit_dict[task_name], MAE, unit_dict[task_name],MAPE*100, '%',
    PCC, Rsquared,
    N_sub, N_samples, t_dur)


# In[ ]:


print(textstr)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data_saver(training_params, 'training_params', outputdir)


# In[ ]:


sys.exit()


# In[ ]:


#     for task in training_params['tasks']:
#         label_est_val = CV_dict['performance_dict_val']['out_dict'][task]
#         label_val = CV_dict['performance_dict_val']['label_dict'][task]

#         label_est_train = CV_dict['performance_dict_train']['out_dict'][task]
#         label_train = CV_dict['performance_dict_train']['label_dict'][task]
        
#         # get performance df for training and testing dataset
#         df_performance_train = df_performance_train.append( get_df_performance(label_train, label_est_train, subject_id, task), ignore_index=True )
#         df_performance_train.to_csv(outputdir+'df_performance_train.csv', index=False)

#         df_outputlabel_train = df_outputlabel_train.append(
#             pd.DataFrame( {
#             'label_est': label_est_train,
#             'label': label_train,
#             'CV': [subject_id]*label_train.shape[0],
#             'task': [task]*label_train.shape[0]
#             }), ignore_index=True )
        
#         df_outputlabel_train.to_csv(outputdir+'df_outputlabel_train.csv', index=False)

#         df_performance_val = df_performance_val.append( get_df_performance(label_val, label_est_val, subject_id, task), ignore_index=True )
#         df_performance_val.to_csv(outputdir+'df_performance_val.csv', index=False)
        
#         df_outputlabel_val = df_outputlabel_val.append(
#             pd.DataFrame( {
#             'label_est': label_est_val,
#             'label': label_val,
#             'CV': [subject_id]*label_val.shape[0],
#             'task': [task]*label_val.shape[0]
#             }), ignore_index=True )
        
#         df_outputlabel_val.to_csv(outputdir+'df_outputlabel_val.csv', index=False)

#         # plot performance training and testing dataset
#         plot_regression(df_outputlabel_train, df_performance_train, task, fig_name='regression_train', show_plot=False, outputdir=outputdir)
#         plot_BA(df_outputlabel_train, task, fig_name='BA_train', show_plot=False, outputdir=outputdir)
        
#         plot_regression(df_outputlabel_val, df_performance_val, task, fig_name='regression_val', show_plot=False, outputdir=outputdir)
#         plot_BA(df_outputlabel_val, task, fig_name='BA_val', show_plot=False, outputdir=outputdir)

# #     sys.exit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# plot_regression(df_outputlabel_val, df_performance_val, task, show_plot=True, outputdir=outputdir)


# In[ ]:


# plot_BA(df_outputlabel_val, task, show_plot=True, outputdir=outputdir)


# In[ ]:





# In[ ]:




