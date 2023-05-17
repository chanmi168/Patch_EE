#!/usr/bin/env python
# coding: utf-8

# # this notebook parse patch data
# # TODO: meet with Venu to confirm the changes I made
# # TODO: figure out the units in COSMED

# In[21]:


import numpy as np
import argparse

import os
import math
from math import sin

import pandas as pd
from sklearn.linear_model import LinearRegression


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# get_ipython().run_line_magic('matplotlib', 'inline')

i_seed = 0

import sys
sys.path.append('../') # add this line so Data and data are visible in this file
sys.path.append('../../') # add this line so Data and data are visible in this file
sys.path.append('../PatchWand/') # add this line so Data and data are visible in this file

# from PatchWand import *
from filters import *
from setting import *
from plotting_tools import *
from stage1_patch import *
from TimeStampReader import *
from stage1_cosmed import *


from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[22]:


parser = argparse.ArgumentParser(description='SpO2_estimate')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--subject_id', metavar='subject_id', help='subject_id',
                    default='sub101')
parser.add_argument('--filter_window', type=int, metavar='filter_window', help='filter_window',
                    default='5')


# checklist 3: comment first line, uncomment second line
# args = parser.parse_args(['--input_folder', '../../data/raw/', 
#                           '--output_folder', '../../data/stage1/',
#                           '--subject_id', 'sub109',
#                           '--filter_window', '1',
#                          ])
args = parser.parse_args()
print(args)


# In[ ]:





# In[23]:


inputdir = args.input_folder
outputdir = args.output_folder
subject_id = args.subject_id
filter_window = args.filter_window

if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
outputdir_sub = outputdir+subject_id+'/cosmed/'

if not os.path.exists(outputdir_sub):
    os.makedirs(outputdir_sub)


# In[ ]:





# In[24]:


if filter_window==1:
    outputdir_sub = outputdir_sub.replace('cosmed', 'cosmed_unfiltered')


# In[25]:


# if 'unfiltered' in outputdir_sub:
#     filter_window = 1
# else:
#     filter_window = 5


# In[26]:


outputdir_sub


# In[ ]:





# In[27]:


import os

for subdir, dirs, files in os.walk(inputdir):
#     print(subdir, dirs, files)
    for dir_one in dirs:
        if subject_id in dir_one:
            inputdir_sub = inputdir + dir_one + '/'


# In[28]:


inputdir_sub


# In[29]:


#     fileName = inputdir_sub + 'sub005_121321/TEST_NO_80.csv'
#     fileName = inputdir + 'sub004_121021/TEST_NO_79.csv'


# In[49]:


# subject_id = 'sub005'

for filename in os.listdir(inputdir_sub):
    if 'TEST' in  filename:
        if '.csv' in  filename:
#     print(filename)

            cosmed_filePath = inputdir_sub + filename
#     fileName = inputdir + 'sub004_121021/TEST_NO_79.csv'
    # fileName = inputdir + 'sub003_112321/TEST_NO_28.csv'

#     subject_id = 'sub'+fileName.split('sub')[1].split('_')[0]
cosmed_filePath


# In[50]:


for filename in os.listdir(inputdir_sub):
    if 'tm_backup' in  filename:
        if 'processed' in filename:
            print(filename)
            tm_filePath = inputdir_sub + filename


# In[51]:


tm_filePath


# # read timestamps, use readTimeStampData to parse data

# In[52]:


ts_dict = readTimeStampData(tm_filePath, dateBegin=None, dateEnd=None)

df_timestamps = pd.DataFrame(ts_dict, columns=['datetime', 'note'])
df_timestamps = df_timestamps.sort_values(by=['datetime'], ascending=True)


# In[ ]:





# In[ ]:





# # create the timestamps dictionary

# In[53]:


def ts2sec(df_timestamps, task_name, ith_taskname):
    """
        this function provides the name of the task and the time in sec of the labeled task
        TODO: provide documentation

    """
    datetime = df_timestamps[df_timestamps['note']==task_name].reset_index(drop=True).iloc[ith_taskname]['datetime']
    seconds = datetime.timestamp()
    
    task_renamed = task_name+' '+str(ith_taskname)
    
    return task_renamed, seconds


# In[54]:


task_ts_dict = {}

for task_name in df_timestamps['note'].unique():
    N_rows = df_timestamps[df_timestamps['note']==task_name].shape[0]

    for ith_taskname in range(N_rows):
        task_renamed, time_in_s = ts2sec(df_timestamps, task_name, ith_taskname)
        task_ts_dict[task_renamed] = time_in_s

t_start = task_ts_dict['3 Taps 0']
for key in task_ts_dict:
    task_ts_dict[key] = task_ts_dict[key] - t_start


# In[55]:


task_ts_dict


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[56]:


# df_cosmed


# In[57]:


# #         df_data = df_cosmed[df_cosmed[df_cosmed[0]=='IDS'].index[0]:] # the 0~25 rows are not timeseries data


# # get cosmed data

# In[58]:


df_cosmed = pd.read_csv(cosmed_filePath, header=None, sep='\n')
df_cosmed = df_cosmed[0].str.split(',', expand=True)
df_cosmed[:30]


# In[59]:


sub = Subject(df_cosmed, task_ts_dict, subject_id=subject_id)
sub.show_demographic()


# In[60]:


sub.task_ts_dict


# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


plt.plot(sub.df_data['task'])


# # create a list of task names

# In[31]:


# from setting
len(FS_tasks)


# # inspect label (TBD)

# In[ ]:





# In[33]:


def filter_df_cosmed(df_cosmed, outputdir=None, show_plot=False):
    cosmed_resampled = {}
    df_cosmed_filtered = df_cosmed.copy()
    
    variables_movingfile = ['HR', 'RR', 'VT', 'VE', 'VO2', 'VCO2', 'EE', 'SPO2', 'FiO2', 'FeO2', 'FiCO2', 'FeCO2', 'K5_R']
    variables_medfile = ['GpsAlt', 'AmbTemp', 'Battery']

    n_plots = len(variables_movingfile) + len(variables_medfile)

    fig, axes = plt.subplots(n_plots, 1, figsize=(30,15) ,dpi=100)
    fig.tight_layout()

    i = 0

    for cosmed_label in list(df_cosmed.columns):
        
        if cosmed_label in ['time(s)', 'task', 'Marker']:
            continue

        v_cosmed_label = df_cosmed[cosmed_label].values.copy()
        t_cosmed = df_cosmed['time(s)'].values

        if cosmed_label in label_range_dict:
            indices_mask = np.where((v_cosmed_label>=label_range_dict[cosmed_label][0]) & (v_cosmed_label<=label_range_dict[cosmed_label][1]))[0]
        else:
            indices_mask = np.arange(v_cosmed_label.shape[0])

        if cosmed_label in variables_movingfile:
#             print('moving filter')
            v_cosmed_label_smooth = get_smooth(v_cosmed_label[indices_mask], N=filter_window) # for HR and RR
        elif cosmed_label in variables_medfile:
#             print('median filter')
            v_cosmed_label_smooth = medfilt(v_cosmed_label[indices_mask], k=filter_window)
        else:
            v_cosmed_label_smooth = v_cosmed_label[indices_mask]
#             print(cosmed_label)
    #         continue

        v_cosmed_label_smooth_resampled = np.interp(t_cosmed, t_cosmed[indices_mask], v_cosmed_label_smooth)


        df_cosmed_filtered[cosmed_label] = v_cosmed_label_smooth_resampled

#         if show_plot:
#         axes[i].set_title(cosmed_label, fontsize=15)
        axes[i].set_ylabel(cosmed_label)
        axes[i].plot(t_cosmed, v_cosmed_label, 'r', alpha=0.6)
        axes[i].plot(t_cosmed[indices_mask], v_cosmed_label[indices_mask], 'blue', alpha=0.6)
        axes[i].plot(t_cosmed[indices_mask], v_cosmed_label_smooth, 'k', alpha=0.9)
        axes[i].set_xlim(t_cosmed.min(), t_cosmed.max())
        i += 1

    axes[i-1].set_xlabel('time(s)')
    fig.tight_layout()

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + 'filtering_label.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

    return df_cosmed_filtered


# In[34]:


df_cosmed = sub.df_data

df_cosmed_filtered = filter_df_cosmed(df_cosmed, outputdir=outputdir_sub, show_plot=False)
sub.df_data = df_cosmed_filtered


# # save subject data

# In[35]:


save_sub(sub, outputdir_sub+'data')


# # show subject id again

# In[36]:


sub.subject_id


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




