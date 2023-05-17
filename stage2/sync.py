#!/usr/bin/env python
# coding: utf-8

# # this notebook parse patch data
# # TODO: meet with Venu to confirm the changes I made
# # TODO: figure out the units in COSMED

# In[1]:


import numpy as np
import argparse
import numba as nb
from numpy_ext import rolling_apply

import json
import os
import math
from math import sin

import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline

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
from PatchParser import *
from stage3_inspection import *


from importlib import reload
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


parser = argparse.ArgumentParser(description='SpO2_estimate')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--subject_id', metavar='subject_id', help='subject_id',
                    default='101')
parser.add_argument('--cosmed_unfiltered', metavar='cosmed_unfiltered', help='cosmed_unfiltered',
                    default='False')
parser.add_argument('--saving_format', metavar='saving_format', help='saving_format',
                    default='feather')



# checklist 3: comment first line, uncomment second line
# args = parser.parse_args(['--input_folder', '../../data/stage1/', 
#                           '--output_folder', '../../data/stage2/',
#                           '--subject_id', 'sub113',
#                           '--cosmed_unfiltered', 'False',
#                           '--saving_format', 'feather',
#                          ])
args = parser.parse_args()
print(args)


# In[ ]:





# In[3]:


# args.cosmed_unfiltered


# In[4]:


inputdir = args.input_folder
outputdir = args.output_folder
subject_id = args.subject_id
saving_format = args.saving_format

if args.cosmed_unfiltered == 'True':
     cosmed_unfiltered = True
elif args.cosmed_unfiltered == 'False':
     cosmed_unfiltered = False
        
        
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
outputdir_sub = outputdir+subject_id+'/'

if cosmed_unfiltered:
    outputdir_sub = outputdir_sub+'cosmed_unfiltered/'

if not os.path.exists(outputdir_sub):
    os.makedirs(outputdir_sub)


# In[ ]:





# In[ ]:





# # get cosmed data

# In[5]:


outputdir_sub


# In[6]:


if cosmed_unfiltered:
    obj_cosmed = load_sub(inputdir+subject_id+'/cosmed_unfiltered/data')
else:
    obj_cosmed = load_sub(inputdir+subject_id+'/cosmed/data')


# In[7]:


# obj_cosmed.task_ts_dict


# In[8]:


with open('subject_patch_dict.json') as json_file:
    subject_patch_dict = json.load(json_file)[0]


# In[9]:


# subject_patch_dict = {}
# for i in range(0,26):
    
#     subject_patch_dict['sub{}'.format(i+101)] = 'sternum'
    
#     if i==5:
#          subject_patch_dict['sub{}'.format(i+101)] = 'clavicle'
#     if i==8:
#          subject_patch_dict['sub{}'.format(i+101)] = 'ribcage'
#     if i==9:
#          subject_patch_dict['sub{}'.format(i+101)] = 'ribcage'
#     if i==17:
#          subject_patch_dict['sub{}'.format(i+101)] = 'ribcage' # or clavicle


# In[10]:


subject_patch_dict


# # get sternum patch data
# # if subject_patch_dict[subject_id]!='sternum', will transplant the ECG from other location. Other signals will remain chest-based
# ## these subjects include 105, 108, 109, 117

# In[ ]:





# In[11]:


if subject_patch_dict[subject_id]!='sternum':
    print('transplanting for {}...'.format(subject_id))
    df_patch_ecg = pd.read_feather(inputdir+subject_id + '/patch/df_patch_{}.feather'.format(subject_patch_dict[subject_id]))

    df_patch = pd.read_feather(inputdir+subject_id + '/patch/df_patch_{}.feather'.format('sternum'))

    TMAX = min([df_patch_ecg['time'].max(), df_patch['time'].max()])
    TMIN = max([df_patch_ecg['time'].min(), df_patch['time'].min()])

    df_patch = df_patch[(df_patch['time']>=TMIN) & (df_patch['time']<=TMAX)]
    df_patch_ecg = df_patch_ecg[(df_patch_ecg['time']>=TMIN) & (df_patch_ecg['time']<=TMAX)]
    
#     ecg_prosthetic = df_patch_ecg['ECG'].values
    ecg_prosthetic = np.interp(df_patch['time'].values, df_patch_ecg['time'].values, df_patch_ecg['ECG'].values)
    df_patch['ECG'] = ecg_prosthetic
    
    del df_patch_ecg
    del ecg_prosthetic
    
else:
    df_patch = pd.read_feather(inputdir+subject_id + '/patch/df_patch_{}.feather'.format(subject_patch_dict[subject_id]))


# In[ ]:





# In[ ]:





# In[12]:


# time_interp = np.arange(my_ceil(-1215.51312, decimal=-3)*FS_RESAMPLE, my_floor(5883.236, decimal=-3)*FS_RESAMPLE+1)/FS_RESAMPLE

# time_interp


# In[ ]:





# In[ ]:





# # COSMED label correction (manually determine how much the delay is)

# ## currently, we only have to do it for sub101 (sync between the cosmed and the patch has delay since the emergence of cosemd marker is late)

# In[13]:


if subject_id =='sub101':
    cosmed_mark_delay_patch_taps_by = 13
else:
    cosmed_mark_delay_patch_taps_by = 0
    
if cosmed_mark_delay_patch_taps_by!= 0:
    df_patch['time'] =  df_patch['time']-cosmed_mark_delay_patch_taps_by


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # inspect the patch data, 50s before and after 1st 6MWT

# In[14]:


def inspect_patch2(df_patch, subject_id, title_str, VIS_START = 200, VIS_END = 230, outputdir=None, show_plot=False):

    t_arr = df_patch['time']
    t_mask = (t_arr>=VIS_START) & (t_arr<VIS_END)
    
    df_patch = df_patch[t_mask]
    t_arr = df_patch['time'].values

    
#     for key in data_dict:
#         if key != 'subject_id':
#             data_dict[key] = data_dict[t_mask]
    
    t_dur = df_patch['time'].values[-1] - df_patch['time'].values[0]

    print('[{}]: {} sec'.format(subject_id, t_dur))

#     t_arr =  df_patch['time']

    # fig = plt.figure(figsize=(3*t_dur/80, 3*10), dpi=50, facecolor='white')
    fig = plt.figure(figsize=(20, 10), dpi=80, facecolor='white')

    fontsize = 15
    scale_factor = 8
    alpha = 0.8

    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(t_arr, df_patch['ECG'].values, color=color_dict[sync_color_dict['ECG']], alpha=0.5 ,zorder=1)

    ax1.set_xlim(t_arr[0],t_arr[-1])
    ax1.set_title(title_str + 'ECG', fontsize=fontsize)
    ax1.set_ylabel('a.u.', fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)


    ax2 = fig.add_subplot(4,1,2)
    ax2.plot(t_arr, df_patch['accelX'].values, color=color_dict[sync_color_dict['accelX']], alpha=alpha ,zorder=1, label='accelX')
    ax2.plot(t_arr, df_patch['accelY'].values, color=color_dict[sync_color_dict['accelY']], alpha=alpha ,zorder=1, label='accelY')
    ax2.plot(t_arr, df_patch['accelZ'].values, color=color_dict[sync_color_dict['accelZ']], alpha=alpha ,zorder=1, label='accelZ')
    ax2.set_xlim(t_arr[0],t_arr[-1])
    ax2.set_title(title_str + 'ACC', fontsize=fontsize)
    ax2.set_ylabel('a.u.', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    ax2.legend(loc='upper left', fontsize=fontsize, frameon=True)

    ax3 = fig.add_subplot(4,1,3)
    ax3.plot(t_arr, df_patch['ppg_g_1'].values, color=color_dict[sync_color_dict['ppg_g_1']], alpha=alpha ,zorder=1, label='ppg_g_1')
    ax3.plot(t_arr, df_patch['ppg_r_1'].values, color=color_dict[sync_color_dict['ppg_r_1']], alpha=alpha ,zorder=1, label='ppg_r_1')
    ax3.plot(t_arr, df_patch['ppg_ir_1'].values, color=color_dict[sync_color_dict['ppg_ir_1']], alpha=alpha ,zorder=1, label='ppg_ir_1')
    ax3.set_xlim(t_arr[0],t_arr[-1])
    ax3.set_title(title_str + 'PPG arr 1', fontsize=fontsize)
    ax3.set_ylabel('ppg intensity (uA)', fontsize=fontsize)
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.tick_params(axis='y', labelsize=fontsize)
    ax3.legend(loc='upper left', fontsize=fontsize, frameon=True)

    ax4 = fig.add_subplot(4,1,4)
    ax4.plot(t_arr, df_patch['ppg_g_2'].values, color=color_dict[sync_color_dict['ppg_g_2']], alpha=alpha ,zorder=1, label='ppg_g_2')
    ax4.plot(t_arr, df_patch['ppg_r_2'].values, color=color_dict[sync_color_dict['ppg_r_2']], alpha=alpha ,zorder=1, label='ppg_r_2')
    ax4.plot(t_arr, df_patch['ppg_ir_2'].values, color=color_dict[sync_color_dict['ppg_ir_2']], alpha=alpha ,zorder=1, label='ppg_ir_2')
    ax4.set_xlim(t_arr[0],t_arr[-1])
    ax4.set_title(title_str + 'PPG arr 2', fontsize=fontsize)
    ax4.set_ylabel('ppg intensity (uA)', fontsize=fontsize)
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.tick_params(axis='y', labelsize=fontsize)
    ax4.legend(loc='upper left', fontsize=fontsize, frameon=True)


    ax4.set_xlabel('time (sec)', fontsize=fontsize)

    fig.tight_layout()
    
#     fig_name = '{}_signl_{}'.format(title_str,subject_id)
    fig_name = '{}_signl_{}'.format(title_str,subject_id)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


# In[15]:


print('3 taps starts at: {}s'.format ( obj_cosmed.task_ts_dict['3 Taps 0'] ))


# In[ ]:





# In[16]:


inpsect_sigs = False
if inpsect_sigs:

#     VIS_START = obj_cosmed.task_ts_dict['Start 6MWT 0']-50
#     VIS_END = obj_cosmed.task_ts_dict['End 6MWT 0']+50
    VIS_START = obj_cosmed.task_ts_dict['Start 6MWT-R 0']-50
    VIS_END = obj_cosmed.task_ts_dict['End Run 0']+50

    inspect_patch2(df_patch, subject_id, title_str='raw', VIS_START=VIS_START, VIS_END=VIS_END, show_plot=True)


# In[17]:


# for key in df_patch:
#     print(key)


# In[ ]:





# In[ ]:





# # get df_cosmed

# In[18]:


df_cosmed = obj_cosmed.df_data
df_cosmed


# # get time vectors of both patch and cosmed

# In[19]:


t_patch = df_patch['time'].values
t_cosmed = df_cosmed['time(s)'].values


# In[ ]:





# # create t_sync (cosmed and patch will share this t vector)

# In[20]:


# t=0 is 3 taps
t_study_start = 0

# the last timestamp that is not equal to Nan
t_study_end = df_cosmed[df_cosmed['task']!='Nan']['time(s)'].max()

t_sync = t_patch[(t_patch>=t_study_start) & (t_patch<=t_study_end)]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # resample cosmed data so its time vector is now t_sync

# In[21]:


cosmed_resampled = {}


for cosmed_label in list(df_cosmed.columns):
    
    if cosmed_label in ['time(s)', 'task', 'Marker']:
        continue
        
    v_cosmed_label = df_cosmed[cosmed_label].values
    t_cosmed = df_cosmed['time(s)'].values

    v_cosmed_label_sync = np.interp(t_sync, t_cosmed, v_cosmed_label)
    cosmed_resampled[cosmed_label+'_cosmed'] = v_cosmed_label_sync


# In[22]:


cosmed_resampled['time'] = t_sync


# # create an equally long Marker vector

# In[23]:


v_cosmed_Marker = np.zeros(t_sync.shape[0])
t_markers = df_cosmed[df_cosmed['Marker']==1]['time(s)'].values
for t_marker in t_markers:
    i_min = np.argmin(np.abs(t_sync-t_marker))
    v_cosmed_Marker[i_min] = 1
    
cosmed_resampled['Marker_cosmed'] = v_cosmed_Marker


# In[ ]:





# In[24]:


# obj_cosmed.task_ts_dict


# # create an equally long task vector
# ## for outdoor walk, start from `Start Walk 0` till `Start Stair 1`

# In[25]:


# FS_tasks


# In[26]:


# obj_cosmed.task_ts_dict


# In[ ]:





# In[ ]:





# In[27]:


# task_ts_dict = obj_cosmed.task_ts_dict

# for task_name in task_ts_dict:
#     if 'Stair' in task_name:
#         print('{}\t\t{:.2f}s'.format(task_name, task_ts_dict[task_name]) )
# # 'Bottom Stair 0' in task_ts_dict


# In[ ]:





# In[ ]:





# In[28]:


# task_ts_dict = obj_cosmed.task_ts_dict

# for task_name in task_ts_dict:
#     if 'Stair' in task_name:
#         print('{}\t\t{:.2f}s'.format(task_name, task_ts_dict[task_name]) )
# # 'Bottom Stair 0' in task_ts_dict


# In[29]:


# if subject_id=='sub104':
    
#     t_SB = task_ts_dict['Stair Bottom 0']
#     t_ST = task_ts_dict['Stair Top 0']
    
#     task_ts_dict['Stair Bottom 0'] = t_ST
#     task_ts_dict['Stair Top 0'] = t_SB


# In[30]:


v_cosmed_task = np.asarray(['Transition']*t_sync.shape[0], dtype=object)

task_ts_dict = obj_cosmed.task_ts_dict

for task_name in FS_tasks:
    print(task_name)
    task_start = 'Start ' + task_name
    task_end = 'End ' + task_name
    
    if 'Stair' not in task_name:
        if task_start not in task_ts_dict:
            print('{} not in task_ts_dict'.format(task_name))
            continue
            
        t_start =  task_ts_dict[task_start]
        t_end =  task_ts_dict[task_end]
    
    if task_name == 'Walk 0':
        t_stair2rest = 70
        
        print('\tUsing ==Start Walk 0== till ==Start Stair 1== for ====={}====='.format(task_name))
        print('\t ==Start Walk 0== till ==End Walk 0==: {:.2f} sec'.format( t_end-t_start ) )
        t_start =  task_ts_dict['Start Walk 0']
        if 'Start Stair 1' in task_ts_dict:
            t_end =  task_ts_dict['Start Stair 1']
#             t_end =  task_ts_dict['End Walk 0']
            print('\t ==Start Walk 0== till ==Start Stair 1==: {:.2f} sec'.format( t_end-t_start ) )
        else:
            t_end =  task_ts_dict['End Walk 0'] - t_stair2rest # remove the last 70s
            print('\t ==Start Walk 0== till ==End Walk 0 - {:.2f} sec==: {:.2f} sec'.format( t_stair2rest, t_end-t_start ) )

        
        
        
    if task_name == 'StairDown0':
        t_start =  task_ts_dict['Start Stair 0']
                
        if 'Stair Bottom 0' in task_ts_dict:
            t_end =  task_ts_dict['Stair Bottom 0']
        elif 'Bottom Stair 0' in task_ts_dict:
            t_end =  task_ts_dict['Bottom Stair 0']

        print('\t ===={}====: {:.2f} sec'.format( task_name, t_end-t_start ) )
        
    if task_name == 'StairUp0':
#         t_start =  task_ts_dict['Stair Bottom 0']
#         t_end =  task_ts_dict['Stair Top 0']

        if 'Stair Bottom 0' in task_ts_dict:
            t_start =  task_ts_dict['Stair Bottom 0']
        elif 'Bottom Stair 0' in task_ts_dict:
            t_start =  task_ts_dict['Bottom Stair 0']
            
        if 'Stair Top 0' in task_ts_dict:
            t_end =  task_ts_dict['Stair Top 0']
        elif 'Top Stair 0' in task_ts_dict:
            t_end =  task_ts_dict['Top Stair 0']
            
        print('\t ===={}====: {:.2f} sec'.format( task_name, t_end-t_start ) )
        
                
    if task_name == 'StairDown1':
#         t_start =  task_ts_dict['Stair Top 0']
        if 'Stair Top 0' in task_ts_dict:
            t_start =  task_ts_dict['Stair Top 0']
        elif 'Top Stair 0' in task_ts_dict:
            t_start =  task_ts_dict['Top Stair 0']
        t_end =  task_ts_dict['End Stair 0']
        print('\t ===={}====: {:.2f} sec'.format( task_name, t_end-t_start ) )
        
        
        
        
        

    
    task_mask = (t_sync>=t_start) & (t_sync<t_end)
    v_cosmed_task[task_mask] = task_name
    
cosmed_resampled['task'] = v_cosmed_task


# In[31]:


# task_ts_dict


# In[32]:


# np.unique(cosmed_resampled['task'])


# In[ ]:





# # create an equally long Sample vector (to record when cosmed makes an sample)

# In[33]:


v_cosmed_sampled = np.zeros(t_sync.shape[0])


# ## time consuming, consider replacing argmin with something else

# In[ ]:





# In[34]:


# indices_mask[0].shape, t_cosmed.shape


# In[35]:


t_cosmed = df_cosmed['time(s)'].values

cosmed_label = 'RR'
v_cosmed_label = df_cosmed[cosmed_label].values

# if RR values are wrong, discard it completely
indices_mask = np.where((v_cosmed_label>=label_range_dict[cosmed_label][0]) & (v_cosmed_label<=label_range_dict[cosmed_label][1]))
t_cosmed = t_cosmed[indices_mask]

for t in t_cosmed:
    i_min = np.argmin( np.abs(t - t_sync) )

    v_cosmed_sampled[i_min] = 1

cosmed_resampled['Sampled_cosmed'] = v_cosmed_sampled


# # patch and cosmed data are sync!

# In[36]:


df_sync = pd.DataFrame.from_dict(cosmed_resampled)


# In[ ]:





# In[37]:


# df_sync[df_sync['task']=='Walk 0']['time'].min(), df_sync[df_sync['task']=='Walk 0']['time'].max()


# In[38]:


# df_sync[df_sync['task']=='Stair 1']['time'].min(), df_sync[df_sync['task']=='Stair 1']['time'].max()


# In[39]:


# for key in cosmed_resampled:
#     print(cosmed_resampled[key].shape)


# In[ ]:





# In[ ]:





# In[40]:


df_sync['task'].unique()


# In[41]:


t_patch = df_patch['time'].values
df_patch_sync = df_patch[(t_patch>=t_study_start) & (t_patch<=t_study_end)]
# df_patch_sync = df_patch_sync.drop(columns=['HR_cosmed'])
df_patch_sync


# In[42]:


df_sync  = pd.merge(df_sync, df_patch_sync, on="time")


# In[43]:


df_sync


# In[44]:


# plt.plot(df_sync['task'])


# In[45]:


df_sync['task'].unique()


# In[46]:


def check_walk(df_sync, task_name='Walk 0', fig_name=None, outputdir=None, show_plot=False):
    df = df_sync[df_sync['task']==task_name]
    t_vector = df['time'].values
    t_vector = t_vector-t_vector.min()
    
    fig, axes = plt.subplots(3,1, figsize=(100,6), dpi=120,  facecolor='white')

    for i, sig_name in enumerate(['accelX', 'accelY', 'accelZ']):
        ax = axes[i]
        ax.plot(t_vector[::15], df[sig_name].values[::15])
        ax.set_xlim(t_vector.min(),t_vector.max())
        ax_no_top_right(ax)

    ax.set_xlabel('time (s)')

    fig.tight_layout()
    plt.show()
    
    if fig_name is None:
        fig_name = 'ACC_Walk0'

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


# In[63]:


# # df = df_sync[df_sync['task']=='StairDown1']
# df = df_sync[df_sync['task']=='Walk 0']


# In[59]:


# df_sync['task'].unique()


# In[62]:


# plt.plot(df['pres'].values)
# plt.show()


# In[65]:


# subject_id


# In[64]:


check_walk(df_sync, task_name='Walk 0', outputdir=outputdir_sub, show_plot=False)


# In[51]:


# outputdir


# In[ ]:


# t_vector.max()-60


# In[51]:


# plt.plot(df_sync['VO2_cosmed'])


# # add OUES

# In[52]:


import sys
epsilon = sys.float_info.epsilon
epsilon


# In[53]:


# lighting fast! 
@nb.njit(fastmath=True)
def get_slope(X, Y):
    downsample_factor = int(FS_RESAMPLE/5) # respiratory params should be slower than 1Hz
#     downsample_factor = 1 # respiratory params should be slower than 1Hz

    X = X[::downsample_factor]
    Y = Y[::downsample_factor]
    m_x=np.mean(X)
    m_y=np.mean(Y)
    return np.sum((X-m_x)*(Y-m_y))/(np.sum((X-m_x)**2) + epsilon)
#     return np.sum((X-m_x)*(Y-m_y))/np.sum((X-m_x)**2)

def roll_dataframe(df_raw, window, col1, col2, col_new, compute_method):
    df = df_raw.copy()
    df[col_new] = np.nan


    df_start = df[:window//2].iloc[::-1].copy()
    df_end = df[df.shape[0]-window//2:].iloc[::-1].copy()

    df_padded = pd.concat([df_start, df, df_end])
    

    df_padded = df_padded.reset_index(drop=True)
    v_arr = rolling_apply(compute_method, window, df_padded[col1].values, df_padded[col2].values,)

    df_padded.loc[(window-1)//2:df_padded.shape[0]-(window//2)-1, col_new] =  v_arr[window-1:]
    
    df_rolled = df_padded[window//2:df_padded.shape[0]-window//2]
    
    return df_rolled


# In[54]:


# %%time
window = int(30*FS_RESAMPLE)
# aaa = roll_dataframe(df_sync[:500*FS_RESAMPLE], window, 'VE_cosmed', 'VO2_cosmed', 'OUES', get_slope)
df_sync = roll_dataframe(df_sync, window, 'VE_cosmed', 'VO2_cosmed', 'OUES_cosmed', get_slope).reset_index(drop=True)


# In[55]:


df_sync


# In[56]:


df = df_sync[df_sync['Sampled_cosmed']==1.].copy()
df = df[(df['task']!='Transition') & 
        (df['task']!='SpeakScripted 0') & (df['task']!='SpeakScripted 1') & (df['task']!='SpeakCasual 0') & (df['task']!='SpeakCasual 1') &
         (df['task']!='LL 0') & (df['task']!='LR 0') & (df['task']!='Cough 0') & (df['task']!='Cough 1')
       ].copy()

# import seaborn as sns
# import pandas as pd

# df = pd.read_csv('worldHappiness2016.csv')

sns.scatterplot(data = df, x = "VE_cosmed", y = "EE_cosmed", hue='task')

# plt.show()


# In[ ]:





# # plot the signals

# In[57]:


Fs = FS_RESAMPLE


# In[58]:


def stage2_get_filt_df(df_sync, Fs):

    df = df_sync.copy()

#     Fs = FS_RESAMPLE
    # lowcutoff = 0.8
    # highcutoff = 8

    df['ECG'] = get_padded_filt(df['ECG'].values, filter_padded=5, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=Fs)

    df['ppg_ir_1'] = -get_padded_filt(df['ppg_ir_1'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
    df['ppg_r_1'] = -get_padded_filt(df['ppg_r_1'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
    df['ppg_g_1'] = -get_padded_filt(df['ppg_g_1'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
    df['ppg_ir_2'] = -get_padded_filt(df['ppg_ir_2'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
    df['ppg_r_2'] = -get_padded_filt(df['ppg_r_2'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
    df['ppg_g_2'] = -get_padded_filt(df['ppg_g_2'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
    
    return df


# In[59]:


# define range
t_start = 0
t_end = df_sync[df_sync['task']=='Recovery 4']['time'].max()

# get a copy of data in this range
# df = get_filt_df(df_sync, Fs).copy()
df = stage2_get_filt_df(df_sync, Fs).copy()
df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]

# downsample it to make plotting quicker
downsample_factor = 1
df = df[df.index%downsample_factor==0]

# plot the signals
plt_scale = 1.2




start_task = df[df['task'].str.contains("6MWT", case=False)]['task'].unique()[0]
end_task = 'Run 0'

t_start = df[df['task']==start_task]['time'].min()
t_end = df[df['task']==end_task]['time'].max()
t_start, t_end

fig_name = '6MWT2Run_{}s'.format(int(t_end-t_start))

plot_all_sync(df, subject_id,plt_scale=plt_scale, outputdir=outputdir_sub, fig_name=fig_name, show_plot=False)


# In[ ]:





# In[ ]:





# In[ ]:





# # save the data

# In[60]:


if saving_format=='csv':
    df_sync.to_csv(outputdir_sub+'df_sync.'+saving_format, index=False)
elif saving_format=='feather':
    df_sync.to_feather(outputdir_sub+'df_sync.'+saving_format)

    



# In[ ]:





# # 4 sub 

# In[61]:


export_4sub = False

if export_4sub:

    df_sub = df_sync[['time', 'HR_cosmed',	'VT_cosmed',	'VE_cosmed',	'VO2_cosmed',	'VCO2_cosmed',	'EE_cosmed',	'SPO2_cosmed',	'AmbTemp_cosmed',	'Sampled_cosmed', 'task']].copy()
    df_sub = df_sub[df_sub['Sampled_cosmed']==1]

#     df_sub.to_csv(outputdir_sub+'df_4sub.csv')
#     plt.plot(df_sub['time'], df_sub['VT_cosmed'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




