import numpy as np

import pandas as pd

import pickle

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline

from plotting_tools import *
from setting import *


def get_sec(time_str):
    """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return float(h) * 3600 + float(m) * 60 + float(s)

class Subject:
    def __init__(self, df_cosmed, task_ts_dict, subject_id, weight_norm=False):       
       
        self.Firstname = df_cosmed[df_cosmed[0]=='Firstname'][1].values[0]
        self.Lastname = df_cosmed[df_cosmed[0]=='Lastname'][1].values[0]

        self.Height = float(df_cosmed[df_cosmed[0]=='Height'][1].values[0])
        self.Weight = float(df_cosmed[df_cosmed[0]=='Weight'][1].values[0])

        self.Gender = df_cosmed[df_cosmed[0]=='Gender'][1].values[0]
        self.mode = df_cosmed[df_cosmed[0]=='Test type'][1].values[0]
        
        self.subject_id = subject_id
        
        self.task_ts_dict = task_ts_dict
        
        # get timeseries data
        df_data = df_cosmed[df_cosmed[df_cosmed[0]=='IDS'].index[0]:] # the 0~25 rows are not timeseries data
        df_data.columns = df_data.iloc[0]
        df_data = df_data[2:].reset_index(drop=True)

        df_data = df_data[['HR','K5_FeO2','K5_FeCO2','FiO2','FiCO2', 'K5_VO2','K5_VCO2','VT', 'K5_Rf', 'K5_VE', 'hh:mm:ss ', 'GpsAltK4', 'Battery', 'Marker', 'AmbTempK4', 'SPO2', 'K5_R']]


        list_labels = ['HR','K5_FeO2','K5_FeCO2','FiO2','FiCO2', 'K5_VO2','K5_VCO2','VT', 'K5_Rf', 'K5_VE', 'GpsAltK4', 'Battery', 'Marker', 'AmbTempK4', 'SPO2', 'K5_R']

        for label in list_labels:
            df_data.loc[:, label] = df_data.loc[:, label].astype(float)


        df_data.loc[:, 'hh:mm:ss '] = df_data.loc[:, 'hh:mm:ss '].map(get_sec)
        
        df_data = df_data.rename(columns={"K5_FeO2": "FeO2","K5_FeCO2": "FeCO2","FiO2": "FiO2","FiCO2": "FiCO2", "K5_VO2": "VO2", "K5_VCO2": "VCO2", "K5_Rf": "RR", "K5_VE": "VE", "hh:mm:ss ": "time(s)", "GpsAltK4": "GpsAlt", "AmbTempK4": "AmbTemp"})

        # Weir formula. ref: https://en.wikipedia.org/wiki/Weir_formula, (kcal per minute)
        df_data['EE'] = (3.94 * df_data['VO2'] + 1.11 * df_data['VCO2']) / 1000
        
        # raw VE is in L/min -> change it to ml/min
        df_data['VE'] = df_data['VE'] * 1000

        df_data = df_data[['time(s)', 'Marker', 'HR', 'RR', 'VT', 'VE', 'FeO2',  'FeCO2',  'FiO2',  'FiCO2', 'VO2', 'VCO2', 'EE', 'SPO2', 'GpsAlt', 'AmbTemp', 'Battery', 'K5_R']]
        df_data = df_data.reset_index(drop=True)

#         print(self.Weight)
        # scale these variables by weight
        if weight_norm:
            df_data['VO2'] = df_data['VO2'].values.astype('float') / self.Weight
            df_data['VCO2'] = df_data['VCO2'].values.astype('float') / self.Weight
            df_data['EE'] = df_data['EE'].values.astype('float') / self.Weight
            df_data['VT'] = df_data['VT'].values.astype('float') / self.Weight
            df_data['VE'] = df_data['VE'].values.astype('float') / self.Weight


        
#         if self.subject_id =='sub101':
#             cosmed_mark_delay_patch_taps_by = 13
#         else:
#             cosmed_mark_delay_patch_taps_by = 0
#         if cosmed_mark_delay_patch_taps_by!= 0:
#             t_Marker = df_data.loc[df_data['Marker']==1, 'time(s)'].values[0]
#             i_Marker = df_data[df_data['Marker']==1].index[0]

#             t_Marker = t_Marker - cosmed_mark_delay_patch_taps_by
#             df_data.loc[i_Marker, 'time(s)'] = t_Marker
#             df_data = df_data.sort_values(by=['time(s)'])
#             df_data = df_data.reset_index(drop=True)




#         if self.subject_id =='sub101':
#             cosmed_mark_delay_patch_taps_by = 13
#         else:
#             cosmed_mark_delay_patch_taps_by = 0

# #         print(df_cosmed[df_cosmed['Marker']==1])
#         if cosmed_mark_delay_patch_taps_by!= 0:
# #             i_marker = df_data[df_data['Marker']==1].index[0]
# #             df_data[df_data[df_data['Marker']==1], 'time(s)']
#             t_Marker = df_data.loc[df_data['Marker']==1, 'time(s)'].values[0]
#             i_Marker = df_data[df_data['Marker']==1].index[0]
# #             print(t_Marker)
# #             print()
        
#             t_Marker = t_Marker - cosmed_mark_delay_patch_taps_by
# #             print(df_data[i_Marker:i_Marker+2])
# #             print(t_Marker)
# #             print(df_data)
#             df_data.loc[i_Marker, 'time(s)'] = t_Marker
#             df_data = df_data.sort_values(by=['time(s)'])
# #             print(df_data)
 
#             df_data = df_data.reset_index(drop=True)

#             i_Marker = df_data[df_data['Marker']==1].index[0]



#             print(df_data[i_Marker-10:i_Marker+10])

#             sys.exit()
            
#             print(df_data[df_data['Marker']==1])
#             df_data['time(s)'] = df_data['time(s)']+cosmed_mark_delay_patch_taps_by
#             df_data['Marker']=0
#             df_data.loc[df_data['time(s)']==0, ['Marker']] = 1
#             print(df_data[df_data['Marker']==1])

        
        self.df_data = df_data
        
        # make the first marker the start of the study
        self.reset_time_col()
        
        # create a task column that specify the task name of a data entry
        self.init_task_col(True)

        
    def reset_time_col(self):
        # TODO: add documentation
        ith_marker = 0 # the 0th marker marks the start of baseline, may be different for each subject

        i_start = np.where(self.df_data['Marker']==1)[0][ith_marker]
        t_start = self.df_data.iloc[i_start,:]['time(s)']
        self.df_data['time(s)'] = self.df_data['time(s)'] - t_start

    def init_task_col(self, debug=False):
        # TODO: add documentation
        self.df_data['task'] = 'Nan'
        
        if debug:
            print('{:<15} |\t{:>10}\t\t{:>10}'.format('task name', 'start time', 'duration'))
            print('===============================================================')
            
        for task_name in FS_tasks:
            task_start = 'Start ' + task_name
            task_end = 'End ' + task_name
            
            if task_start not in self.task_ts_dict:
                print('{} not in task_ts_dict'.format(task_name))
                continue

            if debug:
                print('{:<15} |\t{:>10.2f}s\t\t{:>10.2f}s'.format(task_name, self.task_ts_dict[task_start], self.task_ts_dict[task_end]-self.task_ts_dict[task_start]) )

            t_start =  self.task_ts_dict[task_start]
            t_end =  self.task_ts_dict[task_end]
            indices = self.df_data[(self.df_data['time(s)']>=t_start) & (self.df_data['time(s)']<=t_end)].index
#             print('\t', t_start, t_end, indices.shape)
            self.df_data.loc[indices,'task'] = task_name
            
    def show_demographic(self):
        print('====== subject demographic info ======')
        print("subject_id: " + self.subject_id)
        print("name: " + self.Firstname, self.Lastname)
        print("height: {} cm".format(self.Height) )
        print("weight: {} kg".format(self.Weight) )
        print("gender: " + self.Gender)
        print("cosmed data collected using mode " + self.mode)
        print('======================================')
        


    def inspect_cosmed(self, VIS_START=-1, VIS_END=-1, task_name=None, outputdir=None, show_plot=False):
        df_data = self.df_data.copy()
        
        if VIS_START==-1:
            VIS_START = df_data['time(s)'].min()

        if VIS_END==-1:
            VIS_END = df_data['time(s)'].max()

        if task_name is not None:
            if (df_data['task']==task_name).sum()==0:
                print('CODMED did not capture any data during ', task_name) # this could happen for coughing
                return 
            
            VIS_START = df_data[df_data['task']==task_name]['time(s)'].values[0]
            VIS_END = df_data[df_data['task']==task_name]['time(s)'].values[-1]
            
            title_str = task_name
            
        else:
            title_str = 't={:.2f}~{:.2f}s'.format(VIS_START, VIS_END)

#             VIS_START = task_ts_dict['Start '+task_name]
#             VIS_END = task_ts_dict['End '+task_name]

        
        df_data = df_data[ (df_data['time(s)']>=VIS_START) & (df_data['time(s)']<=VIS_END) ]

        # t_dur = data_dict['time'][-1] - data_dict['time'][0]
        t_dur = df_data['time(s)'].max()-df_data['time(s)'].min()
        print('[{}]: {} sec'.format(self.subject_id, t_dur))

        t_arr =  df_data['time(s)'].values

        N_labels = len(list_cosmed)
        fig, axes = plt.subplots(N_labels, 1, figsize=(10,N_labels), gridspec_kw = {'wspace':0, 'hspace':0}, dpi=80)

        # # fig = plt.figure(figsize=(3*t_dur/80, 3*10), dpi=50, facecolor='white')
        # fig = plt.figure(figsize=(20, 10), dpi=80, facecolor='white', gridspec_kw = {'wspace':0, 'hspace':0})

        fontsize = 10
        scale_factor = 8
        alpha = 0.8

        for i, (ax, label_name) in enumerate(zip(axes, list_cosmed)):
            ax.grid('on', linestyle='--')
            if i<len(axes)-1:
                ax.set_xticklabels([])

            ax.tick_params(axis='y', which='both', labelsize=20)



            ax.plot(t_arr, df_data[label_name].values, color=color_dict[label_color_dict[label_name]], alpha=0.5 ,zorder=1)

            ax.set_xlim(t_arr[0],t_arr[-1])


        #         ax.set_xlim(t_start, t_end)
            ax.spines['right'].set_visible(False)
        #     ax.set_ylabel(label_name, fontsize=fontsize, rotation = 0, labelpad=0)
            ax.set_ylabel(label_name, fontsize=fontsize, labelpad=10)

            ax.tick_params(axis='both', which='major', labelsize=fontsize/1.1)

        ax.set_xlabel('time (sec)', fontsize=fontsize)

        fig.suptitle(title_str, fontsize=fontsize+5)

        fig.subplots_adjust(wspace=0, hspace=0)

        fig.tight_layout()

        # #     fig_name = '{}_signl_{}'.format(title_str,subject_id)
        fig_name = 'inspect_labels_{}'.format(title_str)

        if outputdir is not None:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

        if show_plot == False:
            plt.close(fig)
            pyplot.close(fig)
            plt.close('all')

    #     if log_wandb:
    # #         wandb.log({fig_name: wandb.Image(fig)})
    #         wandb.log({fig_name: fig})


# save sub to file
def save_sub(sub_object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(sub_object, f)

# load sub from file
def load_sub(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f) 

list_cosmed = ['HR', 'RR', 'VT', 'VE', 'VO2', 'VCO2', 'EE', 'SPO2', 'GpsAlt', 'AmbTemp']

label_color_dict = {
    'HR': 'Maroon',
    'RR': 'SteelBlue',
    'VT': 'MidnightBlue',
    'VE': 'MangoTando',
    'VO2': 'Red',
    'VCO2': 'burntumber',
    'EE': 'Orange',
    'SPO2': 'Firebrick',
    'GpsAlt': 'ForestGreen',
    'AmbTemp': 'Deep Carrot Orange',
}

