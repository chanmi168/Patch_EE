import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline

import numpy as np
from filters import *
from setting import *
from plotting_tools import *

def data_interpolation(raw_dict, FS_RESAMPLE=500):

    ecg_time = raw_dict['ecg_time']
    ppg_time = raw_dict['ppg_time']
    accel_time = raw_dict['accel_time']

    t_start = np.max([ecg_time[0], ppg_time[0], accel_time[0]])
    t_end = np.min([ecg_time[-1], ppg_time[-1], accel_time[-1]])

    time_interp = np.arange(my_ceil(t_start, decimal=-3)*FS_RESAMPLE, my_floor(t_end, decimal=-3)*FS_RESAMPLE+1)/FS_RESAMPLE
    
    patch_dict = {}

    # ECG
    patch_dict['ECG'] = np.interp(time_interp, raw_dict['ecg_time'], raw_dict['ecg'])

    # ACC
    patch_dict['accelX'] = np.interp(time_interp, raw_dict['accel_time'], raw_dict['accel_x'])
    patch_dict['accelY'] = np.interp(time_interp, raw_dict['accel_time'], raw_dict['accel_y'])
    patch_dict['accelZ'] = np.interp(time_interp, raw_dict['accel_time'], raw_dict['accel_z'])

    # TODO: find out which PPG arr this belongs to (in terms of its physical location)
    # PPG array 1 (which one is it? left or right?)
    patch_dict['ppg_g_1'] = np.interp(time_interp, raw_dict['ppg_time'], raw_dict['ppg_g_1'])
    patch_dict['ppg_r_1'] = np.interp(time_interp, raw_dict['ppg_time'], raw_dict['ppg_r_1'])
    patch_dict['ppg_ir_1'] = np.interp(time_interp, raw_dict['ppg_time'], raw_dict['ppg_ir_1'])

    # PPG array 2 (which one is it? left or right?)
    patch_dict['ppg_g_2'] = np.interp(time_interp, raw_dict['ppg_time'], raw_dict['ppg_g_2'])
    patch_dict['ppg_r_2'] = np.interp(time_interp, raw_dict['ppg_time'], raw_dict['ppg_r_2'])
    patch_dict['ppg_ir_2'] = np.interp(time_interp, raw_dict['ppg_time'], raw_dict['ppg_ir_2'])

    # skin temperature
    patch_dict['temp_skin'] = np.interp(time_interp, raw_dict['env_time'], medfilt(raw_dict['temp_skin'], k=3))
    # pressure
    patch_dict['pres'] = np.interp(time_interp, raw_dict['env_time'], get_smooth(raw_dict['pres'], N=51))

    time_interp = time_interp-time_interp[0]
    patch_dict['time'] = time_interp
    patch_dict['subject_id'] = raw_dict['subject_id']

    return patch_dict


def get_filt_dict(df_patch):
    print('Filtering the raw patch signals...')
    df_patch_filt = df_patch.copy()
    
#     patch_filt_dict = {}
#     Fs = FS_RESAMPLE
    Fs = np.round(1/(df_patch_filt['time'][1]-df_patch_filt['time'][0]))
    print('Signal has been resampled to {} Hz'.format(Fs))

    lowcutoff = 0.8
    highcutoff = 8

#     df_patch_filt['time'] = df_patch['time']
    
    df_patch_filt['ECG'] = get_padded_filt(df_patch['ECG'].values, filter_padded=5, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=FS_RESAMPLE)
#     patch_filt_dict['ECG'] = (patch_filt_dict['ECG'] - np.mean(patch_filt_dict['ECG'])) / np.std(patch_filt_dict['ECG'])

    
    df_patch_filt['ppg_ir_1'] = -get_padded_filt(df_patch['ppg_ir_1'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
    df_patch_filt['ppg_r_1'] = -get_padded_filt(df_patch['ppg_r_1'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
    df_patch_filt['ppg_g_1'] = -get_padded_filt(df_patch['ppg_g_1'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
    df_patch_filt['ppg_ir_2'] = -get_padded_filt(df_patch['ppg_ir_2'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
    df_patch_filt['ppg_r_2'] = -get_padded_filt(df_patch['ppg_r_2'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
    df_patch_filt['ppg_g_2'] = -get_padded_filt(df_patch['ppg_g_2'].values, filter_padded=5, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
    
    # may be unnecessary
#     ts = patch_filt_dict['time']
    
#     patch_filt_dict['ppg_ir_1'] = -get_padded_filt_DSwrapper(ts, patch_dict['ppg_ir_1'], filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=FS_RESAMPLE)
#     patch_filt_dict['ppg_r_1'] = -get_padded_filt_DSwrapper(ts, patch_dict['ppg_r_1'], filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=FS_RESAMPLE)
#     patch_filt_dict['ppg_g_1'] = -get_padded_filt_DSwrapper(ts, patch_dict['ppg_g_1'], filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=FS_RESAMPLE)
#     patch_filt_dict['ppg_ir_2'] = -get_padded_filt_DSwrapper(ts, patch_dict['ppg_ir_2'], filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=FS_RESAMPLE)
#     patch_filt_dict['ppg_r_2'] = -get_padded_filt_DSwrapper(ts, patch_dict['ppg_r_2'], filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=FS_RESAMPLE)
#     patch_filt_dict['ppg_g_2'] = -get_padded_filt_DSwrapper(ts, patch_dict['ppg_g_2'], filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=FS_RESAMPLE)
    
    df_patch_filt['accelX'] = get_padded_filt(df_patch['accelX'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=FS_RESAMPLE)
    df_patch_filt['accelY'] = get_padded_filt(df_patch['accelY'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=FS_RESAMPLE)
    df_patch_filt['accelZ'] = get_padded_filt(df_patch['accelZ'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=FS_RESAMPLE)

    df_patch_filt['pres'] = df_patch['pres'].values
    df_patch_filt['temp_skin'] = df_patch['temp_skin'].values

#     patch_filt_dict['subject_id'] = patch_dict['subject_id']
    
    
    print('ECG passband: {} Hz'.format(FILT_ECG))
    print('SCG passband: {} Hz'.format(FILT_SCG))
    print('PPG passband: {} Hz'.format(FILT_PPG))
    return df_patch_filt


def inspect_patch(df_patch, title_str, VIS_START = 200, VIS_END = 230, plt_scale=4/3, outputdir=None, show_plot=False):
   
#     # patch_dict
#     VIS_START = 200
#     VIS_END = 230
    
    df = df_patch[(df_patch['time']>VIS_START) & (df_patch['time']<VIS_END)]
        
        
#     for key in data_dict:
#         if key != 'subject_id':
#             data_dict[key] = data_dict[key][VIS_START*FS_RESAMPLE:VIS_END*FS_RESAMPLE]
    
    subject_id = df['subject_id'].unique()[0]

    t_arr =  df['time'].values
    t_dur = t_arr[-1] - t_arr[0]
    
    print('[{}]: {} sec'.format(subject_id, t_dur))

    # fig = plt.figure(figsize=(3*t_dur/80, 3*10), dpi=50, facecolor='white')
    fig = plt.figure(figsize=(15*plt_scale, 15), dpi=80, facecolor='white')

    fontsize = 15
    scale_factor = 8
    alpha = 0.8

    ax1 = fig.add_subplot(6,1,1)
    ax1.plot(t_arr, df['ECG'].values, color=color_dict[sync_color_dict['ECG']], alpha=0.5 ,zorder=1)

    ax1.set_xlim(t_arr[0],t_arr[-1])
    ax1.set_title(title_str + 'ECG', fontsize=fontsize)
    ax1.set_ylabel('mV', fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=fontsize)
    ax1.tick_params(axis='y', labelsize=fontsize)


    ax2 = fig.add_subplot(6,1,2)
    ax2.plot(t_arr, df['accelX'].values, color=color_dict[sync_color_dict['accelX']], alpha=alpha ,zorder=1, label='accelX')
    ax2.plot(t_arr, df['accelY'].values, color=color_dict[sync_color_dict['accelY']], alpha=alpha ,zorder=1, label='accelY')
    ax2.plot(t_arr, df['accelZ'].values, color=color_dict[sync_color_dict['accelZ']], alpha=alpha ,zorder=1, label='accelZ')
    ax2.set_xlim(t_arr[0],t_arr[-1])
    ax2.set_title(title_str + 'ACC', fontsize=fontsize)
    ax2.set_ylabel('g', fontsize=fontsize)
    ax2.tick_params(axis='x', labelsize=fontsize)
    ax2.tick_params(axis='y', labelsize=fontsize)
    ax2.legend(loc='upper left', fontsize=fontsize, frameon=True)

    ax3 = fig.add_subplot(6,1,3)
    ax3.plot(t_arr, df['ppg_g_1'].values, color=color_dict[sync_color_dict['ppg_g_1']], alpha=alpha ,zorder=1, label='ppg_g_1')
    ax3.plot(t_arr, df['ppg_r_1'].values, color=color_dict[sync_color_dict['ppg_r_1']], alpha=alpha ,zorder=1, label='ppg_r_1')
    ax3.plot(t_arr, df['ppg_ir_1'].values, color=color_dict[sync_color_dict['ppg_ir_1']], alpha=alpha ,zorder=1, label='ppg_ir_1')
    ax3.set_xlim(t_arr[0],t_arr[-1])
    ax3.set_title(title_str + 'PPG arr 1', fontsize=fontsize)
    ax3.set_ylabel('ppg intensity (uW)', fontsize=fontsize)
    ax3.tick_params(axis='x', labelsize=fontsize)
    ax3.tick_params(axis='y', labelsize=fontsize)
    ax3.legend(loc='upper left', fontsize=fontsize, frameon=True)

    ax4 = fig.add_subplot(6,1,4)
    ax4.plot(t_arr, df['ppg_g_2'].values, color=color_dict[sync_color_dict['ppg_g_2']], alpha=alpha ,zorder=1, label='ppg_g_2')
    ax4.plot(t_arr, df['ppg_r_2'].values, color=color_dict[sync_color_dict['ppg_r_2']], alpha=alpha ,zorder=1, label='ppg_r_2')
    ax4.plot(t_arr, df['ppg_ir_2'].values, color=color_dict[sync_color_dict['ppg_ir_2']], alpha=alpha ,zorder=1, label='ppg_ir_2')
    ax4.set_xlim(t_arr[0],t_arr[-1])
    ax4.set_title(title_str + 'PPG arr 2', fontsize=fontsize)
    ax4.set_ylabel('ppg intensity (uW)', fontsize=fontsize)
    ax4.tick_params(axis='x', labelsize=fontsize)
    ax4.tick_params(axis='y', labelsize=fontsize)
    ax4.legend(loc='upper left', fontsize=fontsize, frameon=True)
    
    ax5 = fig.add_subplot(6,1,5)
    ax5.plot(t_arr, df['temp_skin'].values, color=color_dict[sync_color_dict['temp_skin']], alpha=alpha ,zorder=1, label='skin_temp')
    ax5.set_xlim(t_arr[0],t_arr[-1])
    ax5.set_title(title_str + 'skin temp', fontsize=fontsize)
    ax5.set_ylabel('skin temp ({})'.format(unit_dict['temp']), fontsize=fontsize)
    ax5.tick_params(axis='x', labelsize=fontsize)
    ax5.tick_params(axis='y', labelsize=fontsize)
    ax5.legend(loc='upper left', fontsize=fontsize, frameon=True)
    
    ax5 = fig.add_subplot(6,1,6)
    ax5.plot(t_arr, df['pres'].values, color=color_dict[sync_color_dict['pres']], alpha=alpha ,zorder=1, label='pressure')
    ax5.set_xlim(t_arr[0],t_arr[-1])
    ax5.set_title(title_str + 'atmospheric pressure', fontsize=fontsize)
    ax5.set_ylabel('pressure ({})'.format(unit_dict['pres']), fontsize=fontsize)
    ax5.tick_params(axis='x', labelsize=fontsize)
    ax5.tick_params(axis='y', labelsize=fontsize)
    ax5.legend(loc='upper left', fontsize=fontsize, frameon=True)


    ax4.set_xlabel('time (sec)', fontsize=fontsize)

    fig.tight_layout()
    
#     fig_name = '{}_signl_{}'.format(title_str,subject_id)
    fig_name = '{}_signl_{}'.format(title_str, subject_id)

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



def my_ceil(arr, decimal=0):
    return np.ceil(arr*(10**-decimal))/(10**-decimal)

def my_floor(arr, decimal=0):
    return np.floor(arr*(10**-decimal))/(10**-decimal)



