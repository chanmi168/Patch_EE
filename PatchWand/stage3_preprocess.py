import numpy as np
import os
import math
from math import sin
import sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot

import seaborn as sns

from segmentation import *
from filters import *
from setting import *
from preprocessing import *
from plotting_tools import *

from SQI_module import *
from resp_module import *
from ECG_module import *
from PPG_module import *

# # list_input = ['ECG', 'accelZ']
# list_input = ['ECG', 'accelZ', 'HR_patch']
# list_input = ['ECG', 'accelZ', 'HR_patch', 'VE_cosmed']
# list_input = ['ECG', 'accelZ', 'HR_patch', 'VE_cosmed']

# # list_output = ['RR_cosmed', 'VT_cosmed', 'EE_cosmed', 'task']
# # list_output = ['RR_cosmed', 'VT_cosmed', 'EE_cosmed', 'SPO2_cosmed', 'HR_cosmed', 'task']
# list_output = ['RR_cosmed', 'VT_cosmed', 'EE_cosmed', 'SPO2_cosmed', 'HR_cosmed', 'VO2_cosmed', 'resp_cosmed']
               
# list_meta = ['subject_id', 'task']

# FS_RESAMPLE_DL = 100


def sweep(f_arr, t_arr, phase0=0):
    # f_arr is the instantaneous freq
    # t_arr is the time associated with each inst. freq
    # phase0 should be the same as the end phase of the previous segment (use 0 if it's the first segment)
    n_steps = f_arr.shape[0]
    v_arr = np.zeros(n_steps)
    
    for i in range(1, n_steps):  
        # phase can be computed by integrating the inst. freq
        phase = 2 * math.pi * np.trapz(f_arr[0:i+1], x=t_arr[0:i+1])
        v_arr[i] = sin(phase0 + phase)
    return v_arr

def get_sim_breath(t_arr, f_arr, downsample_factor=10):
    # simulate a phase changing sine wave using instantaneous freq f_arr
    # first downsample (downsample_factor), freq sweep, upsample (interp), finally estimate instantaneous freq using simulated sine wave
    t_arr_ds = t_arr[::downsample_factor]
    f_arr_ds = f_arr[::downsample_factor]

    # get sinusodal waveform
    v_arr_ds = sweep(f_arr_ds, t_arr_ds)
    
    # resample back to t_arr
    v_arr = np.interp(t_arr, t_arr_ds, v_arr_ds)
    
    i_peaks = find_peaks(v_arr)[0]
    f_sim = 1/np.gradient(t_arr[i_peaks]) # Hz
    f_sim_interp = np.interp(t_arr, t_arr[i_peaks], f_sim)

    return v_arr, f_sim_interp



def get_ppg_normalized(ppg_raw_patch, Fs):
    sig_filt_DC = get_padded_filt_DSwrapper(ppg_raw_patch, filter_padded=1, lowcutoff=None, highcutoff=0.1, Fs=Fs)
    sig_filt_resp = get_padded_filt(ppg_raw_patch, filter_padded=1, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs)
    return sig_filt_DC, sig_filt_resp
    
def plot_ppg_diagnostics(ax_1, ax_2, ax_3, df_task, ppg_name, Fs):
    fontsize = 10

    ppg_raw_patch = df_task[ppg_name].values
    ppg_filt_DC, ppg_filt_resp = get_ppg_normalized(ppg_raw_patch, Fs)
    ts = np.arange(df_task.shape[0])/Fs

    ax_1.plot(ts, ppg_raw_patch, color=color_dict[sync_color_dict[ppg_name]])
    ax_1.plot(ts, ppg_filt_DC, color='gray', alpha=0.5, linewidth=2)
    ax_1.set_ylabel('[{}] raw and smoothed ppg (<0.1Hz)\n{}'.format(ppg_name, unit_dict['ppg']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

    ax_2.plot(ts, ppg_filt_resp, color=color_dict[sync_color_dict[ppg_name]])
    ax_2.set_ylabel('[{}] (0.08-1Hz)\n{}'.format(ppg_name, unit_dict['ppg']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

    ax_3.plot(ts, ppg_filt_resp/ppg_filt_DC, color=color_dict[sync_color_dict[ppg_name]])
    ax_3.set_ylabel('[{}] PI (resp. ppg / DC ppg)\n{}'.format(ppg_name, 'unitless'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    
    
def plot_inputsigs(df_task, Fs, fig_name=None, outputdir=None, show_plot=False):

#     fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6,1, figsize=(50,8), dpi=70)

#     fig, axes = plt.subplots(22,1, figsize=(20,14), dpi=70)
    t_dur = df_task['time'].max() - df_task['time'].min()

    fig, axes = plt.subplots(32,1, figsize=(t_dur*0.15,20), dpi=60)

#     ppg_raw_patch = df_task['ppg_r_1'].values
#     sig_filt_r_1 = get_ppg_normalized(ppg_raw_patch, Fs)
#     sig_filt_DC = get_padded_filt_DSwrapper(ppg_raw_patch, filter_padded=1, lowcutoff=None, highcutoff=0.1, Fs=Fs)
#     sig_filt_r_1 = get_padded_filt(ppg_raw_patch, filter_padded=1, lowcutoff=0.08, highcutoff=1, Fs=Fs)

#     ppg_raw_patch = df_task['ppg_ir_1'].values
#     sig_filt_ir_1 = get_ppg_normalized(ppg_raw_patch, Fs)

#     ppg_raw_patch = df_task['ppg_g_1'].values
#     sig_filt_g_1 = get_ppg_normalized(ppg_raw_patch, Fs)



    ts = np.arange(df_task.shape[0])/Fs

    f_arr_raw = df_task['RR_cosmed'].values/60

#     t_arr = np.arange(df_task.shape[0])/Fs

    # smooth it since it's usally digital and disgusting
    f_arr = get_smooth(f_arr_raw, N=Fs*5)
    # v_arr is the simulated sinusoidal breath signal, sampled eqaully as t_arr
    v_arr, f_sim_interp = get_sim_breath(ts, f_arr, downsample_factor=Fs//5) # Fs = 250 Hz





    fontsize = 10
    # ax.plot(sig_filt_DC)
    




    plot_ppg_diagnostics(axes[0], axes[1], axes[2], df_task, 'ppg_r_1', Fs)
    plot_ppg_diagnostics(axes[3], axes[4], axes[5], df_task, 'ppg_ir_1', Fs)
    plot_ppg_diagnostics(axes[6], axes[7], axes[8], df_task, 'ppg_g_1', Fs)
    
    plot_ppg_diagnostics(axes[0+9], axes[1+9], axes[2+9], df_task, 'ppg_r_2', Fs)
    plot_ppg_diagnostics(axes[3+9], axes[4+9], axes[5+9], df_task, 'ppg_ir_2', Fs)
    plot_ppg_diagnostics(axes[6+9], axes[7+9], axes[8+9], df_task, 'ppg_g_2', Fs)

#     ppg_raw_patch = df_task['ppg_r_1'].values
# #     sig_filt_DC_r_1, sig_filt_r_1 = get_ppg_normalized(ppg_raw_patch, Fs)
#     ax1.plot(ts, ppg_raw_patch)
#     ax1.plot(ts, sig_filt_DC)
#     ax1.set_ylabel('ppg and smoothed ppg (<0.1Hz)\n{}'.format(unit_dict['ppg']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

#     ax2.plot(ts, sig_filt)
#     ax2.set_ylabel('ppg (0.08-1Hz)\n{}'.format(unit_dict['ppg']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

#     ax3.plot(ts, sig_filt/sig_filt_DC)
#     ax3.set_ylabel('PI (resp. ppg / DC ppg)\n{}'.format('unitless'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

    
#     axes[9].plot(ts, get_padded_filt(df_task['accelX'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs), label='accelX')
#     axes[9].plot(ts, get_padded_filt(df_task['accelY'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs), label='accelY')

    i_axes = 18
    axes[i_axes].plot(ts, get_padded_filt(df_task['accelX'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs), color=color_dict[sync_color_dict['accelX']])
    axes[i_axes].set_ylabel('ACC X (resp)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1
    
    axes[i_axes].plot(ts, get_padded_filt(df_task['accelY'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs), color=color_dict[sync_color_dict['accelY']])
    axes[i_axes].set_ylabel('ACC Y (resp)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, get_padded_filt(df_task['accelZ'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs), color=color_dict[sync_color_dict['accelZ']])
    axes[i_axes].set_ylabel('ACC Z (resp)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1


    axes[i_axes].plot(ts, get_padded_filt(df_task['accelZ'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs), color=color_dict[sync_color_dict['scg_z']])
    axes[i_axes].set_ylabel('SCG\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1


#     if np.isnan(features_spectral['0.00Hz']).all()
    axes[i_axes].plot(ts, df_task['scgY_3.91Hz'].values, color=color_dict[sync_color_dict['scg_y']])
    axes[i_axes].set_ylabel('SCG Y (3.91Hz)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, df_task['scgZ_3.91Hz'].values, color=color_dict[sync_color_dict['scg_z']])
    axes[i_axes].set_ylabel('SCG Z (3.91Hz)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, df_task['scgY_std'].values, color=color_dict[sync_color_dict['scg_y']])
    axes[i_axes].set_ylabel('SCG Y (std)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)   
    i_axes = i_axes + 1
    
    axes[i_axes].plot(ts, df_task['scgZ_std'].values, color=color_dict[sync_color_dict['scg_z']])
    axes[i_axes].set_ylabel('SCG Z (std)\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)   
    i_axes = i_axes + 1
    
    
    # sig_filt = get_padded_filt(df_task['ECG'].values, filter_padded=1, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=FS_RESAMPLE)

    axes[i_axes].plot(ts, get_padded_filt(df_task['ECG'].values, filter_padded=1, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=Fs), color=color_dict[sync_color_dict['ECG']])
    axes[i_axes].set_ylabel('ECG\n{}'.format(unit_dict['ecg']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, df_task['HR_patch'].values, color=color_dict[sync_color_dict['ECG']])
    axes[i_axes].set_ylabel('HR_patch\n{}'.format(unit_dict['HR']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    
    
#     axes[18].plot(ts, v_arr*df_task['VT_cosmed'])
#     axes[18].set_ylabel('resp.\n{}'.format('ml/kg'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

    axes[i_axes].plot(ts, df_task['resp_cosmed'].values)
    axes[i_axes].set_ylabel('resp.\n{}'.format('ml/kg'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, df_task['EErq_cosmed'])
    axes[i_axes].set_ylabel('EE\n{}'.format('kcal/min'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, df_task['VT_cosmed'].values)
    axes[i_axes].set_ylabel('VT\n{}'.format('ml/breath'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    axes[i_axes].plot(ts, df_task['RR_cosmed'].values)
    axes[i_axes].set_ylabel('RR\n{}'.format('breath/min'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)
    i_axes = i_axes + 1

    
#     ax4.plot(ts, get_padded_filt(df_task['ECG'].values, filter_padded=1, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=FS_RESAMPLE))
#     ax4.set_ylabel('ECG\n{}'.format(unit_dict['ecg']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

#     ax5.plot(ts, get_padded_filt(df_task['accelZ'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs))
#     ax5.set_ylabel('SCG\n{}'.format(unit_dict['accel']), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

#     ax6.plot(ts, v_arr*df_task['VT_cosmed'])
#     ax6.set_ylabel('resp.\n{}'.format('ml/kg'), rotation = 0,  va='center', ha='center',  labelpad=100, fontsize=fontsize)

    
    
    
    
    # ax7.plot(ts, df_task['VT_cosmed'])
    scale = 1
    
    for ax in axes:
        ax.set_xlim(ts.min(),ts.max()/scale)
        ax_no_top_right(ax)


#     ax1.set_xlim(ts.min(),ts.max()/scale)
#     ax2.set_xlim(ts.min(),ts.max()/scale)
#     ax3.set_xlim(ts.min(),ts.max()/scale)
#     ax4.set_xlim(ts.min(),ts.max()/scale)
#     ax5.set_xlim(ts.min(),ts.max()/scale)
#     ax6.set_xlim(ts.min(),ts.max()/scale)

#     ax6.set_xlabel('time (sec)', fontsize=fontsize)
    axes[-1].set_xlabel('time (sec)', fontsize=fontsize)

    # remove some borders (top and right)
#     ax1.spines['right'].set_visible(False)
#     if i==0:
    axes[0].spines['top'].set_visible(False)
#     ax1.spines['top'].set_visible(False)

    fig.tight_layout()

    fig.subplots_adjust(wspace=0, hspace=0)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if fig_name is None:
            fig_name = 'physio_sigs'
        else:
            fig_name = fig_name

        fig.savefig(outputdir + fig_name,bbox_inches='tight', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
    plt.show()


        
        

        
def get_spectral_features(df_task, i_peaks, accel_name='accelZ', Fs=250, smoothen_feature=True):

    if accel_name=='scgXYZ':
        scg_name = accel_name
        scg = df_task[accel_name].values
        # scg = get_padded_filt(df_task[accel_name].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
    else:
        scg_name = 'scg' + accel_name[-1]
        scg = get_padded_filt(df_task[accel_name].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)

        #     scgZ = df_task['scgZ'].values

    # fig, ax = plt.subplots(1,1,figsize=(30,2), dpi=80)
    # # plt.plot(df_task['accelZ'].values)
    # ax.plot(scgZ)

    beat_size = 0.5 # sec

    scg_beats, i_scg_beats = beat_segmentation(scg, i_peaks, start_offset=0, end_offset=int(beat_size*Fs))
    
    # scg_beats = get_ensemble_beats(scg_beats, N_enBeats=4)

    # reject bad SCG beats
    sqi_xcorr = get_sqi(scg_beats, get_max_xcorr, scg_beats.mean(axis=1))
    
    sqi_deviation = np.abs(sqi_xcorr-np.median(sqi_xcorr))
    # mask_sqi = (sqi_deviation <= sqi_deviation.std()*6) & (sqi_xcorr >= 0.6)
    mask_sqi = (sqi_deviation <= sqi_deviation.std()*5) & (sqi_xcorr >= 0.5)
    
    
    print('\t\t [SCG beats] reject {:.2f}% of the beats'.format((1-mask_sqi.mean())*100))

    
#     print(mask_sqi.mean())
#     mask_sqi = (np.abs(sqi_xcorr-sqi_xcorr.mean()) <= sqi_xcorr.std()*5) & (sqi_xcorr >= 0.7)
#     print(scg_beats.shape)
    freq, _ = get_PSD_beats(scg_beats, Fs)

    scg_beats = scg_beats[:, mask_sqi]
    i_scg_beats = i_scg_beats[mask_sqi]
    
#     print(scg_beats.shape)
    
    # t_beat = np.arange(scg_beats.shape[0])/Fs

    # plt.plot(t_beat, scg_beats, alpha=0.1, color='steelblue')
    # plt.show()

#     scg_beats = get_ensemble_beats(scg_beats, N_enBeats=4)
    
    


    # t_beat = np.arange(scg_beats.shape[0])/Fs
    # plt.plot(t_beat, scg_beats, alpha=0.1, color='steelblue')
    # plt.show()

#     freq, Pxx = get_PSD_beats(scg_beats, Fs)
    _, Pxx = get_PSD_beats(scg_beats, Fs)
#     freq


#     scg_freq_range = {
#         '0_3Hz': [0,3],
#         '3_6Hz': [3,6],
#         '6_9Hz': [6,9],
#         '9_12Hz': [9,12],
#         '12_15Hz': [12,15],
#         '15_18Hz': [15,18],
#         '18_21Hz': [18,21],
#         '21_24Hz': [21,24],
#     }

    smoothing_dur = 5 # second

    features_spectral = {}

    for i_f, f in enumerate(freq):
        

        
        if f > FILT_SCG[1]:
            continue
        key = '{}_{:.2f}Hz'.format(scg_name, f)
        
        if scg_beats.shape[1]==0:
            Pxx_freq_interp = np.empty((df_task.shape[0]))
            Pxx_freq_interp[:] = np.nan
#             features_spectral[key] = Pxx_freq_interp
            
        else:

            Pxx_freq = Pxx[i_f, :]
            t_df = np.arange(df_task.shape[0])/Fs
            Pxx_freq_interp = np.interp(t_df, i_scg_beats/Fs, Pxx_freq)

    #         if i_f ==4:
    #             fig, (ax1, ax2) = plt.subplots(2,1, figsize=(20,14), dpi=70)
    #             ax1.scatter(i_scg_beats/Fs, Pxx_freq, c='blue')
    #             ax1.plot(t_df, Pxx_freq_interp, c='red')

            if smoothen_feature:
                Pxx_freq_interp = get_padded_filt_DSwrapper(Pxx_freq_interp, filter_padded=5, lowcutoff=None, highcutoff=0.05, Fs=Fs)    
#         if i_f ==4:
#             ax1.plot(t_df, Pxx_freq_interp, c='orange')
#             ax2.plot(scg_beats)
#             plt.show()

        
#         if i_f <5:
#             print('hihi')
#             fig, ax = plt.subplots(1,1, figsize=(5,5), dpi=50)

#             ax.scatter(i_scg_beats/Fs, Pxx_freq)

#             ax.plot(t_df, Pxx_freq_interp)
#             plt.show()

        features_spectral[key] = Pxx_freq_interp
#         print(key, Pxx_freq_interp)
    
    
    if scg_beats.shape[1]==0:
        scg_std_interp = np.empty((df_task.shape[0]))
        scg_std_interp[:] = np.nan
    else:
        scg_std = scg_beats.std(axis=0)
        scg_std_interp = np.interp(t_df, i_scg_beats/Fs, scg_std)
        if smoothen_feature:
            scg_std_interp = get_padded_filt_DSwrapper(scg_std_interp, filter_padded=5, lowcutoff=None, highcutoff=0.05, Fs=Fs)

    features_spectral['{}_std'.format(scg_name)] = scg_std_interp

#     sys.exit() 

#     scg_freq_range = {
#         '0_3Hz': [0,3],
#         '3_6Hz': [3,6],
#         '6_9Hz': [6,9],
#         '9_12Hz': [9,12],
#         '12_15Hz': [12,15],
#         '15_18Hz': [15,18],
#         '18_21Hz': [18,21],
#         '21_24Hz': [21,24],
#     }

#     features_spectral = {}

#     for key, freq_range in scg_freq_range.items():
#         mask = (freq>=freq_range[0]) & (freq<freq_range[-1])
#         Pxx_freq = Pxx[mask, :].mean(axis=0)

#         t_df = np.arange(df_task.shape[0])/Fs
#         Pxx_freq_interp = np.interp(t_df, i_scg_beats/Fs, Pxx_freq)

#     #     fig, ax = plt.subplots(1,1,figsize=(10,2), dpi=80)

#     #     ax.scatter(i_scg_beats/Fs, Pxx_freq, alpha=0.3)
#     #     ax.plot(t_df, Pxx_freq_interp, alpha=0.3)

#         features_spectral[key] = Pxx_freq_interp

    return features_spectral
        
        
        
# def ppg_filter(sig_raw, highcutoff, Fs):

#     # get time vector
#     ts = np.arange(sig_raw.shape[0])/Fs
#     # DS from Fs to Fs/5 = 100Hz
#     downsample_factor = 5

#     # estiamte DS signal
#     Fs_DS = Fs / downsample_factor # 100 Hz
#     ts_DS = ts[::downsample_factor]
#     sig_DS = np.interp(ts_DS, ts, sig_raw)    

#     # use padded signal and filter
#     filter_padded = 1 # sec
#     offset = int(filter_padded*Fs) # reversely pad filter_padded sec to beginning and end of the signal
#     sig_DS = np.r_[sig_DS[0:offset][::-1], sig_DS, sig_DS[-offset:][::-1]]
#     sig_filt = butter_filter('low', sig_DS, highcutoff, Fs_DS)
    
#     # recover the filtered signal so it's legnth eqauls to that of sig_raw
#     sig_filt = sig_filt[offset:-offset]
#     sig_filt = np.interp(ts, ts_DS, sig_filt)  

#     return sig_filt

# def ppg_filter(sig_raw, Fs):
#     ts= np.arange(sig_raw.shape[0])/Fs

#     sig_filt_DC = get_padded_filt_DSwrapper(sig_raw, filter_padded=1, lowcutoff=None, highcutoff=0.1, Fs=Fs)
#     sig_filt = get_padded_filt(sig_raw, filter_padded=1, lowcutoff=0.08, highcutoff=1, Fs=Fs)
#     sig_norm = -sig_filt/sig_filt_DC
#     return sig_norm


def ppg_filter(sig_raw, Fs, mode='resp'):
#     ts= np.arange(sig_raw.shape[0])/Fs
    
    sig_filt_DC = get_padded_filt_DSwrapper(sig_raw, filter_padded=1, lowcutoff=None, highcutoff=0.1, Fs=Fs)
    if mode=='resp':
#         print(0.08,1)
        sig_filt = get_padded_filt(sig_raw, filter_padded=1, lowcutoff=0.08, highcutoff=1, Fs=Fs)
        sig_norm = -sig_filt/sig_filt_DC
    elif mode=='cardiac':
        sig_filt = get_padded_filt(sig_raw, filter_padded=1, lowcutoff=FILT_PPG[0], highcutoff=FILT_PPG[1], Fs=Fs)
        sig_norm = -sig_filt/sig_filt_DC
    return sig_norm

def filter_DFcolumns(df_sub, Fs):
    
#     df_sub['ECG'] = get_padded_filt(df_sub['ECG'].values, filter_padded=1, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=Fs)
#     df_sub['accelX'] = get_padded_filt(df_sub['accelX'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
#     df_sub['accelY'] = get_padded_filt(df_sub['accelY'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
#     df_sub['accelZ'] = get_padded_filt(df_sub['accelZ'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)

    df_sub['ECG_filt'] = get_padded_filt(df_sub['ECG'].values, filter_padded=1, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=Fs)
    df_sub['accelX_filt'] = get_padded_filt(df_sub['accelX'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs)
    df_sub['accelY_filt'] = get_padded_filt(df_sub['accelY'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs)
    df_sub['accelZ_filt'] = get_padded_filt(df_sub['accelZ'].values, filter_padded=5, lowcutoff=FILT_RESP[0], highcutoff=FILT_RESP[1], Fs=Fs)

    df_sub['scgX'] = get_padded_filt(df_sub['accelX'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
    df_sub['scgY'] = get_padded_filt(df_sub['accelY'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
    df_sub['scgZ'] = get_padded_filt(df_sub['accelZ'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)


    df_sub['ppg_g_1_cardiac'] = ppg_filter(df_sub['ppg_g_1'].values, Fs, mode='cardiac')
    df_sub['ppg_g_2_cardiac'] = ppg_filter(df_sub['ppg_g_2'].values, Fs, mode='cardiac')
    df_sub['ppg_r_1_cardiac'] = ppg_filter(df_sub['ppg_r_1'].values, Fs, mode='cardiac')
    df_sub['ppg_r_2_cardiac'] = ppg_filter(df_sub['ppg_r_2'].values, Fs, mode='cardiac')
    df_sub['ppg_ir_1_cardiac'] = ppg_filter(df_sub['ppg_ir_1'].values, Fs, mode='cardiac')
    df_sub['ppg_ir_2_cardiac'] = ppg_filter(df_sub['ppg_ir_2'].values, Fs, mode='cardiac')
    
    df_sub['ppg_g_1_resp'] = ppg_filter(df_sub['ppg_g_1'].values, Fs, mode='resp')
    df_sub['ppg_g_2_resp'] = ppg_filter(df_sub['ppg_g_2'].values, Fs, mode='resp')
    df_sub['ppg_r_1_resp'] = ppg_filter(df_sub['ppg_r_1'].values, Fs, mode='resp')
    df_sub['ppg_r_2_resp'] = ppg_filter(df_sub['ppg_r_2'].values, Fs, mode='resp')
    df_sub['ppg_ir_1_resp'] = ppg_filter(df_sub['ppg_ir_1'].values, Fs, mode='resp')
    df_sub['ppg_ir_2_resp'] = ppg_filter(df_sub['ppg_ir_2'].values, Fs, mode='resp')
    
    
    return df_sub



def get_data_condensed(df_window, list_input, Fs, FS_RESAMPLE_DL):
    # down sample data to reduce data dimension
    
    t_arr = np.arange(df_window.shape[0])/Fs
    t_start = t_arr.min()
    t_end = t_arr.max()
    time_interp = np.arange(my_ceil(t_start, decimal=-3)*FS_RESAMPLE_DL, my_floor(t_end, decimal=-3)*FS_RESAMPLE_DL)/FS_RESAMPLE_DL

    data_condensed = []
    for input_name in list_input:
        sig = df_window[input_name].values
            # do ECG
#         if 'ECG' in input_name:
#             sig_filt = get_padded_filt(df_window[input_name].values, filter_padded=1, lowcutoff=1, highcutoff=FILT_ECG[1], Fs=Fs)
#         elif 'accel' in input_name:
#             sig_filt = get_padded_filt(df_window[input_name].values, filter_padded=1, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
#         else:
#             sig_filt = df_window[input_name].values

        sig_resample = np.interp(time_interp, t_arr, sig)
        data_condensed.append(sig_resample)
    data_condensed = np.stack(data_condensed, axis=-1)

    return data_condensed

def get_feature_condensed(df_window, list_feature):
    feature = df_window[list_feature].values
    feature_condensed = feature.mean(axis=0)
    return feature_condensed
#     return feature

def get_label_condensed(df_window, list_output):
    label = df_window[list_output].values
    label_condensed = label.mean(axis=0)
#     label_condensed = np.r_[label[:,:-1].mean(axis=0),  np.asarray([label[0,-1][:-2]])]
    return label_condensed
#     return label


# def get_data_condensed(df_window, list_input, Fs, FS_RESAMPLE_DL):
#     # down sample data to reduce data dimension
    
#     t_arr = np.arange(df_window.shape[0])/Fs
#     t_start = t_arr.min()
#     t_end = t_arr.max()
#     time_interp = np.arange(my_ceil(t_start, decimal=-3)*FS_RESAMPLE_DL, my_floor(t_end, decimal=-3)*FS_RESAMPLE_DL)/FS_RESAMPLE_DL

#     data_condensed = []
#     for input_name in list_input:
#         data = df_window[input_name].values
#             # do ECG
#         if 'ECG' in input_name:
#             sig_filt = get_padded_filt(df_window[input_name].values, filter_padded=1, lowcutoff=1, highcutoff=FILT_ECG[1], Fs=Fs)
#         elif 'accel' in input_name:
#             sig_filt = get_padded_filt(df_window[input_name].values, filter_padded=1, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
#         else:
#             sig_filt = df_window[input_name].values

#         sig_filt_resample = np.interp(time_interp, t_arr, sig_filt)
#         data_condensed.append(sig_filt_resample)
#     data_condensed = np.stack(data_condensed, axis=-1)

#     return data_condensed

# def get_feature_condensed(df_window, list_feature):
#     feature = df_window[list_feature].values
#     feature_condensed = feature.mean(axis=0)
#     return feature_condensed

# def get_label_condensed(df_window, list_output):
#     label = df_window[list_output].values
#     label_condensed = label.mean(axis=0)
# #     label_condensed = np.r_[label[:,:-1].mean(axis=0),  np.asarray([label[0,-1][:-2]])]
#     return label_condensed

def get_meta_condensed(df_window, list_meta):
    meta_condensed = df_window[list_meta].values[0,:]
    return meta_condensed




def get_Nf_dict(df_task, QRS_detector_dict_patch, Fs):
    
#     df_task = filter_DFcolumns(df_task.copy(), Fs)

    # TODO: combine these cells, normalize signals, compute % change in EE, correlate scg features to TV or EE
    Nf_dict = {}

    Nf = get_Nf_ECGresp(df_task['ECG_filt'].values, QRS_detector_dict_patch, Fs)
    Nf_dict['ECG'] = Nf

    Nf = get_Nf_SCGresp(df_task['scgZ'].values, QRS_detector_dict_patch, Fs)
    Nf_dict['accelZ'] = Nf

    for key in df_task.keys():
        if ('ppg' in key) & ('cardiac' in key):
            ppg_name = key
            Nf, _ = get_Nf_PPGresp_AM(df_task, ppg_name, QRS_detector_dict_patch, Fs)
            Nf_dict[ppg_name]= Nf

    return Nf_dict

def get_Nf_dict2(df_task, QRS_detector_dict_patch, Fs):
    
    Nf_dict = {}

    Nf = df_task['ECG_filt'].values.std()
    Nf_dict['ECG'] = Nf

    Nf = df_task['scgZ'].values.std()
    Nf_dict['accelZ'] = Nf

    for key in df_task.keys():
        if ('ppg' in key) & ('cardiac' in key):
            ppg_name = key
            Nf = df_task[ppg_name].values.std()
            Nf_dict[ppg_name]= Nf

    return Nf_dict

def get_Nf_dict3(df_task, QRS_detector_dict_patch, Fs):
    
    Nf_dict = {}

    Nf = 1
#     Nf = df_task['ECG_filt'].values.std()
    Nf_dict['ECG'] = Nf

#     Nf = df_task['scgZ'].values.std()
    Nf_dict['accelZ'] = Nf

    for key in df_task.keys():
        if ('ppg' in key) & ('cardiac' in key):
            ppg_name = key
#             Nf = df_task[ppg_name].values.std()
            Nf_dict[ppg_name]= Nf

    return Nf_dict

def get_Nf_ECGresp(ecg_filt, QRS_detector_dict_patch, Fs):
    # this function will extract the respiratory signal and compute the interdecile range (10-90%), which will be the normalizing factor for all task for the subject
    inspect_resp = False
    # 0. prepare for i_beats
    i_S_peaks = QRS_detector_dict_patch['i_S_peaks']
    i_R_peaks = QRS_detector_dict_patch['i_R_peaks']
    surrogate_dict = {}

    # 1. compute resp. surrogate
#     ecg_filt = get_padded_filt(df_task['ECG'].values, filter_padded=5, lowcutoff=FILT_ECG[0], highcutoff=FILT_ECG[1], Fs=Fs)
    t_ecg = np.arange(ecg_filt.shape[0])/Fs
    ECG_AMr = ECG_AM_extraction(t_ecg, ecg_filt, i_R_peaks, 5, Fs)
    surrogate_dict['ECGr'] = ECG_AMr    
    ECG_AMs = ECG_AM_extraction(t_ecg, -ecg_filt, i_S_peaks, 5, Fs)
    surrogate_dict['ECGs'] = ECG_AMs

    if inspect_resp:
        i_start = 0*Fs
        i_end = i_start + 50*Fs

        plt.figure(figsize=(10,3),dpi=100)

        plt.plot(-ecg_filt[i_start:i_end])
        plt.plot(surrogate_dict['ECG']['raw'][i_start:i_end],'r')
        plt.plot(surrogate_dict['ECG']['smooth'][i_start:i_end],'b')
        # plt.plot(ECG_AMs['filt'][i_start:i_end],'g')
        #     plt.plot(surrogate_dict[sur_name]['normed'][i_start:i_end])

    # 2. compute the interdecile range
    _, _, Nf = get_interdecile_range(surrogate_dict['ECGr']['smooth']-surrogate_dict['ECGs']['smooth'], confidence_interv=90) # get range of 10%-90%
#     print(interdecile_range)
    
    return Nf


def get_Nf_SCGresp(scgZ, QRS_detector_dict_patch, Fs):
    # this function will extract the respiratory signal and compute the interdecile range (10-90%), which will be the normalizing factor for all task for the subject

    inspect_resp = False
    # 0. prepare for i_beats
    i_S_peaks = QRS_detector_dict_patch['i_S_peaks']
    i_R_peaks = QRS_detector_dict_patch['i_R_peaks']
    surrogate_dict = {}

    scgZ_beats, i_scg_peaks = beat_segmentation(scgZ, i_R_peaks, start_offset=0, end_offset=int(0.2*Fs))
    i_aos, data_aos = SCG_ao(scgZ_beats, fs=Fs)
    i_fiducials = (i_aos, data_aos)
    

    # 1. compute resp. surrogate
    t_scg = np.arange(scgZ.shape[0])/Fs
    surrogate_dict['scgZ'] = SCG_AM_extraction(t_scg, scgZ_beats, i_fiducials, i_R_peaks, 5, Fs=Fs)

    if inspect_resp:
        i_start = 0*Fs
        i_end = i_start + 50*Fs

        plt.figure(figsize=(10,3),dpi=100)

        plt.plot(scgZ[i_start:i_end])
        plt.plot(surrogate_dict['scgZ']['raw'][i_start:i_end],'r')
        plt.plot(surrogate_dict['scgZ']['smooth'][i_start:i_end],'b')

    # 2. compute the interdecile range
    # double the amplitude of scgZ ao point since it's too hard to detect IVC
    _, _, Nf = get_interdecile_range(surrogate_dict['scgZ']['smooth']*2, confidence_interv=90) # get range of 10%-90%
    
    return Nf


def get_Nf_PPGresp_AM(df_task, ppg_name, QRS_detector_dict_patch, Fs):

    t_ppg = np.arange(df_task.shape[0])/Fs

    # ppg_name = 'ppg_ir_1_cardiac'
    indices_beats_max, ppg_max, indices_beats_min, ppg_min, i_ppg_peaks, ppg = get_ppg_fiducials(df_task, ppg_name, QRS_detector_dict_patch, Fs)
    surrogate_dict = {}

    surrogate_dict['PPGmax'] = PPG_AM_extraction(t_ppg, indices_beats_max, ppg_max, 5, Fs)
    surrogate_dict['PPGmin'] = PPG_AM_extraction(t_ppg, indices_beats_min, ppg_min, 5, Fs)

    _, _, Nf = get_interdecile_range(surrogate_dict['PPGmax']['smooth']-surrogate_dict['PPGmin']['smooth'], confidence_interv=90) # get range of 10%-90%
    
    PPG_AM = surrogate_dict['PPGmax']['filt']-surrogate_dict['PPGmin']['filt']
    return Nf, PPG_AM

def get_Nf_PPGresp_BW(ppg, Fs):
    _, _, Nf = get_interdecile_range(ppg, confidence_interv=90) # get range of 10%-90%
    return Nf

    
def get_df_resp(t_sig, df_task, QRS_detector_dict_patch, Fs, filter_padded=5):
    surrogate_dict = {}


    surrogate_dict['time'] = t_sig
    # ECG AM
    surrogate_dict['ECG_SR'] = ECG_SR_extraction(t_sig, QRS_detector_dict_patch['ecg_beats_leading'], QRS_detector_dict_patch['i_beat_peaks_leading'], filter_padded, Fs, offset_SR=QRS_detector_dict_patch['offset_SR'])['filt']
    surrogate_dict['ECG_AMr'] = ECG_AM_extraction(t_sig, QRS_detector_dict_patch['ecg_filt1'], QRS_detector_dict_patch['i_R_peaks'], filter_padded, Fs)['filt']
    surrogate_dict['ECG_AMs'] = ECG_AM_extraction(t_sig, -QRS_detector_dict_patch['ecg_filt1'], QRS_detector_dict_patch['i_S_peaks'], filter_padded, Fs)['filt']
    # compute ECGpt_AM (PT peaks)
    surrogate_dict['ECG_AMpt'] = ECG_AMpt_extraction(t_sig, QRS_detector_dict_patch['ecg_PT'], QRS_detector_dict_patch['i_PT_peaks'], filter_padded, Fs)['filt']

    # compute ECG_FM
    surrogate_dict['ECG_FM'] = ECG_FM_extraction(t_sig, QRS_detector_dict_patch['i_R_peaks'], filter_padded, Fs)['filt']

    # compute SCG_AM (AO amplitude)
    # 0. prepare for i_beats
#     i_S_peaks = QRS_detector_dict_patch['i_S_peaks']
#     i_R_peaks = QRS_detector_dict_patch['i_R_peaks']
    # surrogate_dict = {}

    scgX = df_task['scgX'].values
    scgY = df_task['scgY'].values
    scgZ = df_task['scgZ'].values
    scgZ_beats, i_scg_peaks = beat_segmentation(scgZ, QRS_detector_dict_patch['i_R_peaks'], start_offset=0, end_offset=int(0.2*Fs))
    i_aos, data_aos = SCG_ao(scgZ_beats, fs=Fs)
    i_fiducials = (i_aos, data_aos)

    # 1. compute resp. surrogate
    # t_scg = np.arange(scgZ.shape[0])/Fs
    surrogate_dict['SCG_AM'] = SCG_AM_extraction(t_sig, scgZ_beats, i_fiducials, QRS_detector_dict_patch['i_R_peaks'], filter_padded, Fs=Fs)['filt']

    # compute SCG_AMpt (envelope peaks)
    
#     SCG_AMpt_dict = SCG_AMpt_extraction(t_sig, scgZ, filter_padded, Fs)
    

    
    
#     surrogate_dict['SCG_AMpt'] = SCG_AMpt_extraction(t_sig, scgZ, filter_padded, Fs)['filt']
    # use R peaks to better localize SCG smoothed peaks
#     print(QRS_detector_dict_patch['i_R_peaks'])


#     scgM = (df_task['accelX_filt'].values**2 + df_task['accelY_filt'].values**2 + df_task['accelZ_filt'].values**2) ** 0.5

    scgXYZ = df_task[['scgX', 'scgY', 'scgZ']].values

    surrogate_dict['SCGxyz_AMpt'] = SCG_AMpt_extraction(t_sig, scgXYZ, filter_padded, Fs, i_R_peaks=QRS_detector_dict_patch['i_R_peaks'], N_channels=scgXYZ.shape[1])['filt']

#     surrogate_dict['SCG_AMpt'] = SCG_AMpt_extraction(t_sig, scgZ, filter_padded, Fs)['filt']
    surrogate_dict['SCG_AMpt'] = SCG_AMpt_extraction(t_sig, scgZ, filter_padded, Fs, i_R_peaks=QRS_detector_dict_patch['i_R_peaks'])['filt']
    surrogate_dict['SCGx_AMpt'] = SCG_AMpt_extraction(t_sig, scgX, filter_padded, Fs, i_R_peaks=QRS_detector_dict_patch['i_R_peaks'])['filt']
    surrogate_dict['SCGy_AMpt'] = SCG_AMpt_extraction(t_sig, scgY, filter_padded, Fs, i_R_peaks=QRS_detector_dict_patch['i_R_peaks'])['filt']
    

    surrogate_dict['accelX_resp'] = df_task['accelX_filt'].values
    surrogate_dict['accelY_resp'] = df_task['accelY_filt'].values
    surrogate_dict['accelZ_resp'] = df_task['accelZ_filt'].values

    # Accel BW
    # surrogate_dict['ACCEL_BW'] = df_task['accelZ_filt'].values

    # PPG Resp
    # surrogate_dict['ACCEL_BW'] = df_task['accelZ_filt'].values
    for key in df_task.keys():
        if ('resp' in key) & ('ppg' in key):
            surrogate_dict[key] = df_task[key].values
            
        if ('cardiac' in key) & ('ppg' in key):
            indices_beats_max, ppg_max, indices_beats_min, ppg_min, i_ppg_peaks, ppg = get_ppg_fiducials(df_task, key, QRS_detector_dict_patch, Fs)
            surrogate_dict[key+'_max'] = PPG_AM_extraction(t_sig, indices_beats_max, ppg_max, filter_padded, Fs)['filt']
            surrogate_dict[key+'_min'] = PPG_AM_extraction(t_sig, indices_beats_min, ppg_min, filter_padded, Fs)['filt']

    surrogate_dict['RR_cosmed'] = df_task['RR_cosmed'].values
    surrogate_dict['resp_cosmed'] = df_task['resp_cosmed'].values
    
    df_resp = pd.DataFrame(surrogate_dict)
    
    return df_resp




