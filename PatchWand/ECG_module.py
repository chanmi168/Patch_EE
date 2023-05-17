import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline
import sys

import numpy as np

from segmentation import *
from setting import *
from filters import *
from preprocessing import *
from SQI_module import *

# TODO: add documentation

def get_smooth(data, N=5):
    # padding prevent edge effect
    data_padded = np.pad(data, (N//2, N-1-N//2), mode='edge')
    data_smooth = np.convolve(data_padded, np.ones((N,))/N, mode='valid') 
    return data_smooth

def find_peaks_ECG(ecg_raw, width_QRS, filter_padded, Fs, param_dict, hr_max=None):
    
    thre_factor = param_dict['thre_factor']
    dist_factor = param_dict['dist_factor']
    # step 1. filter ECG (passband = 10-30Hz)
    
    ecg_filt1 = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=1, highcutoff=30, Fs=Fs)
    ecg_filt = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=10, highcutoff=30, Fs=Fs)

    # step 2 (WIP). remove outliers (use SQI)

    # step 3. differentiate
    ecg_diff = np.gradient(ecg_filt)

    # step 4. square
    ecg_sqr = ecg_diff**2

    # step 5. moving average (we chose 150ms as suggested in Pan and Tompkins' paper)
    # Generally, the width of the window should be approximately the same as the
    # widest possible QRS complex.
    width_QRS = 0.15 # sec
    ecg_smooth = get_smooth(ecg_sqr, N=int(Fs*width_QRS))
    window_legnth = int(Fs*width_QRS/2)

    # step 6. peak detection
    if hr_max is None:
        hr_max = label_range_dict['HR'][1]
    # no RR wave can occur in less than 200 msec distance
    # distance_min = 0.2 * FS_ECG # (1000 samples/s) / (120 beats/min) * (60s/min)
#     print(hr_max)
#     distance_min = int(Fs / hr_max * 60 * 0.8)# (1000 samples/s) / (120 beats/min) * (60s/min)
    distance_min = int(Fs / hr_max * 60 * dist_factor)# (1000 samples/s) / (120 beats/min) * (60s/min)
#     print(distance_min)

    
    i_peaks, _ = find_peaks(ecg_smooth, distance=distance_min)
#     print(i_peaks)
    indices_CI = arg_CI(ecg_smooth[i_peaks], confidence_interv=75)
#     print(indices_CI)
    thre = get_smooth(ecg_smooth[i_peaks[indices_CI]], N=5)*thre_factor
    thre = np.interp(np.arange(ecg_smooth.shape[0]), i_peaks[indices_CI], thre)
    i_peaks, _ = find_peaks(ecg_smooth, distance=distance_min, height=thre)

    ecg_dict = {
        'ecg_filt1': ecg_filt1,
        'ecg_filt': ecg_filt,
        'ecg_diff': ecg_diff,
        'ecg_sqr': ecg_sqr,
        'ecg_smooth_pks': ecg_smooth,
        'Fs': Fs,
        'window_legnth': window_legnth,
        'thre': thre,
    }
    
    return i_peaks, ecg_dict


def calibrate_peaks(i_peaks, sig, Fs, window_legnth):
    
    i_calibrate = np.zeros(i_peaks.shape)
    
#     window_legnth = int(Fs*width_QRS/2)
#     window_legnth = int(Fs*width_QRS)
    start_offset = -window_legnth
    end_offset = window_legnth

    for i, i_peak in enumerate(i_peaks):
        i_start = i_peak + start_offset
        i_end = i_peak + end_offset

        if i_start < 0 or i_end > sig.shape[0]:
            i_calibrate[i] = i_peak
            continue

        i_max = sig[i_start:i_end].argmax()
        i_calibrate[i] = i_peak + i_max - window_legnth

    return i_calibrate.astype(int)

def i_R_peaks_filter(ecg_filt, i_R_peaks, Fs):
    hr = 1/np.gradient(i_R_peaks/Fs)*60
    indices_good = arg_goodhr(hr)
    
    return i_R_peaks[indices_good], ecg_filt[i_R_peaks[indices_good]]

def unpack_ECG(i_PT_peaks, ecg_dict, offset_SR):    
    Fs = ecg_dict['Fs']
    window_legnth = ecg_dict['window_legnth']

    ecg_PT = ecg_dict['ecg_smooth_pks']
    ecg_filt = ecg_dict['ecg_filt']
    

    i_R_peaks = calibrate_peaks(i_PT_peaks, ecg_filt, Fs, window_legnth)

    i_S_peaks = calibrate_peaks(i_PT_peaks, -ecg_filt, Fs, window_legnth)

    # 3.1 Make sure R peaks extracted are legit (Edit: 3/23)
    i_PT_peaks, _ = i_R_peaks_filter(ecg_PT, i_PT_peaks, Fs)
    i_R_peaks = remove_ECGpeaks_anomalies(i_R_peaks, ecg_dict['ecg_filt1'], Fs)
    i_S_peaks = remove_ECGpeaks_anomalies(i_S_peaks, -ecg_dict['ecg_filt1'], Fs)


    # 4. beats segmentation
#         ecg_beats, i_beat_peaks = beat_segmentation(ecg_dict['ecg_filt1'], i_R_peaks, start_offset=0, end_offset=500)
    ecg_beats_R, i_beat_peaks_R = beat_segmentation(ecg_dict['ecg_filt1'], i_R_peaks, start_offset=0, end_offset=window_legnth)
#     ecg_beats_PT, i_beat_peaks_PT = beat_segmentation(ecg_dict['ecg_filt1'], i_PT_peaks, start_offset=-offset_SR, end_offset=offset_SR)
    ecg_beats_PT, i_beat_peaks_PT = beat_segmentation(ecg_dict['ecg_filt1'], i_PT_peaks, start_offset=-offset_SR, end_offset=window_legnth)
    ecg_beats_S, i_beat_peaks_S = beat_segmentation(ecg_dict['ecg_filt1'], i_S_peaks, start_offset=0, end_offset=window_legnth)

    # 4.1 R/S peak switcher (pick either the +/- peak)
    if np.abs(ecg_beats_R[0,:].mean()) > np.abs(ecg_beats_S[0,:]).mean():
        print('\t use R peaks')
        use_R_S = 'R'
        ecg_beats = ecg_beats_R
        i_beat_peaks = i_beat_peaks_R
        ecg_beats_leading, i_beat_peaks_leading = beat_segmentation(ecg_dict['ecg_filt1'], i_beat_peaks, start_offset=-offset_SR, end_offset=window_legnth)

    else:
        print('\t use S peaks')
        use_R_S = 'S'
        ecg_beats = ecg_beats_S
        i_beat_peaks = i_beat_peaks_S
        ecg_beats_leading, i_beat_peaks_leading = beat_segmentation(-ecg_dict['ecg_filt1'], i_beat_peaks, start_offset=-offset_SR, end_offset=window_legnth)

    return i_beat_peaks, ecg_beats, i_beat_peaks_leading, ecg_beats_leading, i_PT_peaks, ecg_PT, ecg_filt, i_beat_peaks_R, ecg_beats_R, i_beat_peaks_S, ecg_beats_S, use_R_S



def remove_ECGpeaks_anomalies(i_ECG_peaks, ECG_arr, Fs):
    # ECG_arr needs to be filtered
    
    outlier_peaks = []
    real_peaks = []
    N_buffer = 20 # 10 beats
    real_peaks = real_peaks + list(i_ECG_peaks[0:N_buffer])
    
#     HR_std_scale = 15
#     peak_std_scale = 3
    HR_std_scale = 20
    peak_std_scale = 7
    
    peak_abs = np.abs(ECG_arr[real_peaks[-N_buffer:]].mean())
#     HR_abs = (1/(np.gradient(real_peaks[-N_buffer:])/Fs)*60).mean()

    HR_std_threshold = np.std(1/(np.gradient(real_peaks[-N_buffer:])/Fs)*60)*HR_std_scale

    peak_std_threshold = np.std(ECG_arr[real_peaks[-N_buffer:]]/peak_abs)*peak_std_scale

    
 
    for i, i_peak in enumerate(i_ECG_peaks):

        if i < N_buffer:
            continue

        HR_i = 1 / ((i_peak - real_peaks[-1]) / Fs) * 60
        HR_i_1 = 1 / ((real_peaks[-1] - real_peaks[-2]) / Fs) * 60

        diff_HR_i = np.abs(HR_i_1 - HR_i)
        diff_ECG_value_i = np.abs(ECG_arr[i_peak] - ECG_arr[real_peaks[-1]]) / peak_abs
        
        
        if HR_i > label_range_dict['HR'][1]:
            outlier_peaks.append(i_peak)
            continue
            
        if (diff_HR_i > HR_std_threshold) & (diff_ECG_value_i > peak_std_threshold):
            outlier_peaks.append(i_peak)
            continue
        else:
            real_peaks.append(i_peak)

        HR_std_threshold = np.std(1/(np.gradient(real_peaks[-N_buffer:])/Fs)*60)*HR_std_scale
        
        peak_abs = np.abs(ECG_arr[real_peaks[-N_buffer:]].mean())
        peak_std_threshold = np.std(ECG_arr[real_peaks[-N_buffer:]]/peak_abs)*peak_std_scale

    i_ECG_peaks_corrected = np.asarray(real_peaks)
       
    return i_ECG_peaks_corrected

def get_IBI_max(i_beat_peaks):
    hr = 1/ (np.diff(i_beat_peaks)/Fs) * 60
    indices_CI = arg_CI(hr, confidence_interv=96)
    # max window legnth such that the whole pulse can be captured but the next beat won't leak into the current beat
    IBI_min = 1 / (hr[indices_CI].max()/60) * 1.2
    return IBI_min




def detect_QRS(hr, ECG, Fs, param_dict, hr_max=None):
    # 0. compute a good window size (IBI_min*1.4)
    # hr = df['HR_cosmed'].values
    indices_CI = arg_CI(hr, confidence_interv=96)
    IBI_min = 1 / (hr[indices_CI].max()/60)
    end_offset = int(IBI_min * 1.4 * Fs)

    # 1. compute R peaks
    width_QRS = label_range_dict['width_QRS']
    filter_padded = 1

    # 2. find peaks of patch ECG
    # i_PT_peaks, ecg_dict = find_peaks_ECG(df['ECG'].values, width_QRS, filter_padded, Fs)
    i_PT_peaks, ecg_dict = find_peaks_ECG(ECG, width_QRS, filter_padded, Fs, param_dict, hr_max=hr_max)
    offset_SR = int(0.2*Fs) # look at data 0.2s preceeding R peak

    # 3. calibrate PT peaks to R peaks
    i_beat_peaks, ecg_beats, i_beat_peaks_leading, ecg_beats_leading, i_PT_peaks, ecg_PT, ecg_filt, i_R_peaks, ecg_beats_R, i_S_peaks, ecg_beats_S, use_R_S = unpack_ECG(i_PT_peaks, ecg_dict, offset_SR)
#     _, _, _, _, _, _, _, i_R_peaks, _ = unpack_ECG(i_PT_peaks, ecg_dict)
    N_detected = i_PT_peaks.shape[0]
    N_kept = i_R_peaks.shape[0]
    
    return {
        'i_beat_peaks': i_beat_peaks,
        'ecg_beats': ecg_beats,
        'i_beat_peaks_leading': i_beat_peaks_leading,
        'ecg_beats_leading': ecg_beats_leading,
        
        'i_PT_peaks': i_PT_peaks,
        'ecg_PT': ecg_PT,
        'ecg_filt': ecg_filt,
        'ecg_filt1': ecg_dict['ecg_filt1'],
        'i_R_peaks': i_R_peaks,
        'ecg_beats_R': ecg_beats_R,
        'i_S_peaks': i_S_peaks,
        'ecg_beats_S': ecg_beats_S,
        'N_detected': N_detected,
        'N_kept': N_kept,
        'end_offset': end_offset,
        'offset_SR': offset_SR,
        
        'ecg_dict': ecg_dict,
        'use_R_S': use_R_S
        
    }



def get_IP(y1, y2):
    IP = y1.T @ y2 / np.sqrt((y2.T@y2) * (y1.T@y1))
    return IP

def get_innerproduct(beats, beats_template):
    '''
    This function (distance_method) computes the distance [inner product] between the each beat in beats and beats_template
    beats dim: (N,M)
    beats_template dim: (N,)
    IP_vector dim: (M,)
    '''
    
    IP_vector = np.zeros(beats.shape[1])
    for i_col in range(beats.shape[1]):
        IP_vector[i_col] = get_IP(beats_template, beats[:,i_col])

    return IP_vector



def get_hr(i_peaks, Fs):
    hr = 1/(np.diff(i_peaks/Fs)/60)
    hr = np.r_[hr[0], hr]
    return hr

# def plot_ECG_HR(ax_ECG, ax_HR, QRS_detector_dict_patch, Fs, beat_name='i_beat_peaks'):

#     # 1. plot ecg and detected beats
#     ECG_arr = QRS_detector_dict_patch['ecg_dict']['ecg_filt1']    
#     t_arr = np.arange(ECG_arr.shape[0])/Fs    
    
#     t_dur = t_arr[-1]-t_arr[0]
    
#     i_beat_peaks_patch = QRS_detector_dict_patch[beat_name][mask_sqi_patch][mask_hr_patch]
    
    
#     ax_ECG.plot(t_arr, ECG_arr,alpha=0.5)

#     ax_ECG.scatter(i_beat_peaks_patch/FS_RESAMPLE, ECG_arr[i_beat_peaks_patch],s=10, c='g', alpha=0.5, marker='o')
#     ax_ECG.plot(i_beat_peaks_patch/FS_RESAMPLE, ECG_arr[i_beat_peaks_patch], c='g', alpha=0.5, marker='o')
#     ax_ECG.set_xlim(t_arr.min(), t_arr.max())
#     ax_ECG.set_xlabel('time (s)')
#     ax_ECG.set_xticks(np.arange(t_arr.min(), t_arr.max(), 50))
    
    
#     if 'ts_hr' in QRS_detector_dict_patch:
# #         print("Key exists") 

#         # 2. plot derived HR
#         ts_hr_patch = QRS_detector_dict_patch['ts_hr']
#         hr_patch = QRS_detector_dict_patch['hr']

#         ax_HR.plot(ts_hr_patch, hr_patch,alpha=0.5, color='g')
#         ax_HR.scatter(ts_hr_patch, hr_patch,alpha=0.5,s=10, c='g')

#         ax_HR.set_xlabel('time (s)')
#         ax_HR.set_ylabel('hr (bpm)')
#         ax_HR.set_xlim(t_arr.min(), t_arr.max())
#         ax_HR.set_xticks(np.arange(t_arr.min(), t_arr.max(), 5))
        
#     else:
#         pass
# #         print("Key does not exist")

def QRS_detector_dict_diagnostics(QRS_detector_dict_patch, mask_sqi_patch, mask_hr_patch, beat_name='i_R_peaks',fig_name=None, outputdir=None, show_plot=False):

    
    fontsize = 20

    # 1. plot ecg and detected beats
    ECG_arr = QRS_detector_dict_patch['ecg_dict']['ecg_filt1']    
    t_arr = np.arange(ECG_arr.shape[0])/FS_RESAMPLE    
    
    t_dur = t_arr[-1]-t_arr[0]
    
    i_beat_peaks_patch = QRS_detector_dict_patch[beat_name][mask_sqi_patch][mask_hr_patch]
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(t_dur*0.5,5),dpi=60)

    ax1.plot(t_arr, ECG_arr,alpha=0.5)

    ax1.scatter(i_beat_peaks_patch/FS_RESAMPLE, ECG_arr[i_beat_peaks_patch],s=10, c='g', alpha=0.5, marker='o')
    ax1.plot(i_beat_peaks_patch/FS_RESAMPLE, ECG_arr[i_beat_peaks_patch], c='g', alpha=0.5, marker='o')
    ax1.set_xlim(t_arr.min(), t_arr.max())
    ax1.set_xlabel('time (s)')
    ax1.set_xticks(np.arange(t_arr.min(), t_arr.max(), 50))
    
    
    if 'ts_hr' in QRS_detector_dict_patch:
#         print("Key exists") 

        # 2. plot derived HR
        ts_hr_patch = QRS_detector_dict_patch['ts_hr']
        hr_patch = QRS_detector_dict_patch['hr']

        ax2.plot(ts_hr_patch, hr_patch,alpha=0.5, color='g')
        ax2.scatter(ts_hr_patch, hr_patch,alpha=0.5,s=10, c='g')

        ax2.set_xlabel('time (s)')
        ax2.set_ylabel('hr (bpm)')
        ax2.set_xlim(t_arr.min(), t_arr.max())
        ax2.set_xticks(np.arange(t_arr.min(), t_arr.max(), 5))
        
    else:
        pass
#         print("Key does not exist")
    
    if fig_name is None:
        fig_name='ECG_diagnostic'


    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + '{}.png'.format(fig_name),bbox_inches='tight', transparent=False)
    # close it so it doesn't consume memory
    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
def task_HR_detector(ECG_raw_patch, Fs, beat_choice = 'i_R_peaks', fig_name=None, outputdir=None, show_plot=True):

    debug = False
#     show_plot = True

    param_dict = {
        'thre_factor': 0.4, 
        'dist_factor': 0.8, 
    }

#     beat_choice = 'i_beat_peaks'
#     beat_choice = 'i_S_peaks'


    # step 1. get initial i_beats using fastet HR possible

    HR_ref =  np.ones(ECG_raw_patch.shape) * label_range_dict['HR'][1]
    QRS_detector_dict_patch = detect_QRS(HR_ref, ECG_raw_patch, Fs, param_dict)
    if debug:
        mask_fake = np.ones(QRS_detector_dict_patch[beat_choice].shape[0]).astype(bool)
        QRS_detector_dict_diagnostics(QRS_detector_dict_patch, mask_fake, mask_fake, beat_name=beat_choice,
                                   fig_name='ECG_patch_{}window'.format(j), outputdir=None, show_plot=True)


    HR_ref = get_hr(QRS_detector_dict_patch['i_beat_peaks'], Fs)

    # steo 2. get real estimate of i_beats
    indices_CI = arg_CI(HR_ref, confidence_interv=75)
    HR_ref = HR_ref[indices_CI]

    QRS_detector_dict_patch = detect_QRS(HR_ref, ECG_raw_patch, Fs, param_dict, HR_ref.max())

    if debug:
        mask_fake = np.ones(QRS_detector_dict_patch[beat_choice].shape[0]).astype(bool)
        QRS_detector_dict_diagnostics(QRS_detector_dict_patch, mask_fake, mask_fake, beat_name=beat_choice,
                                   fig_name='ECG_patch_{}window'.format(j), outputdir=None, show_plot=True)


    N_detected = QRS_detector_dict_patch['i_beat_peaks'].shape[0]

    # step 3. create a template for patch ecg
    if beat_choice=='i_R_peaks':
        ecg_beats = QRS_detector_dict_patch['ecg_beats_R']
    elif beat_choice=='i_beat_peaks':
        ecg_beats = QRS_detector_dict_patch['ecg_beats']
    elif beat_choice=='i_S_peaks':
        ecg_beats = QRS_detector_dict_patch['ecg_beats_S']

    template = ecg_beats.mean(axis=1)

    # step 4. create a mask based on sqi (median deviation < 15 std)
    # Store mask in mask_sqi_patch
    sqi_IP = get_sqi(ecg_beats, get_innerproduct, template)
    
    
    
    med_deviation = np.abs(get_padded_medfilt(sqi_IP, k=5, offset=5) - sqi_IP)

    mask_sqi_patch = med_deviation<= med_deviation.std()*15

    # step 5. get heart rate and their timestamps
    ts_hr_patch = QRS_detector_dict_patch[beat_choice]/Fs
    hr_patch = get_hr(QRS_detector_dict_patch[beat_choice], Fs)
#     print(ts_hr_patch.shape[0])

    # step 6. filter out bad beats using mask_sqi_patch
    ts_hr_patch = ts_hr_patch[mask_sqi_patch]
    hr_patch = hr_patch[mask_sqi_patch]
#     print(ts_hr_patch.shape[0])

    # sys.exit()
    # step 7. create a mask based on HR (median deviation < 30 bpm)
    mask_hr_patch = np.abs(hr_patch-medfilt(hr_patch,k=5)) <= 20

    # step 8. filter out bad beats using mask_hr_patch
    ts_hr_patch = ts_hr_patch[mask_hr_patch]
    hr_patch = hr_patch[mask_hr_patch]
#     print(ts_hr_patch.shape[0])

    # sys.exit()
    # step 9. do another mask. Using label_range_dict['HR']
    mask_final = (hr_patch >= label_range_dict['HR'][0]) & (hr_patch <= label_range_dict['HR'][1])
    ts_hr_patch = ts_hr_patch[mask_final]
    hr_patch = hr_patch[mask_final]
#     print(ts_hr_patch.shape[0])

    # step 10. store ts_hr and hr in QRS_detector_dict_patch
    QRS_detector_dict_patch['ts_hr'] = ts_hr_patch
    QRS_detector_dict_patch['hr'] = hr_patch


    # step 11. show plots

    if fig_name is None:
        fig_name = 'ECG_diagnostics'
    if outputdir is not None or show_plot is not None:
        QRS_detector_dict_diagnostics(QRS_detector_dict_patch, mask_sqi_patch, mask_hr_patch, beat_name=beat_choice,
                               fig_name=fig_name, outputdir=outputdir, show_plot=show_plot)
    
    return QRS_detector_dict_patch