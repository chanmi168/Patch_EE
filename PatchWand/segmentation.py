import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline

from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import numpy as np

from filters import *
from setting import *

def beat_segmentation(sig, i_peaks, start_offset=-50, end_offset=250):
    # segment a signal using the start index (i_peaks+start_offset) and the end index (i_peaks+end_offset) 
    
    window_legnth = end_offset - start_offset

    sig_beats = []
    i_beat_peaks = []
    for i, i_peak in enumerate(i_peaks):
        i_start = int(i_peak + start_offset)
        i_end = int(i_peak + end_offset)

        if i_start < 0 or i_end > sig.shape[0]:
            continue

        sig_seg = sig[i_start:i_end]

        sig_beats.append(sig_seg)
        i_beat_peaks.append(i_peak)

    sig_beats = np.vstack(sig_beats).T
    i_beat_peaks = np.vstack(i_beat_peaks).T.squeeze()

    return sig_beats, i_beat_peaks

def get_filt_df(df_sync, Fs):

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

#     df['ppg_ir_1'] = -get_padded_filt(df['ppg_ir_1'].values, filter_padded=5, lowcutoff=1, highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
#     df['ppg_r_1'] = -get_padded_filt(df['ppg_r_1'].values, filter_padded=5, lowcutoff=1, highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
#     df['ppg_g_1'] = -get_padded_filt(df['ppg_g_1'].values, filter_padded=5, lowcutoff=1, highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
#     df['ppg_ir_2'] = -get_padded_filt(df['ppg_ir_2'].values, filter_padded=5, lowcutoff=1, highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
#     df['ppg_r_2'] = -get_padded_filt(df['ppg_r_2'].values, filter_padded=5, lowcutoff=1, highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)
#     df['ppg_g_2'] = -get_padded_filt(df['ppg_g_2'].values, filter_padded=5, lowcutoff=1, highcutoff=FILT_PPG[1], Fs=FS_RESAMPLE)

    df['accelX'] = get_padded_filt(df['accelX'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
    df['accelY'] = get_padded_filt(df['accelY'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
    df['accelZ'] = get_padded_filt(df['accelZ'].values, filter_padded=5, lowcutoff=FILT_SCG[0], highcutoff=FILT_SCG[1], Fs=Fs)
    
    return df

def segment_df(df, QRS_detector_dict, Fs, start_offset=None, end_offset=None):
    
    ecg_dict = QRS_detector_dict['ecg_dict']
    i_R_peaks = QRS_detector_dict['i_R_peaks']
#     i_R_peaks = QRS_detector_dict['i_R_peaks']

    
    if start_offset is None:
        start_offset = 0
    if end_offset is None:
        end_offset = QRS_detector_dict['end_offset']
    

    # 4. get patch ECG segmentation
    ecg_beats, i_beat_peaks = beat_segmentation(ecg_dict['ecg_filt1'], i_R_peaks, start_offset=0, end_offset=end_offset)

    # # 5. get biopac ECG segmentation
    # _, ecg_dict_biopac = find_peaks_ECG(df_sub_BH['ECG_raw_biopac'].values, width_QRS, filter_padded, Fs)
    # ecg_beats_biopac, i_beat_peaks_biopac = beat_segmentation(ecg_dict_biopac['ecg_filt1'], i_R_peaks, start_offset=0, end_offset=end_offset)

    # 6. get patch PPG segmentation
    ppg_g_1_beats, i_R_peaks_used = beat_segmentation(df['ppg_g_1'].values, i_R_peaks, start_offset=0, end_offset=end_offset)
    ppg_r_1_beats, _ = beat_segmentation(df['ppg_r_1'].values, i_R_peaks, start_offset=0, end_offset=end_offset)
    ppg_ir_1_beats, _ = beat_segmentation(df['ppg_ir_1'].values, i_R_peaks, start_offset=0, end_offset=end_offset)
    ppg_g_2_beats, _ = beat_segmentation(df['ppg_g_2'].values, i_R_peaks, start_offset=0, end_offset=end_offset)
    ppg_r_2_beats, _ = beat_segmentation(df['ppg_r_2'].values, i_R_peaks, start_offset=0, end_offset=end_offset)
    ppg_ir_2_beats, _ = beat_segmentation(df['ppg_ir_2'].values, i_R_peaks, start_offset=0, end_offset=end_offset)

    # 6.2 get patch DC within the beats
#     ppg_DC_dict = get_ppg_DC_dict(df, i_R_peaks, start_offset=0, end_offset=end_offset, Fs=Fs)

    # 6.3 get patch SCG-Z segmentation
    scg_x_beats, _ = beat_segmentation(df['accelX'].values, i_R_peaks, start_offset=0, end_offset=end_offset)

    # 6.3 get patch SCG-Z segmentation
    scg_y_beats, _ = beat_segmentation(df['accelY'].values, i_R_peaks, start_offset=0, end_offset=end_offset)

    # 6.3 get patch SCG-Z segmentation
    scg_z_beats, _ = beat_segmentation(df['accelZ'].values, i_R_peaks, start_offset=0, end_offset=end_offset)


    # TODO: add cosmed data
    beats_dict = {
        'ecg_beats': ecg_beats, # 1
        'ppg_g_1': ppg_g_1_beats, # 10
        'ppg_r_1': ppg_r_1_beats, # 2
        'ppg_ir_1': ppg_ir_1_beats, # 6
        'ppg_g_2': ppg_g_2_beats, # 11
        'ppg_r_2': ppg_r_2_beats, # 3
        'ppg_ir_2': ppg_ir_2_beats, # 7
        'scg_x': scg_x_beats, # 4
        'scg_y': scg_y_beats, # 8
        'scg_z': scg_z_beats, # 12

        'i_R_peaks': i_R_peaks_used, # 0
    #     't_beats': t_sub_BH[i_R_peaks_used],
    #             'SpO2_biopac': SpO2_biopac,
    #     'SpO2_samples_lead': SpO2_samples_lead,
    #             'SpO2_alignment_dict': SpO2_alignment_dict,

    #     'ppg_g_1_DC': ppg_DC_dict['ppg_g_1'],
    #     'ppg_r_1_DC': ppg_DC_dict['ppg_r_1'],
    #     'ppg_ir_1_DC': ppg_DC_dict['ppg_ir_1'],
    #     'ppg_g_2_DC': ppg_DC_dict['ppg_g_2'],
    #     'ppg_r_2_DC': ppg_DC_dict['ppg_r_2'],
    #     'ppg_ir_2_DC': ppg_DC_dict['ppg_ir_2'],
    }

    return beats_dict


# def get_ppg_DC_dict(df, i_R_peaks, start_offset, end_offset, Fs):

#     ppg_DC_dict = {}

#     ppg_names = ['ppg_g_1', 'ppg_r_1', 'ppg_ir_1', 'ppg_g_2', 'ppg_r_2', 'ppg_ir_2']

#     for ppg_name in ppg_names:
#         ppg_DC = ppg_filter(df[ppg_name].values, highcutoff=R_highcutoff, Fs=Fs)
#         ppg_DC, _ = beat_segmentation(ppg_DC, i_R_peaks, start_offset=start_offset, end_offset=end_offset)
#         ppg_DC = ppg_DC.mean(axis=0)
#         ppg_DC_dict[ppg_name] = ppg_DC
        
#     return ppg_DC_dict

def get_ensemble_beats(sig_beats, N_enBeats=4):
    # i_R_peaks = (N_beats,)
    # sig_beats = (N_samples, N_beats)
    # i_ensembles = (<N_beats,)
    # ensemble_beats = (N_samples, <N_beats)
    
    # this function looks at the past 4 beats of a beat (including itself) and average them to produce an ensemble beat
    
    ensemble_beats = []

    for i_beat in range(sig_beats.shape[1]):
        if i_beat == 0:
            ensemble_beats.append(sig_beats[:, i_beat])
            continue
        if i_beat-N_enBeats < 0:
            ensemble_beats.append(sig_beats[:, :i_beat].mean(axis=1))
        else:
            ensAv = sig_beats[:, i_beat-N_enBeats:i_beat].mean(axis=1)    
            ensemble_beats.append(ensAv)
    
    ensemble_beats = np.stack(ensemble_beats,axis=1)
    return ensemble_beats

# def get_PSD_beats(sig_beats, Fs):
#     N = sig_beats.shape[0]
#     T = 1/Fs
#     freq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

#     Zxx = scipy.fftpack.fft(sig_beats, axis=0)[:N//2, :]
#     Pxx = np.abs(Zxx)**2 # shape = (N bins, M beats)

#     return freq, Pxx

def get_PSD_beats(sig_beats, Fs):

    freq, Pxx = signal.welch(sig_beats.T, Fs, nperseg=64)
    Pxx = Pxx.T # rows = # of freq bins, cols = # of beats
    return freq, Pxx