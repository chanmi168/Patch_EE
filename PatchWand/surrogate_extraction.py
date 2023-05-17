# import numpy as np
# import os
# import sys
# import math
# from math import sin


# from scipy.io import loadmat
# import scipy
# from scipy import signal
# from scipy.fftpack import fft, ifft
# from scipy.signal import hilbert, chirp
# from scipy.signal import find_peaks
# from scipy.interpolate import interp1d

# # from CBPregression.feature_extraction import *
# from preprocessing import *
# from setting import *
# from filters import *
# from sklearn.decomposition import PCA

# # width_QRS = label_range_dict['width_QRS']
# width_QRS = 0.15
# downsample_factor = 10

# # BW_padded = 15 # sec


# # def find_peaks_ECG(ecg_raw, width_QRS, filter_padded, Fs):
# #     # step 1. filter ECG (passband = 10-30Hz)
# # #     ecg_filt1 = get_filt(ecg_raw, lowcutoff=1, highcutoff=30, Fs=Fs)
# # #     ecg_filt = get_filt(ecg_raw, lowcutoff=10, highcutoff=30, Fs=Fs)
    
    
# #     ecg_filt1 = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=1, highcutoff=30, Fs=Fs)
# #     ecg_filt = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=10, highcutoff=30, Fs=Fs)

# #     # step 2 (WIP). remove outliers (use SQI)

# #     # step 3. differentiate
# #     # method 1
# #     ecg_diff = np.gradient(ecg_filt)
# #     # method 2
# #     # ecg_diff = derivative_filter(ecg_filt, Fs=FS_ECG)
# #     # ecg_diff = ecg_diff/ecg_diff.max()

# #     # step 4. square
# #     ecg_sqr = ecg_diff**2

# #     # step 5. moving average (we chose 150ms as suggested in Pan and Tompkins' paper)
# #     # Generally, the width of the window should be approximately the same as the
# #     # widest possible QRS complex.
# #     width_QRS = 0.15 # sec
# #     ecg_smooth = get_smooth(ecg_sqr, N=int(Fs*width_QRS))

# #     # ecg_smooth = np.diff(ecg_smooth, append=ecg_smooth[-1])


# #     # step 6. peak detection
# #     hr_max = label_range_dict['heart_rate_cosmed'][1]
# #     # no RR wave can occur in less than 200 msec distance
# #     # distance_min = 0.2 * FS_ECG # (1000 samples/s) / (120 beats/min) * (60s/min)
# #     distance_min = Fs / hr_max * 60 # (1000 samples/s) / (120 beats/min) * (60s/min)
# #     i_peaks, _ = find_peaks(ecg_smooth, distance=distance_min)


# #     # N_toppeaks = math.floor(i_peaks.shape[0]*0.1)
# #     # thre = ecg_filt[peaks].max()*0.6
# #     # thre = ecg_filt[peaks].argsort()[0:N_toppeaks].mean()

# #     indices_CI = arg_CI(ecg_smooth[i_peaks], confidence_interv=50)
# #     thre = ecg_smooth[i_peaks[indices_CI]].mean()*0.1
# # #     thre = ecg_smooth[i_peaks[indices_CI]].mean()*0.2 # try higher threshold to remove some outliers

# #     # peaks_better = peaks[ecg_filt_tilted[peaks]>thre]

# #     i_peaks, _ = find_peaks(ecg_smooth, distance=distance_min, height=thre)

# #     ecg_dict = {
# #         'ecg_filt1': ecg_filt1,
# #         'ecg_filt': ecg_filt,
# #         'ecg_diff': ecg_diff,
# #         'ecg_sqr': ecg_sqr,
# #         'ecg_smooth_pks': ecg_smooth,
# #     }
    
# #     return i_peaks, ecg_dict




# # def find_peaks_resp(resp_sig):
# #     resp_sig_deriv = np.gradient(resp_sig)
# #     resp_sig_sign = np.sign(resp_sig_deriv)
# #     i_resp_peaks = np.where(np.diff(resp_sig_sign)<0)[0]
# #     i_resp_valleys = np.where(np.diff(resp_sig_sign)>0)[0]
    
# #     return i_resp_peaks, i_resp_valleys

# # def calibrate_peaks(i_peaks, sig, Fs):
    
# #     i_calibrate = np.zeros(i_peaks.shape)
    
# #     window_legnth = int(Fs*width_QRS/2)
# # #     window_legnth = int(Fs*width_QRS)
# #     start_offset = -window_legnth
# #     end_offset = window_legnth

# #     for i, i_peak in enumerate(i_peaks):
# #         i_start = i_peak + start_offset
# #         i_end = i_peak + end_offset

# #         if i_start < 0 or i_end > sig.shape[0]:
# #             i_calibrate[i] = i_peak
# #             continue

# #         i_max = sig[i_start:i_end].argmax()
# #         i_calibrate[i] = i_peak + i_max - window_legnth

# #     return i_calibrate.astype(int)


# # def beat_segmentation(sig, i_peaks, start_offset=-50, end_offset=250):
# #     # segment a signal using the start index (i_peaks+start_offset) and the end index (i_peaks+end_offset) 
    
# #     window_legnth = end_offset - start_offset

# #     sig_beats = []
# #     i_beat_peaks = []
# #     for i, i_peak in enumerate(i_peaks):
# #         i_start = int(i_peak + start_offset)
# #         i_end = int(i_peak + end_offset)

# #         if i_start < 0 or i_end > sig.shape[0]:
# #             continue

# #         sig_seg = sig[i_start:i_end]

# #         sig_beats.append(sig_seg)
# #         i_beat_peaks.append(i_peak)

# #     sig_beats = np.vstack(sig_beats).T
# #     i_beat_peaks = np.vstack(i_beat_peaks).T.squeeze()

# #     return sig_beats, i_beat_peaks

    
# def filt_DS_US(ts, Fs, sig, filter_padded, lowcutoff=0.08, highcutoff=1, downsample_factor=5):

#     Fs_DS = Fs / downsample_factor # 100 Hz
#     ts_DS = ts[::downsample_factor]
#     sig_DS = np.interp(ts_DS, ts, sig)    
#     sig_filt = get_padded_filt(sig_DS, filter_padded=filter_padded, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=Fs_DS)
#     sig_filt = np.interp(ts, ts_DS, sig_filt)  

#     return sig_filt

# def ECG_AM_extraction(ts, ecg_filt, i_R_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

#     ts_ECG_AM = ts[i_R_peaks]
#     ECG_AM = ecg_filt[i_R_peaks]

#     ECG_AM_raw = np.interp(ts, ts_ECG_AM, ECG_AM)
#     ECG_AM_smooth = get_smooth(ECG_AM_raw, N=int(Fs*br_smoothing_dur))
    
# #     ECG_AM_filt = get_padded_filt(ECG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#     # downsample, filter, then upsample
#     ECG_AM_filt = filt_DS_US(ts, Fs, ECG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     ECG_AM_normed = (ECG_AM_filt - ECG_AM_filt.mean()) / ECG_AM_filt.std()
    

#     ECG_AM_dict = {
#         'input': ECG_AM, 
#         'raw': ECG_AM_raw,
#         'smooth': ECG_AM_smooth,
#         'filt': ECG_AM_filt,
#         'normed': ECG_AM_normed,
#     }
#     return ECG_AM_dict

# def ECG_AMpt_extraction(ts, ecg_PT, i_R_peaks_PT, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
#     # extract the smooth PT ECG peaks
    
#     ts_ECGpt_AM = ts[i_R_peaks_PT]
#     ECGpt_AM = ecg_PT[i_R_peaks_PT]
    
#     # useful to deal with subject 18
#     ECGpt_AM_median = np.median(ECGpt_AM)
    
#     ECGpt_AM_med = medfilt(ECGpt_AM, 501)
#     indices_good = np.where(np.abs(ECGpt_AM_med-ECGpt_AM)<ECGpt_AM_median*5)[0]
#     ts_ECGpt_AM = ts_ECGpt_AM[indices_good]
#     ECGpt_AM = ECGpt_AM[indices_good]



#     ECGpt_AM_raw = np.interp(ts, ts_ECGpt_AM, ECGpt_AM)
#     ECGpt_AM_smooth = get_smooth(ECGpt_AM_raw, N=int(Fs*br_smoothing_dur))
    
# #     ECGpt_AM_filt = get_padded_filt(ECGpt_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     ECGpt_AM_filt = filt_DS_US(ts, Fs, ECGpt_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

    
#     ECGpt_AM_normed = (ECGpt_AM_filt - ECGpt_AM_filt.mean()) / ECGpt_AM_filt.std()
    

#     ECGpt_AM_dict = {
#         'input': ECGpt_AM, 
#         'raw': ECGpt_AM_raw,
#         'smooth': ECGpt_AM_smooth,
#         'filt': ECGpt_AM_filt,
#         'normed': ECGpt_AM_normed,
#     }
#     return ECGpt_AM_dict


# def ECG_AMbi_extraction(ts, ecg_filt, i_R_peaks, i_S_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):


#     ts_ECG_AM_R = ts[i_R_peaks]
#     ECG_AM_R = ecg_filt[i_R_peaks]
    
#     ts_ECG_AM_S = ts[i_S_peaks]
#     ECG_AM_S = ecg_filt[i_S_peaks]
    
    
#     ECG_AM_raw_R = np.interp(ts, ts_ECG_AM_R, ECG_AM_R)    
#     ECG_AM_raw_S = np.interp(ts, ts_ECG_AM_S, ECG_AM_S)
    
#     ECG_AM_raw = ECG_AM_raw_R - ECG_AM_raw_S

#     ECG_AM_smooth = get_smooth(ECG_AM_raw, N=int(Fs*br_smoothing_dur))

#     # downsample, filter, then upsample
#     ECG_AM_filt = filt_DS_US(ts, Fs, ECG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     ECG_AM_normed = (ECG_AM_filt - ECG_AM_filt.mean()) / ECG_AM_filt.std()
    

#     ECG_AMbi_dict = {
#         'input': ECG_AM_R, 
#         'input2': ECG_AM_S, 
#         'raw': ECG_AM_raw,
#         'smooth': ECG_AM_smooth,
#         'filt': ECG_AM_filt,
#         'normed': ECG_AM_normed,
#     }
#     return ECG_AMbi_dict

# def ECG_FM_extraction(ts, i_R_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

#     ts_ECG_FM = (ts[i_R_peaks[1:]]+ts[i_R_peaks[:-1]])/2
#     ECG_FM = 1/np.diff(ts[i_R_peaks])*60
    
#     # remove outliers: is not physiologically possible
#     indices_good = arg_goodhr(ECG_FM)    
#     ts_ECG_FM = ts_ECG_FM[indices_good]
#     ECG_FM = ECG_FM[indices_good]

#     # beat2beat HR won't change with more than 20bpm (TODO: look up lit)
#     ECG_FM_med = medfilt(ECG_FM, 501)
#     indices_good = np.where(np.abs(ECG_FM_med-ECG_FM)<20)[0]
#     ts_ECG_FM = ts_ECG_FM[indices_good]
#     ECG_FM = ECG_FM[indices_good]
    
# #     ECG_FM = medfilt(ECG_FM, 3)

# #     ECG_FM_grad = np.gradient(ECG_FM)

# #     print(ECG_FM_grad)
# #     print(ECG_FM)
    
# #     ECG_FM = ECG_FM[ECG_FM_grad<10]
# #     ts_ECG_FM = ts_ECG_FM[ECG_FM_grad<10]

# #     print('hihihi')

#     ECG_FM_raw = np.interp(ts, ts_ECG_FM, ECG_FM)
#     ECG_FM_smooth = get_smooth(ECG_FM_raw, N=int(Fs*br_smoothing_dur))
    
# #     ECG_FM_filt = get_padded_filt(ECG_FM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     ECG_FM_filt = filt_DS_US(ts, Fs, ECG_FM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     ECG_FM_normed = (ECG_FM_filt - ECG_FM_filt.mean()) / ECG_FM_filt.std()
    
#     ECG_FM_dict = {
#         'raw': ECG_FM_raw, 
#         'smooth': ECG_FM_smooth,
#         'filt': ECG_FM_filt,
#         'normed': ECG_FM_normed,
#     }
    
#     return ECG_FM_dict


# def ECG_BW_extraction(ts, ecg_raw, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
    
#     ts_ECG_BW = ts
    
# #     ECG_BW_raw = ecg_filt
#     ECG_BW_raw = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)

# #     ECG_BW_raw = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=0.1, highcutoff=highcutoff_br, Fs=Fs) 
#     ECG_BW_smooth = get_smooth(ECG_BW_raw, N=int(Fs*br_smoothing_dur))
    
# #     ECG_BW_filt = get_padded_filt(ECG_BW_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
# #     ECG_BW_filt = ECG_BW_smooth
#    # downsample, filter, then upsample
#     ECG_BW_filt = filt_DS_US(ts, Fs, ECG_BW_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     ECG_BW_normed = (ECG_BW_filt - ECG_BW_filt.mean()) / ECG_BW_filt.std() # this is faster


#     ECG_BW_dict = {
# #         'input': ecg_raw, 
#         'raw': ECG_BW_raw, 
#         'smooth': ECG_BW_smooth,
#         'filt': ECG_BW_filt,
#         'normed': ECG_BW_normed,
#     }

#     return ECG_BW_dict

# def get_good_beats(sig_beats, i_fiducials, confidence_interv=90):
#     # the output indices_SCG refers to the indices of sig_beats of good quality

#     indices_beats = arg_CI(np.var(sig_beats,axis=0), confidence_interv=confidence_interv)
#     sig_beats = sig_beats[:,indices_beats]
#     (i_aos, data_aos) = i_fiducials

#     # TODO: replace this with SQI based outlier removal
#     indices_beats_aos = arg_CI(i_aos[indices_beats], confidence_interv=confidence_interv)
#     indices_SCG = indices_beats[indices_beats_aos]
    
#     return indices_SCG

# def SCG_AM_extraction(ts, sig_beats, i_fiducials, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

#     (i_aos, data_aos) = i_fiducials
#     indices_SCG = get_good_beats(sig_beats, i_fiducials, confidence_interv=90)


#     ts_SCG_AM = ts[i_beat_peaks[indices_SCG]]
#     SCG_AM =  data_aos[indices_SCG]

#     SCG_AM_raw = np.interp(ts, ts_SCG_AM, SCG_AM)
#     SCG_AM_smooth = get_smooth(SCG_AM_raw, N=int(Fs*br_smoothing_dur))
# #     SCG_AM_filt = butter_filter('low', SCG_AM_smooth, highcutoff_br, Fs)
    
# #     SCG_AM_filt = get_padded_filt(SCG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     SCG_AM_filt = filt_DS_US(ts, Fs, SCG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     SCG_AM_normed = (SCG_AM_filt - SCG_AM_filt.mean()) / SCG_AM_filt.std()
    
#     SCG_AM_dict = {
#         'raw': SCG_AM_raw,
#         'smooth': SCG_AM_smooth,
#         'filt': SCG_AM_filt,
#         'normed': SCG_AM_normed,
#     }
#     return SCG_AM_dict

# def SCG_FM_extraction(ts, sig_beats, i_fiducials, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

#     (i_aos, data_aos) = i_fiducials
    
#     indices_SCG = get_good_beats(sig_beats, i_fiducials, confidence_interv=90)

#     i_beat_peaks_sel = i_beat_peaks[indices_SCG]
    
# #     ts_SCG_FM = (ts[i_beat_peaks_sel[1:]]+ts[i_beat_peaks_sel[:-1]])/2
# #     SCG_FM = 1/np.diff(ts[i_beat_peaks_sel])*60
    
    
#     t_arr = ts[i_beat_peaks_sel]+i_aos[indices_SCG]/2/Fs
#     ts_SCG_FM = (t_arr[1:]+t_arr[:-1])/2
#     SCG_FM = 1/np.diff(t_arr)*60


# #     print(ts_SCG_FM.shape, i_beat_peaks_sel.shape, SCG_FM.shape)
# #     sys.exit()
# #         ts_PEP_FM = ts[i_beat_peaks_sel]+i_aos[indices_SCG]/2/Fs

    
    
# #     SCG_FM = 1/np.diff(ts[i_beat_peaks_sel])*60

#     indices_good = arg_goodhr(SCG_FM)
#     ts_SCG_FM = ts_SCG_FM[indices_good]
#     SCG_FM = SCG_FM[indices_good]

#     # use median filter to remove glitch
#     SCG_FM = medfilt(SCG_FM, 5)
#     ts_SCG_FM = ts_SCG_FM[1:-1]
#     SCG_FM = SCG_FM[1:-1]
# #     print(SCG_FM)
# #     print(t_arr, SCG_FM)
# #     sys.exit()
    
#     SCG_FM_raw = np.interp(ts, ts_SCG_FM, SCG_FM)
#     SCG_FM_smooth = get_smooth(SCG_FM_raw, N=int(Fs*br_smoothing_dur))
# #     SCG_FM_filt = butter_filter('low', SCG_FM_smooth, highcutoff_br, Fs)

# #     SCG_FM_filt = get_padded_filt(SCG_FM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     SCG_FM_filt = filt_DS_US(ts, Fs, SCG_FM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     SCG_FM_normed = (SCG_FM_filt - SCG_FM_filt.mean()) / SCG_FM_filt.std()


#     SCG_FM_dict = {
# #         'aaa': SCG_FM,
#         'raw': SCG_FM_raw, 
#         'smooth': SCG_FM_smooth,
#         'filt': SCG_FM_filt,
#         'normed': SCG_FM_normed,
#     }

#     return SCG_FM_dict

# def SCG_BW_extraction(ts, scg_raw, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
    
#     ts_SCG_BW = ts
    
#     SCG_BW_raw = get_padded_filt(scg_raw, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#     SCG_BW_smooth = get_smooth(SCG_BW_raw, N=int(Fs*br_smoothing_dur))    
# #     SCG_BW_filt = get_padded_filt(SCG_BW_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
# #     SCG_BW_filt = SCG_BW_smooth
#    # downsample, filter, then upsample
#     SCG_BW_filt = filt_DS_US(ts, Fs, SCG_BW_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     SCG_BW_normed = (SCG_BW_filt - SCG_BW_filt.mean()) / SCG_BW_filt.std() # this is faster

#     SCG_BW_dict = {
#         'raw': SCG_BW_raw, 
#         'smooth': SCG_BW_smooth,
#         'filt': SCG_BW_filt,
#         'normed': SCG_BW_normed,
#     }

#     return SCG_BW_dict


# def PEP_FM_extraction(ts, sig_beats, i_fiducials, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

#     (i_aos, data_aos) = i_fiducials

# #     indices_SCG = get_good_beats(sig_beats, i_fiducials, confidence_interv=90)

# #     i_beat_peaks_sel = i_beat_peaks[indices_SCG]
#     indices_SCG = i_aos
#     i_beat_peaks_sel = i_beat_peaks
# #     print(indices_SCG.shape, i_beat_peaks_sel.shape)
    
#     ts_PEP_FM = ts[i_beat_peaks_sel]+indices_SCG/2/Fs
#     PEP_FM = i_aos/Fs
    

    
#     # TODO: implement a arg_goodPEP(PEP) function
    
# #     indices_good = arg_goodhr(PEP_FM)
# #     ts_PEP_FM = ts_PEP_FM[indices_good]
# #     PEP_FM = PEP_FM[indices_good]

#     PEP_FM_raw = np.interp(ts, ts_PEP_FM, PEP_FM)
#     PEP_FM_smooth = get_smooth(PEP_FM_raw, N=int(Fs*br_smoothing_dur))
    
# #     PEP_FM_filt = butter_filter('low', PEP_FM_smooth, highcutoff_br, Fs)
# #     PEP_FM_filt = get_padded_filt(PEP_FM_smooth, filter_padded=filter_padded, lowcutoff=0.1, highcutoff=highcutoff_br, Fs=Fs)


# #     PEP_FM_filt = get_padded_filt(PEP_FM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     PEP_FM_filt = filt_DS_US(ts, Fs, PEP_FM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     PEP_FM_normed = (PEP_FM_filt - PEP_FM_filt.mean()) / PEP_FM_filt.std()


#     PEP_FM_dict = {
#         'raw': PEP_FM_raw, 
#         'smooth': PEP_FM_smooth,
#         'filt': PEP_FM_filt,
#         'normed': PEP_FM_normed,
#     }
    
#     return PEP_FM_dict

# # sig = ecg_beats2[:,0]
# def ECG_SR_extraction(ts, sig_beats, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1, offset_SR=100):
    
#     i=offset_SR-1
#     ts_ECG_SR = ts[i_beat_peaks]
#     sig_diff = np.gradient(sig_beats,axis=0)

#     indicies_valleys = np.argsort(sig_diff,axis=0) # small to large, pick first and >i
#     indicies_peaks = np.argsort(sig_diff*-1,axis=0) # large to small, pick first and < i

#     i_slopemin = np.zeros(indicies_valleys.shape[1])
#     for i_col in range(indicies_valleys.shape[1]):
#         indices = indicies_valleys[:,i_col]
#         i_slopemin[i_col] = indices[indices>=i][0]

        
#     i_slopemax = np.zeros(indicies_peaks.shape[1])
#     for i_col in range(indicies_peaks.shape[1]):
#         indices = indicies_peaks[:,i_col]
#         i_slopemax[i_col] = indices[indices<=i][0]

#     i_slopemin = i_slopemin.astype(int)
#     i_slopemax = i_slopemax.astype(int)

#     slopemin = sig_diff[i_slopemin, np.arange(sig_diff.shape[1])]
#     slopemax = sig_diff[i_slopemax, np.arange(sig_diff.shape[1])]

#     sloperange = slopemax-slopemin


#     ECG_SR_raw = np.interp(ts, ts_ECG_SR, sloperange)
#     ECG_SR_smooth = get_smooth(ECG_SR_raw, N=int(Fs*br_smoothing_dur))
    
# #     ECG_SR_filt = butter_filter('low', ECG_SR_smooth, highcutoff_br, Fs)
# #     ECG_SR_filt = get_padded_filt(ECG_SR_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     ECG_SR_filt = filt_DS_US(ts, Fs, ECG_SR_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     ECG_SR_normed = (ECG_SR_filt - ECG_SR_filt.mean()) / ECG_SR_filt.std()

#     ECG_SR_dict = {
#         'raw': ECG_SR_raw, 
#         'smooth': ECG_SR_smooth,
#         'filt': ECG_SR_filt,
#         'normed': ECG_SR_normed,
#     }
#     return ECG_SR_dict

# def ECG_PCA_extraction(ts, sig_beats, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1, offset_SR=100):
    
#     i=offset_SR-1
#     ts_ECG_PCA = ts[i_beat_peaks]
    
#     pca = PCA(n_components=1)
#     sig_beats_norm = sig_beats-sig_beats.mean(axis=0)
#     sig_beats_norm = sig_beats_norm.T

#     # Fit the PCA and transform the data
#     sig_pca = pca.fit_transform(sig_beats_norm)[:,0]

#     ECG_PCA_raw = np.interp(ts, ts_ECG_PCA, sig_pca)
#     ECG_PCA_smooth = get_smooth(ECG_PCA_raw, N=int(Fs*br_smoothing_dur))
    
#     # downsample, filter, then upsample
#     ECG_PCA_filt = filt_DS_US(ts, Fs, ECG_PCA_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

#     ECG_PCA_normed = (ECG_PCA_filt - ECG_PCA_filt.mean()) / ECG_PCA_filt.std()

#     ECG_PCA_dict = {
#         'raw': ECG_PCA_raw, 
#         'smooth': ECG_PCA_smooth,
#         'filt': ECG_PCA_filt,
#         'normed': ECG_PCA_normed,
#     }
#     return ECG_PCA_dict



# def SCGMAG_AM_extraction(ts, sig_beats, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
#     # square
#     scg_beats_sqr = sig_beats**2

#     scg_beats_smooth = np.zeros(scg_beats_sqr.shape)

#     for i_sample in range(scg_beats_smooth.shape[1]):
#         scg_beats_smooth[:,i_sample] = get_smooth(scg_beats_sqr[:,i_sample], N=int(Fs*width_QRS))

#     # ecg_smooth = np.diff(ecg_smooth, append=ecg_smooth[-1])

#     i_peaks_mag = scg_beats_smooth[:int(width_QRS*Fs),:].argmax(axis=0)

#     ts_SCGMAG_AM = ts[i_beat_peaks]
#     SCGMAG_AM = scg_beats_smooth[i_peaks_mag, np.arange(i_peaks_mag.shape[0])]

#     SCGMAG_AM_raw = np.interp(ts, ts_SCGMAG_AM, SCGMAG_AM)
#     SCGMAG_AM_smooth = get_smooth(SCGMAG_AM_raw, N=int(Fs*br_smoothing_dur))
    
# #     SCGMAG_AM_filt = butter_filter('low', SCGMAG_AM_smooth, highcutoff_br, Fs)
# #     SCGMAG_AM_filt = get_padded_filt(SCGMAG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     SCGMAG_AM_filt = filt_DS_US(ts, Fs, SCGMAG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

    
#     SCGMAG_AM_normed = (SCGMAG_AM_filt - SCGMAG_AM_filt.mean()) / SCGMAG_AM_filt.std()

#     SCGMAG_AM_dict = {
#         'raw': SCGMAG_AM_raw, 
#         'smooth': SCGMAG_AM_smooth,
#         'filt': SCGMAG_AM_filt,
#         'normed': SCGMAG_AM_normed,
#     }
#     return SCGMAG_AM_dict

# # Edit: added on 3/22
# def SCG_AMpt_extraction(ts, scg_filt, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
#     # inspired by PT
    

#     i_peaks, scg_dict = find_SCGpt_peaks(scg_filt, Fs)

#     scg_smooth = scg_dict['scg_smooth']

# #     i_peaks = find_SCGpt_peaks(scg_filt, Fs)

        
#     SCG_AMpt = scg_smooth[i_peaks]
#     ts_SCG_AMpt = ts[i_peaks]
    
#     # useful to deal with pt methods
#     SCG_AMpt_median = np.median(SCG_AMpt)
    
#     SCG_AMpt_med = medfilt(SCG_AMpt, 501)
#     indices_good = np.where(np.abs(SCG_AMpt_med-SCG_AMpt)<SCG_AMpt_median*5)[0]

#     bad_ratio = 1-indices_good.shape[0]/SCG_AMpt.shape[0]
#     print('\t\t [SCG_AMpt] reject {:.2f}% beats (diff with SCG_AMpt_median < SCG_AMpt_median*5)'.format(bad_ratio*100))

#     ts_SCG_AMpt = ts_SCG_AMpt[indices_good]
#     SCG_AMpt = SCG_AMpt[indices_good]

    
# #     # beat2beat HR won't change with more than 20bpm (TODO: look up lit)
# #     ECG_FM_med = medfilt(SCG_AMpt, 501)
# #     indices_good = np.where(np.abs(ECG_FM_med-ECG_FM)<20)[0]
# #     ts_ECG_FM = ts_ECG_FM[indices_good]
# #     ECG_FM = ECG_FM[indices_good]


#     SCG_AMpt_raw = np.interp(ts, ts_SCG_AMpt, SCG_AMpt)
#     SCG_AMpt_smooth = get_smooth(SCG_AMpt_raw, N=int(Fs*br_smoothing_dur))

#     #     SCG_AMpt_filt = get_padded_filt(SCG_AMpt_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#    # downsample, filter, then upsample
#     SCG_AMpt_filt = filt_DS_US(ts, Fs, SCG_AMpt_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=5)

    
#     SCG_AMpt_normed = (SCG_AMpt_filt - SCG_AMpt_filt.mean()) / SCG_AMpt_filt.std()

#     SCG_AMpt_dict = {
#         'raw': SCG_AMpt_raw, 
#         'smooth': SCG_AMpt_smooth,
#         'filt': SCG_AMpt_filt,
#         'normed': SCG_AMpt_normed,
#         'scg_dict': scg_dict,
#         'i_peaks': i_peaks,
#     }
    
#     return SCG_AMpt_dict


# def find_SCGpt_peaks(scg_filt, Fs):
# #         scg_filt = get_padded_filt(scg_raw, filter_padded=filter_padded, lowcutoff=1, highcutoff=40, Fs=Fs)

#     scg_diff = np.gradient(scg_filt)

#     scg_sqr = scg_diff**2


#     scg_smooth = get_smooth(scg_sqr, N=int(Fs*width_QRS))

# #     hr_max = label_range_dict['heart_rate_cosmed'][1]
#     hr_max = 200
#     # no RR wave can occur in less than 200 msec distance
#     # distance_min = 0.2 * FS_ECG # (1000 samples/s) / (120 beats/min) * (60s/min)
#     distance_min = Fs / hr_max * 60 # (1000 samples/s) / (120 beats/min) * (60s/min)
#     i_peaks, _ = find_peaks(scg_smooth, distance=distance_min)

#     indices_CI = arg_CI(scg_smooth[i_peaks], confidence_interv=50)
#     thre = scg_smooth[i_peaks[indices_CI]].mean()*0.2


#     i_peaks, _ = find_peaks(scg_smooth, distance=distance_min, height=thre)

#     scg_dict = {
#         'scg_diff': scg_diff,
#         'scg_sqr': scg_sqr,
#         'scg_smooth': scg_smooth,
#     }
    
#     return i_peaks, scg_dict



