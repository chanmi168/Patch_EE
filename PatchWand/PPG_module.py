import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline

import numpy as np

from segmentation import *
from setting import *
from filters import *
from preprocessing import *
from resp_module import *
from spectral_module import *
# from segmentation import beat_segmentation

deviation_thre = 3



# PAT_min = 0.075
PAT_min = 0.02 # 8/20
PAT_max = 0.25
PAT_max = 0.6 # 9/3


# TODO: add documentation

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



def get_FQI_mask(features, method='mean'):
    '''
    Compute a FQI (feature) mask to classify the beats into good/bad beats. 
    One can use mean absolute deviation or median absolute error (MAD)
    MAD is usually advantage of being very insensitive to the presence of outliers.
    
    Parameters:
            features (M): a vector that store M features (1 feature for each beat)

    Returns:
            mask (M,): contain an array of booleans. Determined using a hard-coded decision rule.
            
    '''
    
    if method=='median':
        median = np.median(features)
        b = 1.4826
        MAD = b * np.median(np.abs(features - median))
        mask = np.abs(features-median) < MAD*deviation_thre
    elif method=='mean':
        mask = np.abs(features-features.mean()) < features.std()*deviation_thre
    else:
        print('WRONG METHOD input')
        mask = None
    return mask



def get_AC(ts, ppg_filt):
    threshold = 20
    i_PPG_peaks, ts_PPG_HR, PPG_HR = get_PPG_peaks(ts, ppg_filt, thre=threshold)
    i_PPG_peaks_neg, ts_PPG_HR_neg, PPG_HR_neg = get_PPG_peaks(ts, -ppg_filt, thre=threshold)

    # PPG positive and negative envelopes
    ts_envelope = ts[i_PPG_peaks]
    ppg_envelope_med = medfilt(ppg_filt[i_PPG_peaks], 5)

    ts_envelope_neg = ts[i_PPG_peaks_neg]
    ppg_envelope_med_neg = medfilt(ppg_filt[i_PPG_peaks_neg], 5)

    # interp the envelope
    ppg_envelope_interp = np.interp(ts, ts_envelope, ppg_envelope_med)
    ppg_envelope_interp_neg = np.interp(ts, ts_envelope_neg, ppg_envelope_med_neg)

    # get AC
    PPG_biopac_AC = ppg_envelope_interp-ppg_envelope_interp_neg
    
    return PPG_biopac_AC

    
def ppg_argmin(ppg_beats, Fs):
    # ppg_beats = (N_samples, N_beats)
    # indices_min = (N_beats,)
    # PAT_mask = (N_beats,)

    # foot will be within the first 400ms only
    window_start = int(PAT_min*Fs)
    window_end = int(PAT_max*Fs)
#     window_end = int(0.4*Fs)
    indices_min = np.argmin(ppg_beats[window_start:window_end,:], axis=0)
    indices_min = indices_min + window_start

    # PPG foot will be between PAT_min and PAT_max (reject those exceed this window)
    # TODO: add reference


#     PAT_mask = (indices_min/Fs > PAT_min*1.05) & (indices_min/Fs < PAT_max*0.95) 
#     t_mask = np.abs(indices_min-indices_min.mean()) < indices_min.std()*std_thre
    t_mask = get_FQI_mask(indices_min)

    
    mins = np.min(ppg_beats[window_start:window_end,:], axis=0)
    
#     value_mask = np.abs(mins-mins.mean()) < mins.std()*std_thre
    value_mask = get_FQI_mask(mins)

    
#     return indices_min, PAT_mask
    return indices_min, mins, t_mask, value_mask
    # PAT_mask = np.where(PAT_mask==1)[0]
    # indices_min_selected = indices_min[PAT_mask]
    # i_ensembles_selected = i_ensembles[PAT_mask]



# indices_min.shape, i_R_peaks_good.shape, indices_min_selected.shape, i_ensembles_selected.shape


PEAK_min = 0.2
PEAK_min = 0.15 # 8/20
PEAK_max = 0.9

def ppg_argmax(ppg_beats, Fs):
    # ppg_beats = (N_samples, N_beats)
    # indices_max = (N_beats,)
    # PEAK_mask = (N_beats,)
    
    # window_start = int(0.4*Fs)
    window_start = int(0.2*Fs)
    window_end = int(PEAK_max*Fs)

#     indices_max = np.argmax(ppg_beats[window_start:,:], axis=0)
    indices_max = np.argmax(ppg_beats[window_start:window_end,:], axis=0)
    indices_max = indices_max + window_start


#     PEAK_mask = (indices_max/Fs > PEAK_min*1.05) & (indices_max/Fs < PEAK_max*0.95) 
#     t_mask = np.abs(indices_max-indices_max.mean()) < indices_max.std()*std_thre
    t_mask = get_FQI_mask(indices_max)

    maxs = np.max(ppg_beats[window_start:window_end,:], axis=0)
#     value_mask = np.abs(maxs-maxs.mean()) < maxs.std()*std_thre
    value_mask = get_FQI_mask(maxs)

    return indices_max, maxs, t_mask, value_mask



def get_corr(y1, y2):
    '''
    This function computes the normalized cross-correlation of signals y1 and y2. 
    y1 dim: (N,)
    y2 dim: (N,)
    corr dim: scalar
    '''
    
    # editted on 10/11: remove mean for input signals
    y1 = y1 - y1.mean()
    y2 = y2 - y2.mean()
    n = len(y1)
    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])
    return corr


def get_max_xcorr(beats, beats_template):
    '''
    This function (distance_method) computes the distance [cross-correlation] between the each beat in beats and beats_template
    beats dim: (N,M)
    beats_template dim: (N,)
    xcorr dim: (M,)
    '''
    corr_matrix = np.zeros(beats.shape)
    for i_col in range(beats.shape[1]):
        corr_matrix[:, i_col] = get_corr(beats_template, beats[:,i_col])

    beats_xcorr = corr_matrix.max(axis=0)
    return beats_xcorr

def get_dtw(beats, beats_template):
    '''
    This function (distance_method) computes the distance [dynamic time warping] between the each beat in beats and beats_template
    beats dim: (N,M)
    beats_template dim: (N,)
    beats_dtw dim: (M,)
    '''
    beats_dtw = np.zeros(beats.shape[1])
    for i in range(beats.shape[1]):
        si = beats[:,i]
        distance = dtw.distance(si, beats_template)

        beats_dtw[i] = np.exp(-distance/np.abs(beats_template).sum())
    
    return beats_dtw


def get_sqi(beats, distance_method, template=None):
    '''
    Returns the sqi score for every beat.

    Parameters:
            beats (NxM): a beat matrix that store M beats of the signal (each beat signal has a dimension of N)
            distance_method (python function): takes a beat matrix and a template and subsequently compute distance measures between each beat and the template 
                                                output is a N dimensional array
            template (N,):  a beat signal used to compute the distance measure. If None, the ensemble average of the best 10% beats will be used as the template. 
                            Be careful if beats contain mostly noisy signals. An external template would be recommended.

    Returns:
            sqi_distances (M,): 
            
    Usage:
            sqi_xcorr = get_sqi(beats, get_max_xcorr, template)

    '''
    
    best_partition = 10 # use the best 10 % to produce the template
    
    if template is None:
        beats_template0 = beats.mean(axis=1)

        sqi_xcorr0 = distance_method(beats, beats_template0)

        indices_template = np.argsort(-sqi_xcorr0)[:sqi_xcorr0.shape[0]//best_partition] # use the top 10% to comptue a new template
        beats_template = beats[:,indices_template].mean(axis=1)
        sqi_distances = distance_method(beats, beats_template)

    else:
        beats_template = template
        sqi_distances = distance_method(beats, beats_template)

    return sqi_distances



def clean_PPG(beats_task, template, Fs):
    # 2.1 create a template
    # if '2' in beat_name:
    # template2 = beats_dict['ppg_g_2'].mean(axis=1)
    # elif '1' in beat_name:
    # template1 = beats_dict['ppg_g_1'].mean(axis=1)

    # 2.2 create a mask based on sqi (statistical test + threshold test)

    # beats_task = beats_dict['ppg_g_1']
    sqi_xcorr = get_sqi(beats_task, get_max_xcorr, template)

    #                 print(sqi_xcorr, sqi_xcorr.mean(), sqi_xcorr.std(), sqi_xcorr.shape)

    #                 mask_sqi = (np.abs(sqi_xcorr-sqi_xcorr.mean()) <= sqi_xcorr.std()*3) & (sqi_xcorr >= 0.8)




    mask_sqi = (np.abs(sqi_xcorr-sqi_xcorr.mean()) <= sqi_xcorr.std()*5) & (sqi_xcorr >= 0.7)
    #                 print(mask_sqi, beats_BH.shape)
    #                 sys.exit()

    sqi_rate = mask_sqi.sum()/mask_sqi.shape[0]

    # 3. fiducial extraction

    # 3.1 extract PPG fiducial points
    indices_beats_min, mins, tVALLEY_mask, vVALLEY_mask  = ppg_argmin(beats_task, Fs)

    #                 print(tVALLEY_mask.mean(), vVALLEY_mask.mean())

    indices_beats_max, maxs, tPEAK_mask, vPEAK_mask = ppg_argmax(beats_task, Fs)

    #                 print(tPEAK_mask.mean(), vPEAK_mask.mean())

    #                 sys.exit()
    #                 if subject_id == '120':
    #                     indices_beats_max, maxs, tPEAK_mask, vPEAK_mask = ppg_argmax(beats_BH)
    #                 else:
    #                     indices_beats_max, maxs, tPEAK_mask, vPEAK_mask = ppg_argmax2(beats_BH)

    # 3.2 extract PPG fiducial mask
    mask_std = tVALLEY_mask & tPEAK_mask
    mask_value = vVALLEY_mask & vPEAK_mask
    mask_fiducial = mask_std & mask_value

    # 4.1 make mask_all
    mask_all = mask_sqi & mask_fiducial

    # 4.2 diagnosis metrics
    std_rate = mask_std.sum()/mask_std.shape[0]
    fiducial_rate = mask_value.sum()/mask_value.shape[0]

    ol_rate = 1 - mask_all.sum() / mask_all.shape[0]
    
    return mask_all, ol_rate

def get_ppg_fiducials(df_task, ppg_name, QRS_detector_dict_patch, Fs):

    # 1. get data 
    # ppg_name = 'ppg_g_1_cardiac'
    ppg = df_task[ppg_name].values
    i_R_peaks = QRS_detector_dict_patch['i_R_peaks']
    ppg_beats, i_ppg_peaks = beat_segmentation(ppg, i_R_peaks, start_offset=0, end_offset=int(0.8*Fs))
    ppg_beats = ppg_beats - ppg_beats.mean(axis=0)

    # 2. SQI block

    # 2.1 create a rough estimate of the template
    wavelength =  ppg_name.split('_')[1]
#     template_name = ppg_name.replace('_'+wavelength+'_', '_r_')
    template_name = ppg_name

    template_beats, _ = beat_segmentation(df_task[template_name].values, i_R_peaks, start_offset=0, end_offset=int(0.8*Fs))
    template = template_beats.mean(axis=1)
    sqi_xcorr = get_sqi(template_beats, get_max_xcorr, template)

    # 2.2 get a more realistic template
    value_up = np.percentile(sqi_xcorr, 90)
    mask_sqi = sqi_xcorr >= value_up
    template = template_beats[:, mask_sqi].mean(axis=1)

    # 2.3 create a mask based on sqi (statistical test + threshold test)
    sqi_xcorr = get_sqi(ppg_beats, get_max_xcorr, template)

    sqi_deviation = np.abs(sqi_xcorr-np.median(sqi_xcorr))

    if template_name!=ppg_name:
        mask_sqi = (sqi_deviation <= sqi_deviation.std()*5) & (sqi_xcorr >= 0.8)
    else:
        mask_sqi = sqi_xcorr >= 0.6

    # mask_sqi = (np.abs(sqi_xcorr-sqi_xcorr.median()) <= sqi_xcorr.std()*5) & (sqi_xcorr >= 0.7)
    # print(mask_sqi.mean())

    ppg_beats = ppg_beats[:, mask_sqi]
    i_ppg_peaks = i_ppg_peaks[mask_sqi]
    #                 print(mask_sqi, beats_BH.shape)
    #                 sys.exit()

    # sqi_rate = mask_sqi.sum()/mask_sqi.shape[0]

    # 3. fiducial extraction

    # 3.1 extract PPG fiducial points
    indices_beats_min, mins, tVALLEY_mask, vVALLEY_mask  = ppg_argmin(ppg_beats, Fs)
    indices_beats_max, maxs, tPEAK_mask, vPEAK_mask = ppg_argmax(ppg_beats, Fs)
    
    
    # 4. get i_max, ppg_max, i_min, ppg_min
    indices_beats_max += i_ppg_peaks
    ppg_max = ppg[indices_beats_max]

    # i_min = np.argmin(ppg_beats, axis=0)
    indices_beats_min += i_ppg_peaks
    ppg_min = ppg[indices_beats_min]
    
    return indices_beats_max, ppg_max, indices_beats_min, ppg_min, i_ppg_peaks, ppg


def get_PPG_HR(ppg, Fs):
    xf, yf = get_psd(ppg, Fs) #xf is in Hz
    hr_ppg = xf[np.argmax(yf)]*60
    
    return hr_ppg