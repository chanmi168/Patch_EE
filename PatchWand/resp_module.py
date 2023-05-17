import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline
import seaborn as sns

from sklearn.decomposition import PCA

from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


import numpy as np

from segmentation import *
from setting import *
from filters import *
from preprocessing import *
from PPG_module import *
from plotting_tools import *
from evaluate import *

# width_QRS = label_range_dict['width_QRS']
width_QRS = 0.15
downsample_factor = 10


# def get_RR_range():
#     return np.asarray([label_range_dict['RR'][0]/60, label_range_dict['RR'][1]/60])
# #     return np.asarray([0.08, 0.7])


def get_PSD_scipy(sig, Fs):
    
    N = sig.shape[0]
    T = 1/Fs

    Zxx = scipy.fftpack.fft(sig)[:N//2]
    Pxx = np.abs(Zxx)**2

    freq = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

    
#     RR_range = get_RR_range()
    RR_range = np.asarray([label_range_dict['RR'][0]/60, label_range_dict['RR'][1]/60])

    
    mask = (freq>RR_range[0]) & (freq<RR_range[-1])
    freq = freq[mask]
 
    Pxx = Pxx[mask]
    
    return freq, Pxx

def inspect_resp(sig, RR_label, Fs, fig_name=None):

    freq, Pxx = get_PSD_scipy(sig, Fs)

    i_max = np.argmax(Pxx.squeeze())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), dpi=80)

    t_sig = np.arange(sig.shape[0])/Fs
    ax1.plot(t_sig, sig)
    if fig_name is not None:
        ax1.set_title(fig_name)


    RR_est = freq[i_max]*60
    ax2.plot(freq*60, Pxx.squeeze())
    ax2.scatter(freq[i_max]*60, Pxx.squeeze()[i_max])
    ax2.set_title('RR_est={:.2f}bpm\nRR_label={:.2f}bpm'.format(RR_est, RR_label))

    fig.tight_layout()
    
# TODO: move this to filters.py
def filt_DS_US(ts, Fs, sig, filter_padded, lowcutoff=0.08, highcutoff=1, downsample_factor=5):

    Fs_DS = Fs / downsample_factor # 100 Hz
    ts_DS = ts[::downsample_factor]
    sig_DS = np.interp(ts_DS, ts, sig)    
    sig_filt = get_padded_filt(sig_DS, filter_padded=filter_padded, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=Fs_DS)
    sig_filt = np.interp(ts, ts_DS, sig_filt)  

    return sig_filt

def ECG_AM_extraction(ts, ecg_filt, i_R_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

    ts_ECG_AM = ts[i_R_peaks]
    ECG_AM = ecg_filt[i_R_peaks]

    ECG_AM_raw = np.interp(ts, ts_ECG_AM, ECG_AM)
    ECG_AM_smooth = get_smooth(ECG_AM_raw, N=int(Fs*br_smoothing_dur))
    
#     ECG_AM_filt = get_padded_filt(ECG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
    # downsample, filter, then upsample

    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    ECG_AM_filt = filt_DS_US(ts, Fs, ECG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    ECG_AM_normed = (ECG_AM_filt - ECG_AM_filt.mean()) / ECG_AM_filt.std()
    

    ECG_AM_dict = {
        'input': ECG_AM, 
        'raw': ECG_AM_raw,
        'smooth': ECG_AM_smooth,
        'filt': ECG_AM_filt,
        'normed': ECG_AM_normed,
    }
    return ECG_AM_dict

def ECG_AMpt_extraction(ts, ecg_PT, i_R_peaks_PT, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
    # extract the smooth PT ECG peaks
    
    ts_ECGpt_AM = ts[i_R_peaks_PT]
    ECGpt_AM = ecg_PT[i_R_peaks_PT]
    
    # useful to deal with subject 18 in GT dataset
    ECGpt_AM_median = np.median(ECGpt_AM)
    
    ECGpt_AM_med = medfilt(ECGpt_AM, math.ceil(Fs / 2.) * 2 + 1)
    indices_good = np.where(np.abs(ECGpt_AM_med-ECGpt_AM)<ECGpt_AM_median*5)[0]
    ts_ECGpt_AM = ts_ECGpt_AM[indices_good]
    ECGpt_AM = ECGpt_AM[indices_good]



    ECGpt_AM_raw = np.interp(ts, ts_ECGpt_AM, ECGpt_AM)
    ECGpt_AM_smooth = get_smooth(ECGpt_AM_raw, N=int(Fs*br_smoothing_dur))
    
#     ECGpt_AM_filt = get_padded_filt(ECGpt_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    ECGpt_AM_filt = filt_DS_US(ts, Fs, ECGpt_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    
    ECGpt_AM_normed = (ECGpt_AM_filt - ECGpt_AM_filt.mean()) / ECGpt_AM_filt.std()
    

    ECGpt_AM_dict = {
        'input': ECGpt_AM, 
        'raw': ECGpt_AM_raw,
        'smooth': ECGpt_AM_smooth,
        'filt': ECGpt_AM_filt,
        'normed': ECGpt_AM_normed,
    }
    return ECGpt_AM_dict


def ECG_AMbi_extraction(ts, ecg_filt, i_R_peaks, i_S_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):


    ts_ECG_AM_R = ts[i_R_peaks]
    ECG_AM_R = ecg_filt[i_R_peaks]
    
    ts_ECG_AM_S = ts[i_S_peaks]
    ECG_AM_S = ecg_filt[i_S_peaks]
    
    
    ECG_AM_raw_R = np.interp(ts, ts_ECG_AM_R, ECG_AM_R)    
    ECG_AM_raw_S = np.interp(ts, ts_ECG_AM_S, ECG_AM_S)
    
    ECG_AM_raw = ECG_AM_raw_R - ECG_AM_raw_S

    ECG_AM_smooth = get_smooth(ECG_AM_raw, N=int(Fs*br_smoothing_dur))

    # downsample, filter, then upsample

    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    ECG_AM_filt = filt_DS_US(ts, Fs, ECG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    ECG_AM_normed = (ECG_AM_filt - ECG_AM_filt.mean()) / ECG_AM_filt.std()
    

    ECG_AMbi_dict = {
        'input': ECG_AM_R, 
        'input2': ECG_AM_S, 
        'raw': ECG_AM_raw,
        'smooth': ECG_AM_smooth,
        'filt': ECG_AM_filt,
        'normed': ECG_AM_normed,
    }
    return ECG_AMbi_dict

def ECG_FM_extraction(ts, i_R_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

    ts_ECG_FM = (ts[i_R_peaks[1:]]+ts[i_R_peaks[:-1]])/2
    ECG_FM = 1/np.diff(ts[i_R_peaks])*60
    
    # remove outliers: is not physiologically possible
    indices_good = arg_goodhr(ECG_FM)    
    ts_ECG_FM = ts_ECG_FM[indices_good]
    ECG_FM = ECG_FM[indices_good]

    # beat2beat HR won't change with more than 20bpm (TODO: look up lit)
    ECG_FM_med = medfilt(ECG_FM, math.ceil(Fs / 2.) * 2 + 1)
    indices_good = np.where(np.abs(ECG_FM_med-ECG_FM)<20)[0]
    ts_ECG_FM = ts_ECG_FM[indices_good]
    ECG_FM = ECG_FM[indices_good]

    ECG_FM_raw = np.interp(ts, ts_ECG_FM, ECG_FM)
    ECG_FM_smooth = get_smooth(ECG_FM_raw, N=int(Fs*br_smoothing_dur))
    
#     ECG_FM_filt = get_padded_filt(ECG_FM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    ECG_FM_filt = filt_DS_US(ts, Fs, ECG_FM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    ECG_FM_normed = (ECG_FM_filt - ECG_FM_filt.mean()) / ECG_FM_filt.std()
    
    ECG_FM_dict = {
        'raw': ECG_FM_raw, 
        'smooth': ECG_FM_smooth,
        'filt': ECG_FM_filt,
        'normed': ECG_FM_normed,
    }
    
    return ECG_FM_dict


def ECG_BW_extraction(ts, ecg_raw, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
    
    ts_ECG_BW = ts
    
#     ECG_BW_raw = ecg_filt
    ECG_BW_raw = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)

#     ECG_BW_raw = get_padded_filt(ecg_raw, filter_padded=filter_padded, lowcutoff=0.1, highcutoff=highcutoff_br, Fs=Fs) 
    ECG_BW_smooth = get_smooth(ECG_BW_raw, N=int(Fs*br_smoothing_dur))
    
#     ECG_BW_filt = get_padded_filt(ECG_BW_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#     ECG_BW_filt = ECG_BW_smooth
   # downsample, filter, then upsample
    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    ECG_BW_filt = filt_DS_US(ts, Fs, ECG_BW_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    ECG_BW_normed = (ECG_BW_filt - ECG_BW_filt.mean()) / ECG_BW_filt.std() # this is faster


    ECG_BW_dict = {
#         'input': ecg_raw, 
        'raw': ECG_BW_raw, 
        'smooth': ECG_BW_smooth,
        'filt': ECG_BW_filt,
        'normed': ECG_BW_normed,
    }

    return ECG_BW_dict

def get_good_beats(sig_beats, i_fiducials, confidence_interv=90):
    # the output indices_SCG refers to the indices of sig_beats of good quality

    indices_beats = arg_CI(np.var(sig_beats,axis=0), confidence_interv=confidence_interv)
    sig_beats = sig_beats[:,indices_beats]
    (i_aos, data_aos) = i_fiducials

    # TODO: replace this with SQI based outlier removal
    indices_beats_aos = arg_CI(i_aos[indices_beats], confidence_interv=confidence_interv)
    indices_SCG = indices_beats[indices_beats_aos]
    
    return indices_SCG

def SCG_AM_extraction(ts, sig_beats, i_fiducials, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

    (i_aos, data_aos) = i_fiducials
    indices_SCG = get_good_beats(sig_beats, i_fiducials, confidence_interv=90)


    ts_SCG_AM = ts[i_beat_peaks[indices_SCG]]
    SCG_AM =  data_aos[indices_SCG]

    SCG_AM_raw = np.interp(ts, ts_SCG_AM, SCG_AM)
    SCG_AM_smooth = get_smooth(SCG_AM_raw, N=int(Fs*br_smoothing_dur))
#     SCG_AM_filt = butter_filter('low', SCG_AM_smooth, highcutoff_br, Fs)
    
#     SCG_AM_filt = get_padded_filt(SCG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    SCG_AM_filt = filt_DS_US(ts, Fs, SCG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    SCG_AM_normed = (SCG_AM_filt - SCG_AM_filt.mean()) / SCG_AM_filt.std()
    
    SCG_AM_dict = {
        'raw': SCG_AM_raw,
        'smooth': SCG_AM_smooth,
        'filt': SCG_AM_filt,
        'normed': SCG_AM_normed,
    }
    return SCG_AM_dict

def SCG_FM_extraction(ts, sig_beats, i_fiducials, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

    (i_aos, data_aos) = i_fiducials
    
    indices_SCG = get_good_beats(sig_beats, i_fiducials, confidence_interv=90)

    i_beat_peaks_sel = i_beat_peaks[indices_SCG]
    
#     ts_SCG_FM = (ts[i_beat_peaks_sel[1:]]+ts[i_beat_peaks_sel[:-1]])/2
#     SCG_FM = 1/np.diff(ts[i_beat_peaks_sel])*60
    
    
    t_arr = ts[i_beat_peaks_sel]+i_aos[indices_SCG]/2/Fs
    ts_SCG_FM = (t_arr[1:]+t_arr[:-1])/2
    SCG_FM = 1/np.diff(t_arr)*60


#     print(ts_SCG_FM.shape, i_beat_peaks_sel.shape, SCG_FM.shape)
#     sys.exit()
#         ts_PEP_FM = ts[i_beat_peaks_sel]+i_aos[indices_SCG]/2/Fs

    
    
#     SCG_FM = 1/np.diff(ts[i_beat_peaks_sel])*60

    indices_good = arg_goodhr(SCG_FM)
    ts_SCG_FM = ts_SCG_FM[indices_good]
    SCG_FM = SCG_FM[indices_good]

    # use median filter to remove glitch
    SCG_FM = medfilt(SCG_FM, 5)
    ts_SCG_FM = ts_SCG_FM[1:-1]
    SCG_FM = SCG_FM[1:-1]
#     print(SCG_FM)
#     print(t_arr, SCG_FM)
#     sys.exit()
    
    SCG_FM_raw = np.interp(ts, ts_SCG_FM, SCG_FM)
    SCG_FM_smooth = get_smooth(SCG_FM_raw, N=int(Fs*br_smoothing_dur))
#     SCG_FM_filt = butter_filter('low', SCG_FM_smooth, highcutoff_br, Fs)

#     SCG_FM_filt = get_padded_filt(SCG_FM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    SCG_FM_filt = filt_DS_US(ts, Fs, SCG_FM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    SCG_FM_normed = (SCG_FM_filt - SCG_FM_filt.mean()) / SCG_FM_filt.std()


    SCG_FM_dict = {
#         'aaa': SCG_FM,
        'raw': SCG_FM_raw, 
        'smooth': SCG_FM_smooth,
        'filt': SCG_FM_filt,
        'normed': SCG_FM_normed,
    }

    return SCG_FM_dict

def SCG_BW_extraction(ts, scg_raw, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
    
    ts_SCG_BW = ts
    
    SCG_BW_raw = get_padded_filt(scg_raw, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
    SCG_BW_smooth = get_smooth(SCG_BW_raw, N=int(Fs*br_smoothing_dur))    
#     SCG_BW_filt = get_padded_filt(SCG_BW_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
#     SCG_BW_filt = SCG_BW_smooth
   # downsample, filter, then upsample
    SCG_BW_filt = filt_DS_US(ts, Fs, SCG_BW_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    SCG_BW_normed = (SCG_BW_filt - SCG_BW_filt.mean()) / SCG_BW_filt.std() # this is faster

    SCG_BW_dict = {
        'raw': SCG_BW_raw, 
        'smooth': SCG_BW_smooth,
        'filt': SCG_BW_filt,
        'normed': SCG_BW_normed,
    }

    return SCG_BW_dict


def PEP_FM_extraction(ts, sig_beats, i_fiducials, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

    (i_aos, data_aos) = i_fiducials

#     indices_SCG = get_good_beats(sig_beats, i_fiducials, confidence_interv=90)

#     i_beat_peaks_sel = i_beat_peaks[indices_SCG]
    indices_SCG = i_aos
    i_beat_peaks_sel = i_beat_peaks
#     print(indices_SCG.shape, i_beat_peaks_sel.shape)
    
    ts_PEP_FM = ts[i_beat_peaks_sel]+indices_SCG/2/Fs
    PEP_FM = i_aos/Fs
    

    
    # TODO: implement a arg_goodPEP(PEP) function
    
#     indices_good = arg_goodhr(PEP_FM)
#     ts_PEP_FM = ts_PEP_FM[indices_good]
#     PEP_FM = PEP_FM[indices_good]

    PEP_FM_raw = np.interp(ts, ts_PEP_FM, PEP_FM)
    PEP_FM_smooth = get_smooth(PEP_FM_raw, N=int(Fs*br_smoothing_dur))
    
#     PEP_FM_filt = butter_filter('low', PEP_FM_smooth, highcutoff_br, Fs)
#     PEP_FM_filt = get_padded_filt(PEP_FM_smooth, filter_padded=filter_padded, lowcutoff=0.1, highcutoff=highcutoff_br, Fs=Fs)


#     PEP_FM_filt = get_padded_filt(PEP_FM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    PEP_FM_filt = filt_DS_US(ts, Fs, PEP_FM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    PEP_FM_normed = (PEP_FM_filt - PEP_FM_filt.mean()) / PEP_FM_filt.std()


    PEP_FM_dict = {
        'raw': PEP_FM_raw, 
        'smooth': PEP_FM_smooth,
        'filt': PEP_FM_filt,
        'normed': PEP_FM_normed,
    }
    
    return PEP_FM_dict

# sig = ecg_beats2[:,0]
def ECG_SR_extraction(ts, sig_beats, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1, offset_SR=100):
    
    i=offset_SR-1
    ts_ECG_SR = ts[i_beat_peaks]
    sig_diff = np.gradient(sig_beats,axis=0)

    indicies_valleys = np.argsort(sig_diff,axis=0) # small to large, pick first and >i
    indicies_peaks = np.argsort(sig_diff*-1,axis=0) # large to small, pick first and < i

    i_slopemin = np.zeros(indicies_valleys.shape[1])
    for i_col in range(indicies_valleys.shape[1]):
        indices = indicies_valleys[:,i_col]
        i_slopemin[i_col] = indices[indices>=i][0]

        
    i_slopemax = np.zeros(indicies_peaks.shape[1])
    for i_col in range(indicies_peaks.shape[1]):
        indices = indicies_peaks[:,i_col]
        i_slopemax[i_col] = indices[indices<=i][0]

    i_slopemin = i_slopemin.astype(int)
    i_slopemax = i_slopemax.astype(int)

    slopemin = sig_diff[i_slopemin, np.arange(sig_diff.shape[1])]
    slopemax = sig_diff[i_slopemax, np.arange(sig_diff.shape[1])]

    sloperange = slopemax-slopemin


    ECG_SR_raw = np.interp(ts, ts_ECG_SR, sloperange)
    ECG_SR_smooth = get_smooth(ECG_SR_raw, N=int(Fs*br_smoothing_dur))
    
#     ECG_SR_filt = butter_filter('low', ECG_SR_smooth, highcutoff_br, Fs)
#     ECG_SR_filt = get_padded_filt(ECG_SR_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    ECG_SR_filt = filt_DS_US(ts, Fs, ECG_SR_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    ECG_SR_normed = (ECG_SR_filt - ECG_SR_filt.mean()) / ECG_SR_filt.std()

    ECG_SR_dict = {
        'raw': ECG_SR_raw, 
        'smooth': ECG_SR_smooth,
        'filt': ECG_SR_filt,
        'normed': ECG_SR_normed,
    }
    return ECG_SR_dict

def ECG_PCA_extraction(ts, sig_beats, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1, offset_SR=100):
    
    i=offset_SR-1
    ts_ECG_PCA = ts[i_beat_peaks]
    
    pca = PCA(n_components=1)
    sig_beats_norm = sig_beats-sig_beats.mean(axis=0)
    sig_beats_norm = sig_beats_norm.T

    # Fit the PCA and transform the data
    sig_pca = pca.fit_transform(sig_beats_norm)[:,0]

    ECG_PCA_raw = np.interp(ts, ts_ECG_PCA, sig_pca)
    ECG_PCA_smooth = get_smooth(ECG_PCA_raw, N=int(Fs*br_smoothing_dur))
    
    # downsample, filter, then upsample
    ECG_PCA_filt = filt_DS_US(ts, Fs, ECG_PCA_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    ECG_PCA_normed = (ECG_PCA_filt - ECG_PCA_filt.mean()) / ECG_PCA_filt.std()

    ECG_PCA_dict = {
        'raw': ECG_PCA_raw, 
        'smooth': ECG_PCA_smooth,
        'filt': ECG_PCA_filt,
        'normed': ECG_PCA_normed,
    }
    return ECG_PCA_dict



def SCGMAG_AM_extraction(ts, sig_beats, i_beat_peaks, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):
    # square
    scg_beats_sqr = sig_beats**2

    scg_beats_smooth = np.zeros(scg_beats_sqr.shape)

    for i_sample in range(scg_beats_smooth.shape[1]):
        scg_beats_smooth[:,i_sample] = get_smooth(scg_beats_sqr[:,i_sample], N=int(Fs*width_QRS))

    # ecg_smooth = np.diff(ecg_smooth, append=ecg_smooth[-1])

    i_peaks_mag = scg_beats_smooth[:int(width_QRS*Fs),:].argmax(axis=0)

    ts_SCGMAG_AM = ts[i_beat_peaks]
    SCGMAG_AM = scg_beats_smooth[i_peaks_mag, np.arange(i_peaks_mag.shape[0])]

    SCGMAG_AM_raw = np.interp(ts, ts_SCGMAG_AM, SCGMAG_AM)
    SCGMAG_AM_smooth = get_smooth(SCGMAG_AM_raw, N=int(Fs*br_smoothing_dur))
    
#     SCGMAG_AM_filt = butter_filter('low', SCGMAG_AM_smooth, highcutoff_br, Fs)
#     SCGMAG_AM_filt = get_padded_filt(SCGMAG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample
    SCGMAG_AM_filt = filt_DS_US(ts, Fs, SCGMAG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    
    SCGMAG_AM_normed = (SCGMAG_AM_filt - SCGMAG_AM_filt.mean()) / SCGMAG_AM_filt.std()

    SCGMAG_AM_dict = {
        'raw': SCGMAG_AM_raw, 
        'smooth': SCGMAG_AM_smooth,
        'filt': SCGMAG_AM_filt,
        'normed': SCGMAG_AM_normed,
    }
    return SCGMAG_AM_dict

# Edit: added on 3/22
def SCG_AMpt_extraction(ts, scg_filt, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1, i_R_peaks=None, N_channels=1):
    # inspired by PT
    

    i_peaks, scg_dict = find_SCGpt_peaks(scg_filt, Fs, i_R_peaks, N_channels)

    scg_smooth = scg_dict['scg_smooth']

#     i_peaks = find_SCGpt_peaks(scg_filt, Fs)

        
    SCG_AMpt = scg_smooth[i_peaks]
    ts_SCG_AMpt = ts[i_peaks]
    
    # useful to deal with pt methods
    SCG_AMpt_median = np.median(SCG_AMpt)
    
#     SCG_AMpt_med = medfilt(SCG_AMpt, 501)
    SCG_AMpt_med = medfilt(SCG_AMpt, math.ceil(Fs / 2.) * 2 + 1)
    indices_good = np.where(np.abs(SCG_AMpt_med-SCG_AMpt)<SCG_AMpt_median*5)[0]
#     print(indices_good, SCG_AMpt_med.shape, SCG_AMpt.shape, SCG_AMpt_median.shape)

    bad_ratio = 1-indices_good.shape[0]/SCG_AMpt.shape[0]
    print('\t\t [SCG_AMpt] reject {:.2f}% beats (diff with SCG_AMpt_median < SCG_AMpt_median*5)'.format(bad_ratio*100))

    ts_SCG_AMpt = ts_SCG_AMpt[indices_good]
    SCG_AMpt = SCG_AMpt[indices_good]

    
#     # beat2beat HR won't change with more than 20bpm (TODO: look up lit)
#     ECG_FM_med = medfilt(SCG_AMpt, 501)
#     indices_good = np.where(np.abs(ECG_FM_med-ECG_FM)<20)[0]
#     ts_ECG_FM = ts_ECG_FM[indices_good]
#     ECG_FM = ECG_FM[indices_good]


    SCG_AMpt_raw = np.interp(ts, ts_SCG_AMpt, SCG_AMpt)
    SCG_AMpt_smooth = get_smooth(SCG_AMpt_raw, N=int(Fs*br_smoothing_dur))

    #     SCG_AMpt_filt = get_padded_filt(SCG_AMpt_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
   # downsample, filter, then upsample

    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    SCG_AMpt_filt = filt_DS_US(ts, Fs, SCG_AMpt_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)

    
    SCG_AMpt_normed = (SCG_AMpt_filt - SCG_AMpt_filt.mean()) / SCG_AMpt_filt.std()

    SCG_AMpt_dict = {
        'raw': SCG_AMpt_raw, 
        'smooth': SCG_AMpt_smooth,
        'filt': SCG_AMpt_filt,
        'normed': SCG_AMpt_normed,
        'scg_dict': scg_dict,
        'i_peaks': i_peaks,
    }
    
    
    debug_SCG_AMpt = False
    if debug_SCG_AMpt:
        fig, axes = plt.subplots(4,1, figsize=(15,8), dpi=80)

        axes[0].plot(ts, scg_filt)
        axes[0].set_xlim(0,150)

        axes[1].plot(ts, scg_smooth**0.5 *2, alpha=0.5)
        axes[1].scatter(ts_SCG_AMpt, SCG_AMpt**0.5 *2)
        axes[1].set_xlim(0,150)

        axes[2].plot(ts, SCG_AMpt_dict['raw'])
        axes[2].set_xlim(0,150)

        axes[3].plot(ts, SCG_AMpt_dict['smooth'])
        axes[3].set_xlim(0,150)


        sys.exit()
    
    return SCG_AMpt_dict

# TBD
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

def find_SCGpt_peaks(scg_filt, Fs, i_R_peaks=None, N_channels=1):
#         scg_filt = get_padded_filt(scg_raw, filter_padded=filter_padded, lowcutoff=1, highcutoff=40, Fs=Fs)


    if N_channels==1:
        scg_diff = np.gradient(scg_filt)
        scg_sqr = scg_diff**2
        scg_smooth = get_smooth(scg_sqr, N=int(Fs*width_QRS))
    else:
        scg_norm = 0
        scg_diff = np.zeros((scg_filt.shape[0],0))
        scg_sqr = np.zeros((scg_filt.shape[0],0))

        for i_ch in range(N_channels):
            
            scg_diff_i = np.gradient(scg_filt[:,i_ch])
            scg_diff = np.c_[scg_diff, scg_diff_i]

            scg_sqr_i = scg_diff_i**2
            scg_sqr = np.c_[scg_sqr, scg_sqr_i]
            scg_smooth_i = get_smooth(scg_sqr_i, N=int(Fs*width_QRS))
            scg_norm += scg_smooth_i

            scg_smooth = scg_norm **0.5

        # scg_norm = (scgX**2 + scgY**2 + scgZ**2) **0.5
#         scg_norm = (get_smooth(scgX_diff**2, N=int(Fs*width_QRS)) + get_smooth(scgY_diff**2, N=int(Fs*width_QRS)) + get_smooth(scgZ_diff**2, N=int(Fs*width_QRS))) **0.5

#     hr_max = label_range_dict['heart_rate_cosmed'][1]
    hr_max = 200
    # no RR wave can occur in less than 200 msec distance
    # distance_min = 0.2 * FS_ECG # (1000 samples/s) / (120 beats/min) * (60s/min)
    distance_min = Fs / hr_max * 60 # (1000 samples/s) / (120 beats/min) * (60s/min)
    i_peaks, _ = find_peaks(scg_smooth, distance=distance_min)

    indices_CI = arg_CI(scg_smooth[i_peaks], confidence_interv=50)
    thre = scg_smooth[i_peaks[indices_CI]].mean()*0.2


    i_peaks, _ = find_peaks(scg_smooth, distance=distance_min, height=thre)
    

    if i_R_peaks is not None:
        # SCGpt peak locations (Nx1) - i_R_peaks locations (1xM)
        distance_R_SCGpt = np.min(np.abs(i_peaks[:,None] - i_R_peaks[None,:]),axis=1) # NxM matrix -> Nx1 vector
        mask_close2R = distance_R_SCGpt < int(Fs*width_QRS)
        i_peaks = i_peaks[mask_close2R]
    

    scg_dict = {
        'scg_diff': scg_diff,
        'scg_sqr': scg_sqr,
        'scg_smooth': scg_smooth,
    }
    
    return i_peaks, scg_dict

def PPG_AM_extraction(ts, i_beat_peaks, ppg_fiducials, filter_padded, Fs, highcutoff_br=1, br_smoothing_dur=1):

    
    # i_peaks_mag = scg_beats_smooth[:int(width_QRS*Fs),:].argmax(axis=0)

    ts_ppg_AM = ts[i_beat_peaks]

    # SCGMAG_AM = scg_beats_smooth[i_peaks_mag, np.arange(i_peaks_mag.shape[0])]

    PPG_AM_raw = np.interp(ts, ts_ppg_AM, ppg_fiducials)
    PPG_AM_smooth = get_smooth(PPG_AM_raw, N=int(Fs*br_smoothing_dur))

    #     SCGMAG_AM_filt = butter_filter('low', SCGMAG_AM_smooth, highcutoff_br, Fs)
    #     SCGMAG_AM_filt = get_padded_filt(SCGMAG_AM_smooth, filter_padded=filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, Fs=Fs)
    # downsample, filter, then upsample

    if Fs >100:
        downsample_factor = 5
    else:
        downsample_factor = 1
    PPG_AM_filt = filt_DS_US(ts, Fs, PPG_AM_smooth, filter_padded, lowcutoff=0.08, highcutoff=highcutoff_br, downsample_factor=downsample_factor)


    PPG_AM_normed = (PPG_AM_filt - PPG_AM_filt.mean()) / PPG_AM_filt.std()

    PPG_AM_dict = {
        'raw': PPG_AM_raw, 
        'smooth': PPG_AM_smooth,
        'filt': PPG_AM_filt,
        'normed': PPG_AM_normed,
    }

    return PPG_AM_dict


def get_surrogate_names(column_names):
    column_names = column_names[(column_names!='time') & (column_names!='br_sim_sig') & (column_names!='subject_id') & (column_names!='task')]
    column_names = column_names[(column_names!='resp_cosmed') & (column_names!='RR_cosmed') & (column_names!='HR_cosmed') & (column_names!='VT_cosmed') 
                                & (column_names!='task_id') & (column_names!='domain')]
    surrogate_names = column_names
    return surrogate_names
        


def get_RR_est(sig, Fs):
    N = sig.shape[0]
    freq = np.linspace (0.0, Fs/2, int (N/2))
    
    sig_spectral = fft(sig)
    Pxx = 2/N * np.abs (sig_spectral [0:np.int (N/2)])
    
    RR_range = np.asarray([label_range_dict['RR'][0]/60, label_range_dict['RR'][1]/60])
    
    mask = (freq>=RR_range[0]) & (freq<=RR_range[1])
    freq = freq[mask]
    Pxx = Pxx[mask]
    
    RR_est = freq[np.argmax(Pxx)]*60
    return RR_est, freq, Pxx

def get_RR_cutoff(resp_ref, Fs):
    RR_est, freq, Pxx = get_RR_est(resp_ref, Fs)
    ECG_RR_freq =  RR_est/60
    lowcutoff = ECG_RR_freq - 0.1
    highcutoff = ECG_RR_freq + 0.1

    if lowcutoff<RR_range[0]:
        lowcutoff = RR_range[0]

    if highcutoff>RR_range[1]:
        highcutoff = RR_range[1]
        
    return ECG_RR_freq, lowcutoff, highcutoff

    


def plot_surrogate_resp(df_resp, outputdir=None, fig_name=None, show_plot=False):        
    t_sig = df_resp['time'].values
    t_sig = t_sig - t_sig[0]
    t_dur = t_sig[-1] - t_sig[0]
    t_start = t_sig[0]
    t_end = t_sig[-1]

    RR_label = df_resp['RR_cosmed'].values
    resp_cosmed = df_resp['resp_cosmed'].values
    br_sim_sig = df_resp['br_sim_sig'].values
    
    subject_id = df_resp['subject_id'].values[0]
    task_name = df_resp['task'].values[0]
    
    
#     surrogate_names = np.asarray(df_resp.columns)
#     surrogate_names = surrogate_names[(surrogate_names!='time') & (surrogate_names!='RR_cosmed') & (surrogate_names!='subject_id') & (surrogate_names!='task')]
    column_names = np.asarray(df_resp.columns)
    surrogate_names = get_surrogate_names(column_names)
    surrogate_names = np.r_[surrogate_names, ['resp_cosmed']]    
    
    N_sur = len(surrogate_names)

    fig, axes = plt.subplots(N_sur, 1, figsize=(20, N_sur*0.7), gridspec_kw = {'wspace':0, 'hspace':0}, dpi=60)

    # fontdict = {'size': 20}
    fontsize = 15

    for i_ax, (ax, surrogate_name) in enumerate(zip(axes, surrogate_names)):
        ax.plot(t_sig, df_resp[surrogate_name], alpha=0.5 ,zorder=1)
    #     ax.plot(t_sig, df_resp[surrogate_name], color=color_dict[colornames[list_sigs.index('ECG_filt')]], alpha=0.5 ,zorder=1)

        ax.set_xlim(t_start, t_end) # remove the weird white space at the beg and end of the plot
        ax.grid('on', linestyle='--')
        # no x ticks except for the bottom ax
        if i_ax<len(axes)-1:
            ax.set_xticklabels([])
        # add y ticks to all axes
        ax.tick_params(axis='y', which='both', labelsize=fontsize-3)

        ax.set_ylabel(surrogate_name, fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=100)
        ax_no_top_right(ax)

    ax.set_xlabel('time (sec)', fontsize=fontsize)

    if fig_name is None:
        fig_name = 'surrogate_label_{}_{}.png'.format(fig_name)

    # fig.suptitle(fig_name, fontsize=fontsize*1.5)
    fig.tight_layout()

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name, facecolor=fig.get_facecolor(),  transparent=True)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


def plot_HM(df, x_name, y_name, value_name, fig_name=None, outputdir=None, show_plot=False):

    fig, ax = plt.subplots(1,1,figsize=(8,8),dpi=100)
    
#     column_names = np.asarray(df.columns)
#     surrogate_names = get_surrogate_names(column_names)
#     print(surrogate_names)
#     keep_names = np.r_[surrogate_names, [y_name, value_name]]    
#     df = df[keep_names]

#     pivot = df.pivot(index=x_name, columns=y_name, values=value_name)
    pivot = df.pivot_table(index=x_name, columns=y_name, values=value_name)

    sns.heatmap(pivot, vmin=0, vmax=40, annot=True,annot_kws={"size": 5}, fmt='.2f', ax=ax, linewidths=.5)
#     plt.show()

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if fig_name is None:
            fig_name = 'RR_est'

        fig.savefig(outputdir + fig_name, bbox_inches='tight', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
    plt.show()

def plot_RR_est_error(df_RR_est_subs, fig_name=None, outputdir=None, show_plot=False):
    fig, ax = plt.subplots(1,1, figsize=(10,5), dpi=80)
    
    ax = sns.boxplot(x='surrogate_names', y='RR_mae', data=df_RR_est_subs, ax=ax)
    ax.set_ylabel('RR_mae (BPM)')
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
    fig.tight_layout()

    fig.subplots_adjust(wspace=0, hspace=0)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        if fig_name is None:
            fig_name = 'RR_est_error'
        else:
            fig_name = fig_name

        fig.savefig(outputdir + fig_name,bbox_inches='tight', transparent=False)

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
    plt.show()
        
        

def get_df_RR_est(df_resp_subs):
    # this function estiamte RR for each subject, and for each task
    # df_RR_est is a dataframe that stores RR_est, RR_label, surrogate_names, subject_id, task

    df_RR_est = pd.DataFrame()
    for subject_id in df_resp_subs['subject_id'].unique():
        df_resp_sub = df_resp_subs[df_resp_subs['subject_id']==subject_id]

        for task_name in df_resp_sub['task'].unique():

            df_resp = df_resp_sub[df_resp_sub['task']==task_name]
            column_names = np.asarray(df_resp.columns)
            surrogate_names = get_surrogate_names(column_names)
#             print(surrogate_names)
#             surrogate_names = surrogate_names[(surrogate_names!='time') & (surrogate_names!='br_sim_sig') & (surrogate_names!='subject_id') & (surrogate_names!='task')]
#             surrogate_names = surrogate_names[(surrogate_names!='resp_cosmed') & (surrogate_names!='RR_cosmed') & (surrogate_names!='task_id') & (surrogate_names!='domain')]
            
#             N_sur = len(surrogate_names)
#             print(N_sur, surrogate_names)
            
            RR_label, _, _ = get_RR_est(df_resp['br_sim_sig'].values, FS_RESAMPLE_resp)

            for surrogate_name in surrogate_names:
                RR_est, _, _ = get_RR_est(df_resp[surrogate_name].values, FS_RESAMPLE_resp)

                if surrogate_name=='br_sim_sig':
                    continue

                df_RR_est = df_RR_est.append(
                    pd.DataFrame({
                        'RR_est': RR_est,
                        'RR_label': RR_label,
                        'surrogate_names': surrogate_name,
                        'subject_id': subject_id,
                        'task': task_name,
                    }, index=[0])
                )

    df_RR_est['RR_mae'] = np.abs(df_RR_est['RR_est'] - df_RR_est['RR_label'])
    return df_RR_est


#         sig_filt = get_padded_filt_DSwrapper(sig, filter_padded=5, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=Fs, downsample_factor=5)

#         df_resp_subs[key+'_respsur_filtered'] = sig_filt
#         analytic_signal = hilbert(sig_filt)
#         amplitude_envelope = np.abs(analytic_signal)

#         amplitude_envelope_smooth = get_smooth(amplitude_envelope, N=int(20*Fs))

#         df_resp_subs[key+'_respsur_filtered_env'] = amplitude_envelope


def get_corr(y1, y2):
    n = len(y1)
    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])
    return corr

def get_df_resp_PCC(df_resp_subs):

    df_resp_PCC = pd.DataFrame()
    for subject_id in df_resp_subs['subject_id'].unique():
        df_resp_sub = df_resp_subs[df_resp_subs['subject_id']==subject_id]

        for task_name in df_resp_sub['task'].unique():
            df_resp = df_resp_sub[df_resp_sub['task']==task_name]
            
#             surrogate_names = np.asarray(df_resp.columns)
#             surrogate_names = surrogate_names[(surrogate_names!='time') & (surrogate_names!='RR_cosmed') & (surrogate_names!='subject_id') & (surrogate_names!='task')]
#             surrogate_names = surrogate_names[(surrogate_names!='resp_cosmed') & (surrogate_names!='RR_cosmed') & (surrogate_names!='task_id') & (surrogate_names!='domain')]

            column_names = np.asarray(df_resp.columns)
            surrogate_names = get_surrogate_names(column_names)
        
            N_sur = len(surrogate_names)

            for resp_name_1 in surrogate_names:
                for resp_name_2 in surrogate_names:
                    PCC = get_PCC(df_resp[resp_name_1].values, df_resp[resp_name_2].values)

                    
#                     print()
                    corr_vector = get_corr(df_resp[resp_name_1].values, df_resp[resp_name_2].values)
                    xcorr_max = corr_vector.max()

                    df_resp_PCC = df_resp_PCC.append(pd.DataFrame(
                        {
                            'resp_name_1': resp_name_1,
                            'resp_name_2': resp_name_2,
                            'PCC': np.abs(PCC),
                            'xcorr_max': xcorr_max
                        }, index=[0]
                    ))
                
    return df_resp_PCC

def get_kurtosis(sig):
    return scipy.stats.kurtosis(sig)

def get_RQI_fft(fft_matrix):
    # fft_matrix = model_out_dict_TEST['out_concat'][:,0,:]
    # fft_matrix: dim = (# of samples, # of freq bins)
    # RQI_fft: dim = (# of samples)

    fft_max = (fft_matrix[:,:-1]+fft_matrix[:,1:]).max(axis=1)
    fft_sum = fft_matrix.sum(axis=1)

    RQI_fft = fft_max/fft_sum
    # moving_average(fft_matrix, w=2).shape
    return RQI_fft

def get_RQI_kurtosis(fft_matrix):
    RQI_kurtosis = get_kurtosis(fft_matrix.T)
    return RQI_kurtosis