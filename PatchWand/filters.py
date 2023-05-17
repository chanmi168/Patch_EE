from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert, chirp
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import numpy as np

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_filter(btype, data, cutoff, fs, order=5):
    
    if btype == 'high': 
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    elif btype == 'low': 
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y
    
def get_filt(sig_raw, Fs, lowcutoff=1, highcutoff=40):
    sig_filt = sig_raw
    if lowcutoff is not None:
        sig_filt = butter_filter('high', sig_filt, lowcutoff, Fs)
    
    if highcutoff is not None:
        sig_filt = butter_filter('low', sig_filt, highcutoff, Fs)

    return sig_filt

def get_padded_filt(sig_raw, filter_padded=5, lowcutoff=0.1, highcutoff=40, Fs=1000):
    
    offset = int(filter_padded*Fs) # reversely pad 10 sec to beginning and end of the signal
    sig_raw = np.r_[sig_raw[0:offset][::-1], sig_raw, sig_raw[-offset:][::-1]]
    sig_filt = get_filt(sig_raw, Fs, lowcutoff=lowcutoff, highcutoff=highcutoff)
    sig_filt = sig_filt[offset:-offset]
    
    return sig_filt


def get_padded_filt_DSwrapper(sig, filter_padded=5, lowcutoff=0.1, highcutoff=40, Fs=1000, downsample_factor=5):
#     downsample_factor = 5
    # ts = patch_dict['time']
    ts = np.arange(sig.shape[0])/Fs
    Fs_DS = Fs // downsample_factor # 100 Hz
    ts_DS = ts[::downsample_factor]

    sig_DS = np.interp(ts_DS, ts, sig)    
    sig_filt = get_padded_filt(sig_DS, filter_padded=filter_padded, lowcutoff=lowcutoff, highcutoff=highcutoff, Fs=Fs_DS)
    sig_filt = np.interp(ts, ts_DS, sig_filt) 
    
    return sig_filt


def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w
    return np.convolve(x, np.ones(w), 'same') / w

# def my_ceil(arr, decimal=0):
#     return np.ceil(arr*(10**-decimal))/(10**-decimal)

# def my_floor(arr, decimal=0):
#     return np.floor(arr*(10**-decimal))/(10**-decimal)



def get_smooth(data, N=5):
    # padding prevent edge effect
    data_padded = np.pad(data, (N//2, N-1-N//2), mode='edge')
    data_smooth = np.convolve(data_padded, np.ones((N,))/N, mode='valid') 
    return data_smooth


def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)

def get_padded_medfilt(x, k, offset=5):
    
    x = np.r_[x[0:offset][::-1], x, x[-offset:][::-1]] # reversely pad offset datapoints to beginning and end of the signal
    x_filt = medfilt(x, k)
    x_filt = x_filt[offset:-offset]
    
    return x_filt
