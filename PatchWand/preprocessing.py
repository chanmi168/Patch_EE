import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
import os

import matplotlib
import matplotlib.pyplot as plt

from scipy.io import loadmat
import scipy
from scipy import signal
from scipy.fftpack import fft, ifft
pd.set_option('display.max_columns', 500)

# from preprocessing import *
from setting import *

i_seed = 0
  
# def sig_smoothing(data, lowcutoff, fs=2000):
#     sig_filt = np.zeros(data.shape)
#     for i in range(data.shape[1]):
#         sig_filt[:, i] = butter_filter('high', data[:,i], lowcutoff, fs)
#     return sig_filt
  
# TODO: move this to PPG_module
def PPG_argfoot(data_pulse, fs=1000):
    # usage: i_min, i_derivmax, i_PTT, data_derivmax, data_min = argfoot(data_pulse)
    
    # 1. remove signal lower than PPG_lowcutoff Hz
    data_pulse_raw = data_pulse
    
#     sig_filt = np.zeros(data_pulse.shape)
#     for i in range(data_pulse.shape[1]):
#         sig_filt[:, i] = butter_filter('high',data_pulse[:,i],PPG_lowcutoff,fs)
#     data_pulse = sig_filt

#     n_pca = 2
#     if data_pulse.shape[1] >= n_pca:
#         PCA_transformer = PCA(n_components=n_pca)
#         data_pulse = PCA_transformer.inverse_transform(PCA_transformer.fit_transform(data_pulse))

    # find minimum
    i_sorted = np.argsort(data_pulse, axis=0)
    i_hidden = np.argmax(i_sorted<data_pulse.shape[0]/2,axis=0)
    i_min = i_sorted[ i_hidden, range(i_sorted.shape[1])]
    
    # find max fist derivative
    data_deriv = np.diff(data_pulse,axis=0)
    i_derivsorted = np.argsort(data_deriv, axis=0)[::-1]
    i_hidden = np.argmax(i_derivsorted<data_pulse.shape[0]/2,axis=0)
    i_derivmax = i_derivsorted[ i_hidden, range(i_derivsorted.shape[1])]
    
    # find max slope
    max_slopes = np.max(data_deriv, axis=0)

    data_derivmax = data_pulse_raw[ i_derivmax, range(data_pulse.shape[1])]
    data_min = data_pulse_raw[ i_min, range(data_pulse.shape[1])]
    
    # TODO: need to mitiage zero slopes
    # find i_foot index
    i_foot = i_derivmax - (data_derivmax - data_min) / max_slopes

    return i_min, i_derivmax, i_foot, data_derivmax, data_min

# TODO: move this to SCG_module
def SCG_ao(data_scg, fs=2000):
    
    # 1. remove signal lower than SCG_lowcutoff Hz
#     data_scg_filt = np.zeros(data_scg.shape)
#     for i in range(data_scg.shape[1]):
#         data_scg_filt[:, i] = butter_filter('high',data_scg[:,i],SCG_lowcutoff,fs)
#     data_scg_filt = data_scg
#     n_pca = 2
#     if data_scg_filt.shape[1] >= n_pca:
#         PCA_transformer = PCA(n_components=n_pca)
#         data_scg_filt = PCA_transformer.inverse_transform(PCA_transformer.fit_transform(data_scg_filt))

    data_scg_derv1 = np.diff(data_scg, axis=0)
    
    condi1 = (data_scg > 0)[:-2,:]
    condi2 = (np.diff(np.sign(data_scg_derv1), axis=0) < 0)*1
    # check when SCG first change sign
    rows, cols = np.where((condi1) & (condi2))
    
    i_peaks_ao = np.zeros(data_scg.shape[1])
    for i_col in range(data_scg.shape[1]):
        if (cols==i_col).sum() > 1:
#             i_peaks_ao[i_col] = rows[cols==i_col][0]
            j = np.argmax(rows[cols==i_col]>50)
            i_peaks_ao[i_col] = rows[cols==i_col][j]
    
        else:
            i_peaks_ao[i_col] = 0

    i_peaks_ao += 1
    data_SCGao = data_scg[ i_peaks_ao.astype(int), range(data_scg.shape[1])]

    return i_peaks_ao.astype(int), data_SCGao

# all the arg function
def arg_lowvar(data, lowcutoff=20, highcutoff=5):
    thre_max = np.median(data.var(axis=0))*highcutoff
    thre_min = np.median(data.var(axis=0))/lowcutoff
    indince_clean = np.where((data.var(axis=0)<thre_max) & (data.var(axis=0)>thre_min))[0][:-1]
    return indince_clean

def arg_CI(data, confidence_interv=80):
    value_up = np.percentile(data, int((100-confidence_interv)/2+confidence_interv))
    value_lw = np.percentile(data, int((100-confidence_interv)/2))
    indices_CI = np.where((data<=value_up) & (data>=value_lw))[0]
    return indices_CI

def get_interdecile_range(data, confidence_interv=90):
    indices_CI = arg_CI(data, confidence_interv=confidence_interv) # get middle confidence_interv %
    decile_min = data[indices_CI].min()
    decile_max = data[indices_CI].max()
    interdecile_range = decile_max - decile_min
    return decile_min, decile_max, interdecile_range

def arg_sqi(data_sqi, confidence_interv=80):
    value_lw = np.percentile(data_sqi, int(100-confidence_interv))
    indices_good = np.where(data_sqi>=value_lw)[0]

    return indices_good



# def get_smooth(data, N=5):
#     # padding prevent edge effect
#     data_padded = np.pad(data, (N//2, N-1-N//2), mode='edge')
#     data_smooth = np.convolve(data_padded, np.ones((N,))/N, mode='valid') 
#     return data_smooth


# def medfilt (x, k):
#     """Apply a length-k median filter to a 1D array x.
#     Boundaries are extended by repeating endpoints.
#     """
#     assert k % 2 == 1, "Median filter length must be odd."
#     assert x.ndim == 1, "Input must be one-dimensional."
#     k2 = (k - 1) // 2
#     y = np.zeros ((len (x), k), dtype=x.dtype)
#     y[:,k2] = x
#     for i in range (k2):
#         j = k2 - i
#         y[j:,i] = x[:-j]
#         y[:j,i] = x[0]
#         y[:-j,-(i+1)] = x[j:]
#         y[-j:,-(i+1)] = x[-1]
#     return np.median (y, axis=1)


def arg_smooth(data, N=5, thre_scale=5):
    # remove outliers that are too far away from moving average
    # 5 point moving average
    data_smooth = get_smooth(data, N=N)

    # find outlier threshold
    smooth_diff = np.abs(data-data_smooth)
    indices_CI = arg_CI(smooth_diff, confidence_interv=80)
    thre = smooth_diff[indices_CI].mean()*thre_scale

    # exclude outlier indices
    indices_good = np.where(smooth_diff < thre)[0]
    return indices_good

def arg_goodbr(data, thre_max=40, thre_min=5):
    # exclude outlier indices
    indices_good = np.where((data < thre_max) & (data > thre_min))[0]
    return indices_good

def arg_goodhr(data, thre_max=180, thre_min=40):
    # exclude outlier indices
    indices_good = np.where((data < thre_max) & (data > thre_min))[0]
    return indices_good



def my_ceil(arr, decimal=0):
    return np.ceil(arr*(10**-decimal))/(10**-decimal)

def my_floor(arr, decimal=0):
    return np.floor(arr*(10**-decimal))/(10**-decimal)


def merge_dict(dict1, dict2):
    return {**dict1, **dict2}