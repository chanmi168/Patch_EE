import numpy as np
import os
import sys
import math
from math import sin

from scipy.stats import pearsonr

# from scipy.io import loadmat
# import scipy
# from scipy import signal
# from scipy.fftpack import fft, ifft
# from scipy.signal import hilbert, chirp
# from scipy.signal import find_peaks
# from scipy.interpolate import interp1d

from sklearn.metrics import mean_squared_error

# from feature_extraction import *
# from preprocessing import *

def get_RMSE(RR_label, RR_est):
    RMSE = np.sqrt(mean_squared_error(RR_label, RR_est))
    return RMSE


def get_MAE(RR_label, RR_est):
    MAE_mean = np.abs(RR_label - RR_est).mean()
    MAE_std = np.abs(RR_label - RR_est).std()
    return MAE_mean, MAE_std

def get_MAPE(RR_label, RR_est):
    MAPE_mean = (np.abs   ( (RR_label- RR_est) / (RR_label+sys.float_info.epsilon))    ).mean()
    MAPE_std = (np.abs   ( (RR_label- RR_est) / (RR_label+sys.float_info.epsilon))    ).std()
#     MAPE_std = (np.abs(RR_label- RR_est) / (RR_label+sys.float_info.epsilon)).std()
    return MAPE_mean, MAPE_std

def get_CoeffDeterm(label=0, predictions=0):    
    SStot = np.square(label-label.mean()).sum()
    SSres = np.square(predictions-label).sum()

    Rsquared = 1 - SSres/SStot

    return Rsquared


def get_PCC(label, est):
    PCC, _ = pearsonr(label, est)
    return PCC
