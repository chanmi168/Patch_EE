import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline

import numpy as np

from setting import *
from filters import *
from preprocessing import *
from segmentation import *
# from segmentation import beat_segmentation

deviation_thre = 3

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


