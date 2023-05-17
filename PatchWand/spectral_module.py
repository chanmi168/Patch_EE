import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
# %matplotlib inline
import matplotlib.gridspec as gs

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

# use this to visualize spectral representation
def get_RR_range():
    # this is the RR range designed to capture a wider spectral range of STFT for better U-Net computation
# #     return np.asarray(label_range_dict['br'])/60
#     return np.asarray([0.02, 0.8])
    return np.asarray([0.02, 1])


def plot_sig_spectrogram(sig, br, FS, windowing_params, cmap='viridis', show_plot=True):
    x = sig
    N = x.shape[0]

    t_sig = np.linspace(0, (N-1)/FS, N, endpoint=False)

    f, t, Sxx = signal.spectrogram(x, FS, nperseg=128)

    fig, (ax_raw, ax_br, ax_spectrogram) = plt.subplots(3, 1, figsize=(15,5))

    ax_raw.plot(t_sig, x, 'red')
    ax_raw.set_xlim([0,t_sig[-1]])
    
    ax_br.plot(t_sig, br, 'steelblue')
    ax_br.set_xlim([0,t_sig[-1]])

    fontdict = {'size': 10}

    NFFT = int(FS*windowing_params['window_size'])  # window_size sec window
#     NFFT = int(1024*10)  # 5ms window
    noverlap = int(NFFT*windowing_params['overlap'])
    

#     for i_axis in range(v_acc.shape[1]-1): # don't do it for HR
#         Pxx = scipy.fftpack.fft(data[indices[i_plot],:,i_axis])
    
#     yf = fft(normalized_tone)
#     xf = fftfreq(N, 1 / SAMPLE_RATE)

    Pxx, freqs, bins, im = ax_spectrogram.specgram(x,Fs=FS, NFFT=NFFT, noverlap=noverlap, cmap=cmap)
    
    ax_spectrogram.set_xlim([0,t_sig[-1]])
    ax_spectrogram.set_ylim([0,1])

    ax_spectrogram.set_xlabel('time (sec)', fontdict=fontdict)
    ax_spectrogram.set_ylabel('Freq (Hz)', fontdict=fontdict)
    
    fig.tight_layout()
    
    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
    
    return Pxx, freqs, bins


def Pxx_segmentation(ts_Pxx, Pxx, i_peaks, start_offset=-50, end_offset=250):
    # segment STFT matrix into sequences of smaller matrices
#     seq_legnth = end_offset - start_offset

    Pxx_seq = []
    ts_Pxx_seq = []
    
    for i, i_peak in enumerate(i_peaks):
        i_start = int(i_peak + start_offset)
        i_end = int(i_peak + end_offset)

        if i_start < 0 or i_end > Pxx.shape[0]:
            continue
        Pxx_seq.append(Pxx[i_start:i_end, :])
        ts_Pxx_seq.append(ts_Pxx[i_start:i_end])

    Pxx_seq = np.stack(Pxx_seq)
    # Pxx_seq.shape = (276, seq_len, N_feature)
    # Pxx_seq = np.transpose(Pxx_seq, (0, 2, 1))
    # Pxx_seq.shape = (276, N_feature, seq_len)

    ts_Pxx_seq = np.stack(ts_Pxx_seq)
    
    return ts_Pxx_seq, Pxx_seq

def get_hr_seq(ts_Pxx_seq, t_sig, hr):
    (M,N) = ts_Pxx_seq.shape

    hr_seq = np.zeros((M,N))

    for i in range(M):
        for j in range(N):
#             index = np.where(t_sig==ts_Pxx_seq[i,j])[0]
            index = np.where(np.around(t_sig, decimals=3)==np.around(ts_Pxx_seq[i,j], decimals=3))[0]
            hr_seq[i,j] = hr[index]
    return hr_seq




# TBD, this is not as good as signal.stft
def get_psd(sig, Fs):
    # keep last dimension of sig the temporal diemnsion
    # the last dimension of yf_scaled will be the spectral dimension
    
    NFFT = sig.shape[-1]
    yf = scipy.fftpack.fft(sig)
    # yf_scaled = 2.0/NFFT * (np.abs(yf[:NFFT//2])**2)
    yf_scaled = 2.0/NFFT * (np.abs(yf[...,:NFFT//2])**2)
    # data[...,mask].shape

    xf = np.linspace(0.0, 1.0/2.0*Fs , NFFT//2)    

    return xf, yf_scaled

def get_PSD_windows_scipy(t_sig, sig, Fs, NFFT, noverlap):
    # already tested, this is equivalent to ax.specgram
    # usage:
    # ts_Pxx, Pxx, freq = get_PSD_windows_scipy(t_sig, sig, Fs, NFFT, noverlap)

    freq, t, Zxx = signal.stft(sig, Fs, noverlap=noverlap, nperseg=NFFT, padded=False, boundary=None)
    Pxx = np.abs(Zxx)**2
    ts_Pxx = t + t_sig[0]

    
#     mask = (freq>0.08) & (freq<0.8)
#     freq = freq[mask] # RR can be as low as 0.08 Hz = 4.8 bpm and as high as 0.8 Hz = 48 bpm 
#     RR_range = np.asarray(label_range_dict['br'])/60
    RR_range = get_RR_range()
    
    
    mask = (freq>RR_range[0]) & (freq<RR_range[-1])
    freq = freq[mask]
    
    Pxx = Pxx[mask,:]
    Pxx = Pxx.T
    return ts_Pxx, Pxx, freq

# TBD: not used in the pipine (exact same outptu as the scipy)
def get_PSD_windows_matplotlib(t_sig, sig, Fs, NFFT, noverlap):

    N = sig.shape[0]
    N_windows = (N-noverlap) // (NFFT-noverlap)

    i_starts = np.arange(0,N_windows)*(NFFT-noverlap)
    i_ends = i_starts + NFFT

    freq = scipy.fftpack.fftfreq(NFFT, d=1/Fs)[:NFFT//2]

#     mask = (freq>0.08) & (freq<0.8)

#     freq = freq[mask] # RR can be as low as 0.08 Hz = 4.8 bpm and as high as 0.8 Hz = 48 bpm 
#     RR_range = np.asarray(label_range_dict['br'])/60
    RR_range = get_RR_range()

    mask = (freq>RR_range[0]) & (freq<RR_range[-1])
    freq = freq[mask]
    
    N_freq = freq.shape[0]

    Pxx = np.zeros((N_windows, N_freq))
    Pxx_label = np.zeros((N_windows, N_freq))
    ts_Pxx = np.zeros(N_windows)

    sig_segments = np.zeros((N_windows, NFFT))

    for i, (i_start, i_end) in enumerate(zip(i_starts, i_ends)):
        ts_Pxx[i] = np.median(t_sig[i_start:i_end])
        xf, yf_scaled = get_psd(sig[int(i_start):int(i_end)], Fs)
        Pxx[i, :] = yf_scaled[mask]
    
    Pxx = Pxx.T
    return ts_Pxx, Pxx, freq


def get_cwtmatr(sig, widths, Fs, wavelet_name='gaus1'):
    # this function will generate cwt matrix with dimension (N_samples, N_frequencies)
    # tuning the scales can give you different images and freq resolution
    
#     # usage:
#     a = 1.9
#     b = 3.0

#     widths = 10**b-np.logspace(a,b, num=40)+10**a
#     widths = widths[::-1]
#     wavelet_name = 'gaus1'

#     sig_cwtmatr, freq_cwt = get_cwtmatr(sig, widths, wavelet_name=wavelet_name)
#     sig_cwtmatr_label, freq_cwt = get_cwtmatr(sig_label, widths, wavelet_name=wavelet_name)

    cwtmatr, freq = pywt.cwt(sig, widths, wavelet_name, sampling_period = 1/Fs)
    
#     RR_range = np.asarray(label_range_dict['br'])/60
    RR_range = get_RR_range()

    mask = (freq>RR_range[0]) & (freq<RR_range[-1])
    freq = freq[mask] # RR can be as low as 0.08 Hz = 4.8 bpm and as high as 0.8 Hz = 48 bpm 
    
    cwtmatr = cwtmatr[mask].T
    return cwtmatr, freq


def extract_windows(t_sig, sig, Fs, windowing_params=None):
    
    norm_param = windowing_params['norm_param']
    if norm_param is not None:
#         sig = get_sur_norm(sig, length_medfilt=length_medfilt)
        sig = get_sur_norm(t_sig, sig, norm_param, debug=False)
#         print(norm_param)

        
    NFFT = int(Fs*windowing_params['window_size'])  # 20 sec per window
    noverlap = int(NFFT*windowing_params['overlap']) # int(60*0.8*Fs) = 4.8 sec * Fs


    ts_Pxx, Pxx, freq = get_PSD_windows_scipy(t_sig, sig, Fs, NFFT, noverlap)

    
    # this is equivalent to roughly N_windows_perim*(1-overlap)*window_size + window_size*overlap
    # == 10 * (1-0.95) * (20 sec) + (20 sec) * (0.95) = 29 sec
    N_windows_seq = windowing_params['seq_len']
    

    N_windows = Pxx.shape[0] - (N_windows_seq - 1) 
    i_starts = np.arange(N_windows)

    ts_Pxx_seq, Pxx_seq = Pxx_segmentation(ts_Pxx, Pxx, i_starts, start_offset=0, end_offset=N_windows_seq)

    return ts_Pxx_seq, Pxx_seq, freq, ts_Pxx, Pxx



def get_envelope(sig):
    analytic_signal = hilbert(sig)
    amplitude_envelope = np.abs(analytic_signal)

    return amplitude_envelope


def get_threshold(ECG_SQI):
    return ECG_SQI.mean()*0.9

# TODO: move to modules
def extract_windows_sqi(t_sig, ts_Pxx_seq, ECG_SQI):
    # sqi_seq has the same dimension as ts_Pxx_seq
    # it's a average sqi of each seg
    
    sqi_seq = np.zeros(ts_Pxx_seq.shape)

    ECG_SQI = df_task['ECG_SQI'].values
    threshold_SQI = get_threshold(ECG_SQI)

    t_start = ts_Pxx_seq-30
    t_end = ts_Pxx_seq+30

    for i in range(sqi_seq.shape[0]):
        for j in range(sqi_seq.shape[1]):
            indices = np.where((t_sig>=t_start[i,j]) & (t_sig<=t_end[i,j]))[0]
            sqi_seq[i,j] = ECG_SQI[indices].mean()
            
    return sqi_seq, threshold_SQI


# plot data snippet in time and TF domains

def inspect_TF_rep(ts, sig, sig_label, windowing_params=None, task_name='Recovery', subject_id='CW0', sig_name='arbitrary', outputdir=None, show_plot=False):
    norm_param = windowing_params['norm_param']
    Fs = windowing_params['Fs']
    NFFT = int(Fs*windowing_params['window_size'])  # 20 window
    noverlap = int(NFFT*windowing_params['overlap']) # int(60*0.8*Fs) = 4.8 sec * Fs

#     i_start = 0
#     i_end = 120000
#     i_end = -1
#     ts = t_sig .
#     sig = sig[i_start:i_end]

#     sig_label = br_sim_sig[i_start:i_end]
#     label = br[i_start:i_end]
    
    if norm_param is not None:
        sig = get_sur_norm(ts, sig, norm_param, debug=False)

    # get STFT
    ts_Pxx, Pxx, freq = get_PSD_windows_scipy(ts, sig, Fs, NFFT, noverlap)
    ts_Pxx, Pxx_label, freq = get_PSD_windows_scipy(ts, sig_label, Fs, NFFT, noverlap)

    # normalize Pxx such that maximum energy at each time step is 1
#     Pxx = Pxx/Pxx.max(axis=1)[:,None]
#     Pxx = (Pxx-Pxx.min())/(Pxx.max()-Pxx.min())
#     Pxx_label = (Pxx_label-Pxx_label.min())/(Pxx_label.max()-Pxx_label.min())



    # plot
    gs0 = gs.GridSpec(4,2, width_ratios=[10,0.1])
    fig = plt.figure(figsize=(10, 8), dpi=50)

    ax1 = fig.add_subplot(gs0[0,0])
    ax2 = fig.add_subplot(gs0[1,0])
    ax3 = fig.add_subplot(gs0[2,0])
    ax4 = fig.add_subplot(gs0[3,0])

    cax2 = fig.add_subplot(gs0[1,1])
    cax4 = fig.add_subplot(gs0[3,1])    
    
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), dpi=50)
    fontsize = 15
    cmap = 'jet'

    i_final = Pxx_label.shape[0]*(NFFT-noverlap)+noverlap
    sig_plot = sig[:i_final]
    label_plot = sig_label[:i_final]
    ts = np.arange(sig[:i_final].shape[0])/Fs + ts[0]
    
    
    ts_Pxx -= ts[0]
    ts -= ts[0]
    
    extent = [ts_Pxx[0], ts_Pxx[-1], freq[0], freq[-1]]
    
    vmin = 0
    vmax = 0.3
    
    ax1.plot(ts, sig_plot)
    ax1.set_xlim(ts_Pxx[0], ts_Pxx[-1])
    ax1.set_title('extracted resp. (time)',fontsize=fontsize)

    im2 = ax2.imshow(Pxx.T, label='out', cmap='viridis', aspect="auto", origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax2.set_xlim(ts_Pxx[0], ts_Pxx[-1])
    ax2.set_ylim(freq[0], freq[-1])
    ax2.set_title('extracted resp. (STFT)',fontsize=fontsize)

    ax3.plot(ts, label_plot)
    ax3.set_xlim(ts_Pxx[0], ts_Pxx[-1])
    ax3.set_title('COSMED resp. (time)',fontsize=fontsize)
    
    im4 = ax4.imshow(Pxx_label.T, label='out', cmap='viridis', aspect="auto", origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax4.set_ylim(freq[0], freq[-1])
    ax4.set_xlim(ts_Pxx[0], ts_Pxx[-1])
    ax4.set_title('COSMED resp. (STFT)',fontsize=fontsize)

    fig.colorbar(im2, cax=cax2)
    fig.colorbar(im4, cax=cax4)
    
    fig.suptitle('[Subject {}] - TF representation'.format(subject_id), fontsize=fontsize+5)
    fig.tight_layout()
    
    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + 'sub{}_{}_{}.png'.format(int(subject_id), sig_name, task_name), facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        
    plt.show()


        
# TODO: migrate to module later
def get_sur_norm(t_sig, sig, norm_param, debug=False):
    if norm_param['mode']=='EMD':
        imf = emd.sift.sift( sig )

        # hard coded, the first one is usually resp. imf (is this the right assumption tho?)
        resp_imf = imf[:,0]

        i_resp_peaks, i_resp_valleys = find_peaks_resp(resp_imf)

        f_peaks = interp1d(t_sig[i_resp_peaks], resp_imf[i_resp_peaks], kind='quadratic',fill_value='extrapolate')
        RR_peaks = f_peaks(t_sig)

        f_valleys = interp1d(t_sig[i_resp_valleys], resp_imf[i_resp_valleys], kind='quadratic',fill_value='extrapolate')
        RR_valleys = f_valleys(t_sig)

        scaling_factor = (RR_peaks - RR_valleys)/2
        sig_norm = resp_imf/scaling_factor

        if debug:
            IP,IF,IA = emd.spectra.frequency_transform( imf, sample_rate, 'nht' )

            freq_edges,freq_bins = emd.spectra.define_hist_bins(0,1,100)
            hht = emd.spectra.hilberthuang( IF, IA, freq_edges )
            
            plt.figure( figsize=(16,8) )
            plt.subplot(211,frameon=False)
            plt.plot(t_sig,x,'k')
            plt.plot(t_sig,imf[:,0]-4,'r')
            plt.plot(t_sig,imf[:,1]-8,'g')
            plt.plot(t_sig,imf[:,2]-12,'b')
            plt.xlim(t_sig[0], t_sig[-1])
            plt.grid(True)
            plt.subplot(2,1,2)
            plt.pcolormesh( t_sig, freq_bins, hht, cmap='ocean_r' )
            plt.ylabel('Frequency (Hz)')
            plt.xlabel('Time (secs)')
            plt.grid(True)
            plt.show()

        
    elif norm_param['mode']=='medfilt':
        if norm_param.get('length_medfilt'): 
            length_medfilt = norm_param['length_medfilt']
            sig_med = medfilt(sig, length_medfilt)
        else:
            sig_med = sig.mean()

        sig_filt = sig - sig_med
        sig_norm = sig_filt/get_envelope(sig_filt)

        if debug:
            plt.figure( figsize=(16,8) )
            plt.subplot(211,frameon=False)
            plt.plot(t_sig,sig,'k')
            plt.plot(t_sig,sig_filt,'r')
            plt.plot(t_sig,sig_norm,'g')
            plt.xlim(t_sig[0], t_sig[-1])
            plt.grid(True)
            plt.xlabel('Time (secs)')
            plt.show()



    # hard coded, but it's generally ok
    threshold = 1.5
    indices = np.where((np.abs(sig_norm)>threshold))[0]
    sig_norm[indices] = np.sign(sig_norm[indices])*threshold

    return sig_norm


# TODO: remove int8, use 0-1 normalization
def float2uint8(arr_float):
    arr_uint8 = ((arr_float - arr_float.min()) / (arr_float.max() - arr_float.min()) * 255. ).astype('uint8')
    return arr_uint8