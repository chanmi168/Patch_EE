import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models

import math
from tqdm import trange
import numpy as np
from sklearn.metrics import r2_score

# from CBPregression.models import *
# from CBPregression.dataset_util import *
# from CBPregression.evaluate import *

from unet_extension.dataset_util import *
from unet_extension.models import *

import sys
sys.path.append('../') # add this line so Data and data are visible in this file
from evaluate import *
from plotting_tools import *
from setting import *
from handy_tools import *
# from unet_extension.training_util import *

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
from matplotlib import colors
# import matplotlib.gridspec as gs
import matplotlib.gridspec as gridspec

def get_mask_RR(freq_arr):
    RR_freq_min = label_range_dict['RR'][0]/60
    RR_freq_max = label_range_dict['RR'][1]/60

    mask_RR = (freq_arr>=RR_freq_min) & (freq_arr<=RR_freq_max)
    return mask_RR

def torch_restore(data, out, label, meta, ts, training_params, subject_id=None):
    # data = torch.Size([N, N_ch, seq_len, N_freq]) 
    # out = torch.Size([N, 2, seq_len, N_freq]) 
    # label = torch.Size([N, 2, seq_len, N_freq]) 
    # meta = torch.Size([N])
    
#     if subject_id is None:
#         indices = np.arange(label.size()[0])
#     else:
#         indices = np.where(meta.numpy()==subject_id)[0]

#     print('hihi', data.size())
    i_shift = label.size()[2]//2

    indices = np.arange(label.size()[0])

    out = out[indices,:,:, :]
    label = label[indices,:,:,:]
#     print(label)

    # get data for a subject, pick the first time step for each sample
#     data_concat = data.detach().cpu().numpy().squeeze()[:,i_shift,:] # N x N_freq
    
    data = data.detach().cpu().numpy() # N x N_ch x seq_len x N_freq
    # binarize data (0=white, 1=black) along the frequency axis
    data_bin = binarize_axis(data, axis=-1) # N x N_ch x seq_len x N_freq
#     data_concat_bin = data_bin[:,:,i_shift,:]  # N x N_ch x N_freq
    data_concat_bin = data_bin[:,:,i_shift-1:i_shift+1,:].mean(axis=-2) # N x N_ch x N_freq
#     data_concat = data[:,:,i_shift,:] # N x N_ch x N_freq
    data_concat = data[:,:,i_shift-1:i_shift+1,:].mean(axis=-2) # N x N_ch x N_freq

    
    # get sigmoid for easier argmax (may not be necessary)
    out = out.detach().cpu().numpy() # N x 2 x seq_len x N_freq
    out = out[:, [1], :, :] # N x 1 x seq_len x N_freq
#     out_concat = out[:, :, i_shift, :] # N x 1 x N_freq
    out_concat = out[:, :, i_shift-1:i_shift+1,:].mean(axis=-2) # N x N_ch x N_freq
    out_bin = binarize_axis(out, axis=-1) # N x 1 x seq_len x N_freq
#     out_concat_bin = out_bin[:, :, i_shift,:] # N x 1 x N_freq
    out_concat_bin = out_bin[:, :, i_shift-1:i_shift+1,:].mean(axis=-2) # N x N_ch x N_freq

    # get corresponding labels
    # axis=1, choose 1 because it's mostly zeros
    label = label.detach().cpu().numpy() # N x 2 x seq_len x N_freq
    label = label[:, [1], :, :] # N x 1 x seq_len x N_freq
#     label_concat = label[:, :, i_shift, :] # N x 1 x N_freq
    label_concat = label[:, :, i_shift-1:i_shift+1,:].mean(axis=-2) # N x 1 x N_freq
    label_bin = binarize_axis(label, axis=-1) # N x N_freq
#     label_concat_bin = label_bin[:, :, i_shift,:] # N x 1 x N_freq
    label_concat_bin = label_bin[:, :, i_shift-1:i_shift+1,:].mean(axis=-2) # N x 1 x N_freq
    
#     ts_concat = ts[:,i_shift]
    ts_concat = ts[:,i_shift-1:i_shift+1].mean(axis=-1) # N x 1 x N_freq

#     label_concat_bin = label.detach().cpu().numpy()[:,1,i_shift,:] # N x N_freq
#     plt.imshow(label.detach().cpu().numpy()[0,2,:,:])
#     print(label.detach().cpu().numpy()[0,2,:,:])
#     sys.exit()

    
    freq_arr = np.asarray(list(training_params['freq_dict'].values()))
    mask_RR = get_mask_RR(freq_arr)


    data_concat = data_concat[:,:,mask_RR]
    data_concat_bin = data_concat_bin[:,:,mask_RR]
    out_concat = out_concat[:,:,mask_RR]
    out_concat_bin = out_concat_bin[:,:,mask_RR]
    label_concat = label_concat[:,:,mask_RR]
    label_concat_bin = label_concat_bin[:,:,mask_RR]

#     print( data_concat.shape, data_concat_bin.shape, out_concat.shape, out_concat_bin.shape, label_concat.shape, label_concat_bin.shape, ts_concat.shape)
#     (1850, 2, 58) (1850, 2, 58) (1850, 1, 58) (1850, 1, 58) (1850, 1, 58) (1850, 1, 58) torch.Size([1850])

#     sys.exit()
    
    
    return data_concat, data_concat_bin, out_concat, out_concat_bin, label_concat, label_concat_bin, ts_concat
    


# def get_RR(label_concat, out_model_concat_bin, training_params):
#     # label_concat = (1784, 20)
#     # out_model_concat_bin = (1784, 20)
#     freq_dict = training_params['freq_dict']

#     RR_label = label_concat.argmax(axis=1)
#     RR_label = np.vectorize(freq_dict.get)(RR_label)

#     RR_model = out_model_concat_bin.argmax(axis=1)
#     RR_model = np.vectorize(freq_dict.get)(RR_model)

#     return RR_label, RR_model

# def get_RR2(data_bin, training_params):
#     # data_bin = (N, N_ch, N_freq)
#     freq_dict = training_params['freq_dict']

#     RR_est = data_bin.argmax(axis=-1)
#     RR_est = np.vectorize(freq_dict.get)(RR_est)

#     return RR_est # (N, N_ch)


def mask_freq_dict(freq_dict):
    freq_dict_masked = freq_dict.copy()
    for key in list(freq_dict_masked):
        value = freq_dict_masked[key]
        if (value<label_range_dict['RR'][0]/60) or (value>label_range_dict['RR'][1]/60):
            freq_dict_masked.pop(key)
    return freq_dict_masked

def get_RR2(data_bin, training_params):
    # data_bin = (N, N_ch, N_freq)
    freq_dict = training_params['freq_dict']
    freq_dict_masked = mask_freq_dict(freq_dict)

    RR_est = data_bin.argmax(axis=-1) + min(list(freq_dict_masked.keys()))
    
#     print(data_bin.shape, RR_est, freq_dict_masked)
    
#     RR_est = np.vectorize(freq_dict.get)(RR_est)
#     freq_dict_masked = mask_freq_dict(freq_dict)
    RR_est = np.vectorize(freq_dict_masked.get)(RR_est)

    return RR_est # (N, N_ch), in bpm

def get_model_out(model, dataloader, training_params):

    device = training_params['device']
    flipper = training_params['flipper']

    model.eval()
    for i, (data, label, meta, ts, _) in enumerate(dataloader):

        data = data.to(device).float()
        label = label.to(device=device, dtype=torch.float)

        if flipper:
            data_flipped = torch.flip(data, [3])
            data = torch.cat((data, data_flipped), 3)
            label_flipped = torch.flip(label, [3])
            label = torch.cat((label, label_flipped), 3)

        # 2. infer by net
#         out, RR_expectation = model(data)
        out = model(data)
#         out = torch.sigmoid(out)       
        out = nn.functional.softmax(out, dim=1)

        
        # clip so it returns to the right dimension
        if flipper:
            N_freq = data.size()[-1]

            data = data[:,:,:,:N_freq//2]
            label = label[:,:,:,:N_freq//2]
            out = out[:,:,:,:N_freq//2]



#         print(np.unique(meta), ts.shape)
#         print(data.size())
#         sys.exit()
        
#         data = data[:,-1,:,:]

        data_concat, data_concat_bin, out_concat, out_concat_bin, label_concat, label_concat_bin, ts_concat = torch_restore(data, out, label, meta, ts, training_params, subject_id=None)

#         RR_label, RR_input = get_RR(label_concat_bin, data_concat_bin, training_params)
#         RR_label, RR_model = get_RR(label_concat_bin, out_model_concat_bin, training_params)        
        
        RR_label = get_RR2(label_concat_bin, training_params) # (N x 1 x N_freq) -> (N x 1)
        RR_model = get_RR2(out_concat_bin, training_params) # (N x 1 x N_freq) -> (N x 1)

        RR_input = get_RR2(data_concat_bin, training_params) # (N x N_ch x N_freq) -> (N x N_ch)
        # added on 11/13 (replace simple averaging with spectral averaging)
        RR_input_spectral = get_RR2(data_concat_bin.mean(axis=1)[:,None,:], training_params) # (N x N_ch x N_freq) -> (N x N_ch)
#         print(data_concat_bin.shape)
#         print(data_concat_bin.mean(axis=1)[:,None,:].shape)
#         sys.exit()
        
#         RR_input_spectral = get_RR2(data_concat_bin.mean(axis), training_params) # (N x N_ch x N_freq) -> (N x N_ch)
        
#         RR_input = np.zeros((data_concat_bin[:,:,0].shape))
#         for i_ch in RR_input.shape[1]:
#             RR_input[:,i_ch] = get_RR2(data_concat_bin[:,i_ch,:], training_params) # (N x N_ch x N_freq) -> (N x N_ch)
        
#         print(RR_label.shape, RR_model.shape, RR_input.shape)
#         sys.exit()
    model_out_dict = {
        'data_concat': data_concat,
        'data_concat_bin': data_concat_bin, 
        'out_concat': out_concat, 
        'out_concat_bin': out_concat_bin, 
        'label_concat': label_concat, 
        'label_concat_bin': label_concat_bin, 
        'RR_label': RR_label, 
        'RR_input': RR_input, 
        'RR_model': RR_model, 
        'RR_input_spectral': RR_input_spectral,
        'meta': meta.numpy(),
        'ts': ts_concat,
#         'RR_expectation': RR_expectation.detach().cpu().numpy(),
    }

    return model_out_dict

def get_performance(model, dataloader, training_params, print_performance=False):

    model_out_dict = get_model_out(model, dataloader, training_params)
    
    RR_label = model_out_dict['RR_label']
#     RR_input = model_out_dict['RR_input'].mean(axis=1, keepdims=True)    
    # added on 11/13 (replace simple averaging with spectral averaging)
    RR_input = model_out_dict['RR_input_spectral']
    RR_model = model_out_dict['RR_model']
    
#     RR_expectation = model_out_dict['RR_expectation']
    
#     print(RR_label.shape, RR_input.shape, RR_input_spectral.shape, RR_model.shape)
#     sys.exit()
#     print('RR_label',RR_label)
#     print('RR_input',RR_input)
#     print('RR_model',RR_model)

    MAE_mean_input, MAE_std_input = get_MAE(RR_label, RR_input)
    MAPE_mean_input, MAPE_std_input = get_MAPE(RR_label, RR_input)

    MAE_mean_model, MAE_std_model = get_MAE(RR_label, RR_model)
    MAPE_mean_model, MAPE_std_model = get_MAPE(RR_label, RR_model)
    
#     MAE_mean_expectation, MAE_std_expectation = get_MAE(RR_label, RR_expectation)
#     MAPE_mean_expectation, MAPE_std_expectation = get_MAPE(RR_label, RR_expectation)
    
    performance_dict = {
        'MAE_mean_input': MAE_mean_input,
        'MAPE_mean_input': MAPE_mean_input,
        'MAE_mean_model': MAE_mean_model,
        'MAPE_mean_model': MAPE_mean_model,
#         'MAE_mean_expectation': MAE_mean_expectation,
#         'MAPE_mean_expectation': MAPE_mean_expectation,

        'MAE_std_input': MAE_std_input,
        'MAPE_std_input': MAPE_std_input,
        'MAE_std_model': MAE_std_model,
        'MAPE_std_model': MAPE_std_model,
#         'MAE_std_expectation': MAE_std_expectation,
#         'MAPE_std_expectation': MAPE_std_expectation,
    }
    
    if print_performance == True:
        print('=====input prediction=====')
        print('MAE: {:.2f} ± {:.2f} bpm'.format(MAE_mean_input, MAE_std_input))
        print('MAPE: {:.2f} ± {:.2f} %'.format(MAPE_mean_input*100, MAPE_std_input*100))

        print('=====model prediction=====')
        print('MAE: {:.2f} ± {:.2f} bpm'.format(MAE_mean_model, MAE_std_model))
        print('MAPE: {:.2f} ± {:.2f} %'.format(MAPE_mean_model*100, MAPE_std_model*100))

#         print('=====model expectation prediction=====')
#         print('MAE: {:.2f} ± {:.2f} bpm'.format(MAE_mean_exoectation, MAE_std_exoectation))
#         print('MAPE: {:.2f} ± {:.2f} %'.format(MAPE_mean_exoectation*100, MAPE_std_exoectation*100))

    return performance_dict, model_out_dict


def plot_loss(total_loss_train, total_loss_val, outputdir=None, show_plot=False):
    fig = plt.figure(figsize=(5,5), dpi=100)

    fontsize = 15

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(total_loss_train, color='steelblue', label='train')
    ax.plot(total_loss_val,color='firebrick', label='val')
    ax.set_xlabel('epochs', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title('2D U-Net loss', fontsize=fontsize*1.3)

    ax.legend(frameon=True)
    

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + 'loss.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

def plot_unet_results(model_out_dict, training_params, title_str=None, outputdir=None, show_plot=False, paper_fig=False):
    meta = model_out_dict['meta']
    freq_dict = training_params['freq_dict']
    input_names = training_params['input_names']
    
    freq = np.array(list(freq_dict.values()))

#     RR_range = label_range_dict['br']
    RR_range = label_range_dict['RR']
    freq_mask = (freq>=RR_range[0]/60) & (freq<=RR_range[1]/60)
    freq = freq[freq_mask]
    
    colornames = list(color_dict.keys())
    
    vmin_input = 0
    vmax_input = 1
    
    vmin_output = 0
    vmax_output = 1
    
    if title_str is None:
        title_str = ''
    else:
        title_str = '({})'.format(title_str)

    for subject_id in np.unique(meta[:,0]).astype(int):

        indices = np.where(meta[:,0]==subject_id)[0]

        if indices.shape[0]==0:
            print('this dataset doesnt have subject {}'.format(subject_id))
            continue

        data_concat = model_out_dict['data_concat'][indices,:,:]/256
        out_concat = model_out_dict['out_concat'][indices,0,:]
        label_concat = model_out_dict['label_concat'][indices,0,:]
        RR_label = model_out_dict['RR_label'][indices, 0]
        RR_input = model_out_dict['RR_input'][indices, :]
        RR_model = model_out_dict['RR_model'][indices, 0]
#         RR_expectation = model_out_dict['RR_expectation'][indices]
        

#         data_concat = data_concat[:,:,freq_mask]
#         out_concat = out_concat[:,freq_mask]
#         label_concat = label_concat[:,freq_mask]      
        
#         print(data_concat.shape, out_concat.shape, label_concat.shape, freq.shape, freq_mask.shape)
#         sys.exit()


        ts = np.arange(RR_label.shape[0]) # 1Hz

#         extent = [ts[0], ts[-1], freq[0]*60, freq[-1]*60]
        extent = [ts[0], ts[-1], RR_range[0], RR_range[-1]]
        
        
        N_modalities = data_concat.shape[1]
        N_axes = 3 + N_modalities
        
        
        gs0 = gridspec.GridSpec(N_axes,2, width_ratios=[0.1, 10])
        fig = plt.figure(figsize=(10,2*N_axes), dpi=100)

        axes = []
        caxes = []
        for i_ch in range(N_axes):
            axes.append( fig.add_subplot(gs0[i_ch,1]) )
            if i_ch == N_axes-1:
                continue
            caxes.append( fig.add_subplot(gs0[i_ch,0]) )

#         ax1 = fig.add_subplot(gs0[0,0])
#         ax2 = fig.add_subplot(gs0[1,0])
#         ax3 = fig.add_subplot(gs0[2,0])
#         ax4 = fig.add_subplot(gs0[3,0])

#         cax2 = fig.add_subplot(gs0[1,1])
#         cax4 = fig.add_subplot(gs0[3,1])          
        
#         fig = plt.figure(figsize=(10,2*N_axes), dpi=100)
        fontsize = 15
        fig.suptitle('subject {}'.format(subject_id)+title_str, fontsize=fontsize*1.5)

        
        for i_ch in range(N_modalities):
            ax = axes[i_ch]
            cax = caxes[i_ch]
#             ax1 = fig.add_subplot(N_axes, 1, i)
            im = ax.imshow(data_concat[:,i_ch].T, label='data', cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin_input, vmax=vmax_input)
#             ax.set_title('input (ch={})'.format(input_names[i_ch]), fontsize=fontsize, y=1.05)
            ax.set_ylabel('input (ch={})'.format(input_names[i_ch]), fontsize=fontsize*0.6, y=1.05, va='center')

            fig.colorbar(im, cax=cax)
#             print('data_concat[:,i_ch]', data_concat[:,i_ch].min(), data_concat[:,i_ch].max())

            
#         print(i_ch)

        i_ch += 1
        ax = axes[i_ch]
        cax = caxes[i_ch]
#         ax2 = fig.add_subplot(N_axes, 1, i)
        im = ax.imshow(out_concat.T, label='output', cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin_output, vmax=vmax_output)
#         ax.set_title('model output (min={:.2f}, max={:.2f})'.format(out_concat.min(), out_concat.max()), fontsize=fontsize, y=1.05)
        ax.set_ylabel('model output (min={:.2f}, max={:.2f})'.format(out_concat.min(), out_concat.max()), fontsize=fontsize*0.5, y=1.05, va='center')
        fig.colorbar(im, cax=cax)
#         print('out_concat', out_concat.min(), out_concat.max())

#         print(i_ch)

        i_ch += 1
        ax = axes[i_ch]
        cax = caxes[i_ch]
#         ax = fig.add_subplot(N_axes, 1, i)
        im = ax.imshow(label_concat.T, label='label', cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin_output, vmax=vmax_output)
#         ax.set_title('label', fontsize=fontsize, y=1.05)
        ax.set_ylabel('label', fontsize=fontsize*0.5, y=1.05, va='center')
        fig.colorbar(im, cax=cax)
#         print('label_concat', label_concat.min(), label_concat.max())
#         print(i_ch)

        
        ## plot RR estimation
        i_ch += 1
#         print(i_ch, len(axes))
        ax = axes[i_ch]
    
        # plot RR label timeseries
        ax.plot(ts, RR_label*60, color='steelblue', linewidth=1, alpha=1 ,zorder=1)
        ax.scatter(ts, RR_label*60, s=10, alpha=1, facecolors='white', edgecolors='steelblue', zorder=3, label='label')
        ax.fill_between(ts, RR_label*60+1, RR_label*60-1, alpha=0.3, color = 'steelblue')

        # plot RR model timeseries
        color = color_dict[colornames[10]]
        ax.plot(ts, RR_model*60, linestyle='--', linewidth=0.7, color=color, alpha=0.9 ,zorder=5)
        ax.scatter(ts, RR_model*60, s=10, alpha=1, facecolors='white', edgecolors=color, zorder=3, label='model est.')
        
        # plot RR expectation timeseries
#         color = color_dict[colornames[11]]
#         ax.plot(ts, RR_expectation, linestyle='--', linewidth=0.5, color=color, alpha=0.4 ,zorder=2)
#         ax.scatter(ts, RR_expectation, s=5, alpha=0.5, facecolors='white', edgecolors=color, zorder=2, label='model exp.')
        
        for i_ch in range(N_modalities):
            ax.plot(ts, RR_input[:, i_ch]*60, linestyle='--', linewidth=0.5, color=color_dict[colornames[14+i_ch]], alpha=0.4 ,zorder=2)
            ax.scatter(ts, RR_input[:, i_ch]*60, s=5, alpha=0.5, facecolors='white', edgecolors=color_dict[colornames[14+i_ch]], zorder=2, label='input est. (ch={})'.format(i_ch))

        ax.set_xlim(ts[0], ts[-1])
        ax.set_ylim(freq[0]*60, freq[-1]*60)
#         ax.set_title('RR estimation', y=1.05)
        ax.set_title('RR estimation', fontsize=fontsize*0.5, va='center')
        ax.legend(loc='upper right', fontsize=fontsize*0.7, frameon=True, bbox_to_anchor=(1.3, 1))
        ax.set_xlabel('time (sec)', fontsize=fontsize*1.2)
        

        
#         axes[0].set_ylabel('RR (bpm)', fontsize=fontsize*1.2)
        
        y_label_list = [5,10,20,30,40,50,60]
        
        for i_ch in range(len(axes)):
            ax = axes[i_ch]
            ax.set_yticks(y_label_list)
            ax.set_yticklabels(y_label_list)
            if i_ch!=len(axes)-1:
                ax.get_xaxis().set_visible(False)

#         ax = axes[N_modalities]
#         ax2.set_yticks(y_label_list)
#         ax2.set_yticklabels(y_label_list)
        
#         ax = axes[N_modalities+1]
#         ax3.set_yticks(y_label_list)
#         ax3.set_yticklabels(y_label_list)
        
#         ax4.set_yticks(y_label_list)
#         ax4.set_yticklabels(y_label_list)
        
        fig.tight_layout()

        if outputdir is not None:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)
            fig.savefig(outputdir + 'model_predictions_sub{}{}.png'.format(subject_id, title_str), facecolor=fig.get_facecolor())
        if show_plot == False:
            plt.close(fig)
            pyplot.close(fig)
            plt.close('all')
            
            
            
        if paper_fig:
            
#             if subject_id!=7 and subject_id!=1:
#                 continue
#             fig = plt.figure(figsize=(10,2*N_axes), dpi=100)

            fontsize = 20
#             fig.suptitle('subject {}'.format(subject_id)+title_str, fontsize=fontsize*1.5)

            outputdir_paper = outputdir+'/sub{}/'.format(subject_id)
            if not os.path.exists(outputdir_paper):
                os.makedirs(outputdir_paper)
            
            for i_ch in range(N_modalities):
                fig = plt.figure(figsize=(4,2), dpi=120)

                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(data_concat[:,i_ch].T, label='data', cmap='viridis', aspect="auto", origin='lower',  extent=extent)
#                 ax.set_title('input (ch={})'.format(i_ch), fontsize=fontsize, y=1.05)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)

                fig.tight_layout()

                fig.savefig(outputdir_paper + 'model_predictions_{}_{}.png'.format(title_str, i_ch), facecolor=fig.get_facecolor())
                plt.close(fig)
                pyplot.close(fig)
                plt.close('all')
                    
            fig = plt.figure(figsize=(4,2), dpi=120)
            ax = fig.add_subplot(1, 1, 1)
            i = i + 1
            ax.imshow(out_concat.T, label='output', cmap='viridis', aspect="auto", origin='lower',  extent=extent)
#             ax.set_title('model output', fontsize=fontsize, y=1.05)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)

            fig.tight_layout()
            fig.savefig(outputdir_paper + 'model_predictions_{}_{}.png'.format(title_str, 'output'), facecolor=fig.get_facecolor())
            plt.close(fig)
            pyplot.close(fig)
            plt.close('all')

            fig = plt.figure(figsize=(4,2), dpi=120)
            ax = fig.add_subplot(1, 1, 1)

            ax.imshow(label_concat.T, label='label', cmap='viridis', aspect="auto", origin='lower',  extent=extent)
#             ax.set_title('label', fontsize=fontsize, y=1.05)
            ax.tick_params(axis='both', which='major', labelsize=fontsize)

            fig.tight_layout()
            fig.savefig(outputdir_paper + 'model_predictions_{}_{}.png'.format(title_str, 'label'), facecolor=fig.get_facecolor())
            plt.close(fig)
            pyplot.close(fig)
            plt.close('all')

def plot_unet_scatter(ax, RR_label, RR_model, meta, i_start=0, color='steelblue' ,title_str='Training RR', show_label=False):
    fontsize = 27
    colornames = list(color_dict.keys())
    markernames = list(marker_dict.keys())
    N_subjects = np.unique(meta[:,0]).shape[0]
#     RR_range = label_range_dict['br']
    RR_range = label_range_dict['RR']


    i = i_start
    for subject_id in np.unique(meta[:,0]):
        indices_sub = np.where(meta[:,0]==subject_id)[0]
        ax.scatter(RR_label[indices_sub]*60, RR_model[indices_sub]*60, label='sub {}'.format(int(subject_id)), color=color,  marker=marker_dict[markernames[i%len(markernames)]], s=80, alpha=0.3)
        i += 1

    ax.plot( RR_range,RR_range , '--', color='gray', alpha=0.8)

    # these are matplotlib.patch.Patch properties
#     props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.5)
    props = dict(boxstyle='round,pad=0.7', facecolor='white', edgecolor='black', alpha=0.5)
    
# t.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='red'))

    # bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))


#     textstr = r'$r^{2}$' + '= {:.2f}'.format(r2_score(RR_label, RR_model)) + '\n' + \
#     'MAE = {:.2f} bpm'.format(get_MAE(RR_label, RR_model)[0]*60)
    textstr = r'$r^{2}$' + '= {:.2f}'.format(r2_score(RR_label, RR_model))
    
    # place a text box in upper left in axes coords
    ax.text(0.45, 0.8, textstr, transform=ax.transAxes, fontsize=fontsize-2,
    verticalalignment='bottom', horizontalalignment='left', bbox=props)
#     ax.legend(loc='upper right', frameon=True, fontsize=fontsize-3)

    if show_label:
        ax.set_title(title_str, fontsize=fontsize*1.3, pad=20)
        ax.set_xlabel('RR reference (bpm)', fontsize=fontsize, labelpad=20)
        ax.set_ylabel('RR estimation (bpm)', fontsize=fontsize, labelpad=20)


    tick_major_arr = (np.arange(RR_range[1]//5)+1)*5
    tick_minor_arr = np.arange(RR_range[1]//1)+1

    
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    
    ax.set_yticks(tick_major_arr, minor=False)
    ax.set_xticks(tick_major_arr, minor=False)

    ax.set_yticks(tick_minor_arr, minor=True)
    ax.set_xticks(tick_minor_arr, minor=True)


    ax.yaxis.grid(True, which='major')
#     ax.yaxis.grid(True, which='minor')
    ax.xaxis.grid(True, which='major')
#     ax.xaxis.grid(True, which='minor')
    
    ax.set_xlim(RR_range)
    ax.set_ylim(RR_range)
    
def plot_unet_scatter_compare(model_out_dict_TEST, outputdir=None, show_plot=False):

    RR_label = model_out_dict_TEST['RR_label'][:,0]
#     RR_input = model_out_dict_TEST['RR_input'][:,0]
    RR_input = model_out_dict_TEST['RR_input'].mean(axis=1)
    RR_model = model_out_dict_TEST['RR_model'][:,0]

    meta = model_out_dict_TEST['meta']

    i_start = 0

    color = 'steelblue'

    title_str = 'Training RR'

    fig = plt.figure(figsize=(14,7), dpi=100)

    ax1 = fig.add_subplot(1, 2, 1)
    plot_unet_scatter(ax1, RR_label, RR_input, meta, i_start=0, color='seagreen' ,title_str='Input RR')


    ax2 = fig.add_subplot(1, 2, 2)
    plot_unet_scatter(ax2, RR_label, RR_model, meta, i_start=0, color='steelblue' ,title_str='Model RR')

    fig.tight_layout()

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + 'model_scatter.png', facecolor=fig.get_facecolor())
    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

def plot_unet_scatter_single(model_out_dict_TEST, outputdir=None, show_plot=False):

    RR_label = model_out_dict_TEST['RR_label'][:,0]
#     RR_input = model_out_dict_TEST['RR_input'][:,0]
    RR_input = model_out_dict_TEST['RR_input'].mean(axis=1)
    RR_model = model_out_dict_TEST['RR_model'][:,0]

    meta = model_out_dict_TEST['meta']

    i_start = 0

    color = 'steelblue'

    title_str = 'Training RR'

    fig = plt.figure(figsize=(7,7), dpi=100)

#     ax1 = fig.add_subplot(1, 2, 1)
#     plot_unet_scatter(ax1, RR_label, RR_input, meta, i_start=0, color='seagreen' ,title_str='Input RR')


    ax2 = fig.add_subplot(1, 1, 1)
    plot_unet_scatter(ax2, RR_label, RR_model, meta, i_start=0, color='steelblue' ,title_str='Model RR')

    fig.tight_layout()

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + 'model_scatter_single.png', facecolor=fig.get_facecolor())
    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

        
def get_data_activation(model, dataloaders, training_params, verbose=False):
    # after model is trained, this function will be called to store all intermediate output to ``activation``

    device = training_params['device']

#     dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    dataloader = dataloaders['val']

    # # 1. set up the hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    def get_activation_tuple(name):
        def hook(model, input, output):
            out, weights = output
            activation[name+'_out'] = out.detach()
            activation[name+'_weights'] = weights.detach()
        return hook

    data = torch.from_numpy(dataloader.dataset.data)
    data = data.to(device).float()

    label = torch.from_numpy(dataloader.dataset.label)
    label = label.to(device).float()

    meta = dataloader.dataset.meta
    ts = dataloader.dataset.ts

    if training_params['flipper']:
        data_flipped = torch.flip(data, [3])
        data = torch.cat((data, data_flipped), 3)
        label_flipped = torch.flip(label, [3])
        label = torch.cat((label, label_flipped), 3)

    variant_name = training_params['variant']
#     model_name = training_params['model_name']
    layer_names = []

    # 3. define the layers that I want to look at
    
#     if model_name == 'UNet':
    if variant_name == 'baseline':
        if variant_name=='SE_block_final':
            SE_name= 'SE'
            model.SE.conv_layers[0].register_forward_hook(get_activation('{}_conv1'.format(SE_name)))
            model.SE.conv_layers[4].register_forward_hook(get_activation('{}_conv2'.format(SE_name)))
            
            model.SE.avg_pool.register_forward_hook(get_activation('{}_avg_pool'.format(SE_name)))
            model.SE.fc[0].register_forward_hook(get_activation('{}_fc1'.format(SE_name)))
            model.SE.fc[3].register_forward_hook(get_activation('{}_fc2'.format(SE_name)))
            model.SE.fc[-1].register_forward_hook(get_activation('{}_weight'.format(SE_name)))


            layer_names =['SE1_avg_pool', 'SE1_fc1', 'SE1_fc2', 'SE1_weight', 'SE1_conv1', 'SE1_conv2', 'Maxpool1']

#         model.Conv0.conv[-1].register_forward_hook(get_activation('Conv0'))
        model.Conv1.conv[-1].register_forward_hook(get_activation('Conv1'))
        model.Conv2.conv[-1].register_forward_hook(get_activation('Conv2'))
        model.Conv3.conv[-1].register_forward_hook(get_activation('Conv3'))
        model.Up_conv3.conv[-1].register_forward_hook(get_activation('Up_conv3'))
        model.Up_conv2.conv[-1].register_forward_hook(get_activation('Up_conv2'))
        model.Conv.register_forward_hook(get_activation('Conv'))
        layer_names = layer_names + ['Conv0', 'Conv1', 'Conv2', 'Conv3', 'Up_conv3', 'Up_conv2']

        if variant_name=='AT_block':
            model.atten_block.register_forward_hook(get_activation_tuple('atten_block'))
            layer_names = layer_names + ['atten_block_out', 'atten_block_weights']

#     elif model_name == 'Late_UNet':
    elif variant_name == 'Late_UNet':
        if variant_name=='SE_block':
            SE_name= 'SE'
            model.SE_block.conv_layers[0].register_forward_hook(get_activation('{}_conv1'.format(SE_name)))
            model.SE_block.conv_layers[4].register_forward_hook(get_activation('{}_conv2'.format(SE_name)))
            
            model.SE_block.avg_pool.register_forward_hook(get_activation('{}_avg_pool'.format(SE_name)))
            model.SE_block.fc[0].register_forward_hook(get_activation('{}_fc1'.format(SE_name)))
            model.SE_block.fc[3].register_forward_hook(get_activation('{}_fc2'.format(SE_name)))
            model.SE_block.fc[-1].register_forward_hook(get_activation('{}_weight'.format(SE_name)))
            

            layer_names = ['SE_conv1', 'SE_conv2', 'SE_avg_pool', 'SE_fc1', 'SE_fc2', 'SE_weight', ]
#             layer_names =['SE0_conv1', 'SE0_conv2', 'SE0_avg_pool', 'SE0_fc1', 'SE0_fc2', 'SE0_weight', 
#                           'SE1_conv1', 'SE1_conv2', 'SE1_avg_pool', 'SE1_fc1', 'SE1_fc2', 'SE1_weight',]
            
        for input_name in training_params['input_names']:
            model.unets[input_name].Conv.register_forward_hook(get_activation('{}_out'.format(input_name)))
            layer_names.append('{}_out'.format(input_name))
            

        for input_name in training_params['input_names']:
            model.unets[input_name].Conv1.conv[-1].register_forward_hook(get_activation(input_name+'_Conv1'))
            model.unets[input_name].Conv2.conv[-1].register_forward_hook(get_activation(input_name+'_Conv2'))
            model.unets[input_name].Conv3.conv[-1].register_forward_hook(get_activation(input_name+'_Conv3'))
            model.unets[input_name].Up_conv3.conv[-1].register_forward_hook(get_activation(input_name+'_Up_conv3'))
            model.unets[input_name].Up_conv2.conv[-1].register_forward_hook(get_activation(input_name+'_Up_conv2'))
            model.unets[input_name].Conv.register_forward_hook(get_activation(input_name+'_Conv'))
            


    
    model.eval()

    # 4. pass the data to the model and the hook will take care of the rest (output stored in activation), don't really need out
#     out, _ = model(data)
    out = model(data)

#     out = torch.sigmoid(out)
    out = nn.functional.softmax(out, dim=1)

    data_concat, data_concat_bin, out_concat, out_concat_bin, label_concat, label_concat_bin, ts_concat = torch_restore(data, out, label, meta, ts, training_params, subject_id=None)
    
#     print(out_concat_bin.shape)
    RR_model = get_RR2(out_concat_bin, training_params) # (N x 1 x N_freq) -> (N x 1)
    RR_label = get_RR2(label_concat_bin, training_params) # (N x 1 x N_freq) -> (N x 1)

#     sys.exit()


    data = data.cpu().detach().numpy()
    out = out.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    b, c, h, w = data.shape

#     print(activation.keys())

    for key in activation:
        
#         if 'atten_block' in key:
#             out, weights = activation[key]
#             out = out.cpu().detach().numpy()
#             weights = weights.cpu().detach().numpy()    
#             print(out, weights)
#         else:
        activation[key] = activation[key].cpu().detach().numpy()
        
        if key=='SE_avg_pool':
            activation[key] = activation[key].squeeze()
        if key=='SE0_avg_pool':
            activation[key] = activation[key].squeeze()
        if key=='SE1_avg_pool':
            activation[key] = activation[key].squeeze()

#     activation['SE_conv1'] = activation['SE_conv1'][:,0,:,:].reshape(b, -1, activation['SE_conv1'].shape[-2], activation['SE_conv1'].shape[-1])
#     activation['SE_conv2'] = activation['SE_conv2'].reshape(b, -1, activation['SE_conv2'].shape[-2], activation['SE_conv2'].shape[-1])



#     activation['SE_conv2'] = activation['SE_conv2'].reshape(b, -1, activation['SE_conv2'].shape[-2], activation['SE_conv2'].shape[-1])

#     activation['SE0_conv1'] = activation['SE0_conv1'][:,0,:,:].reshape(b, c, activation['SE0_conv1'].shape[-2], activation['SE0_conv1'].shape[-1])
#     activation['SE0_conv2'] = activation['SE0_conv2'].reshape(b, c, activation['SE0_conv2'].shape[-2], activation['SE0_conv2'].shape[-1])

#     activation['SE1_conv1'] = activation['SE1_conv1'][:,0,:,:].reshape(b, c, activation['SE1_conv1'].shape[-2], activation['SE1_conv1'].shape[-1])
#     activation['SE1_conv2'] = activation['SE1_conv2'].reshape(b, c, activation['SE1_conv2'].shape[-2], activation['SE1_conv2'].shape[-1])

    if verbose:
        for key in activation:
            print(key, activation[key].shape)
            
    RR_model = RR_model * 60
    RR_label = RR_label * 60
    
    return data, out, label, RR_model, RR_label, activation, meta



def check_filters(model, training_params, input_choice=None, fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
    # after model is trained, this function will plot the conv filters

    if input_choice is None:
        model_sub = model
    else:
        model_sub = model.unets[input_choice]
    
    model_params = {}

    # in_ch, out_ch, kernel, kernel -> in_ch * out_ch, kernel, kernel (reshape(-1,3,3))
    model_params['Conv1_0'] = model_sub.Conv1.conv[0].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)
    model_params['Conv1_1'] = model_sub.Conv1.conv[3].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)

    model_params['Conv2_0'] = model_sub.Conv2.conv[0].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)
    model_params['Conv2_1'] = model_sub.Conv2.conv[3].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)

    model_params['Conv3_0'] = model_sub.Conv3.conv[0].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)
    model_params['Conv3_1'] = model_sub.Conv3.conv[3].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)

    model_params['Up_conv3_0'] = model_sub.Up_conv3.conv[0].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)
    model_params['Up_conv3_1'] = model_sub.Up_conv3.conv[3].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)

    model_params['Up_conv2_0'] = model_sub.Up_conv2.conv[0].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)
    model_params['Up_conv2_1'] = model_sub.Up_conv2.conv[3].weight.data.cpu().detach().numpy().reshape(-1,1,3,3)

    fig = plt.figure(figsize=(20,5), constrained_layout=False, dpi=120)
    N_layers = len(model_params.keys())

    gs = gridspec.GridSpec(1, N_layers)

    for i_layer, layer_name in enumerate(model_params.keys()):
        ax = fig.add_subplot(gs[0,i_layer])
        nrow = math.ceil(model_params[layer_name].shape[0]**0.5)
        grid_img = torchvision.utils.make_grid(torch.from_numpy(model_params[layer_name]), normalize=True, padding=1, pad_value=0.95, nrow=nrow)
        ax.imshow(grid_img.permute(1, 2, 0))
        ax.set_title(layer_name)

#     fig.tight_layout()
    fig.subplots_adjust(wspace=0.2, hspace=0.8)

    
    if fig_name is None:
        fig_name = 'filters'

    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

    plt.show()
    
    
def check_weights_TBD(model, dataloaders, training_params, fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
    data, out, label, RR_model, RR_label, activation, meta = get_data_activation(model, dataloaders, training_params, verbose=False)

    weights_sample = activation['atten_block_weights']
#     weight_string = ''
#     for input_name, weight in zip(training_params['input_names'], weights_sample):
#         weight_string = weight_string + 'P_{}={:.2f}; '.format(input_name, weight)
    fig, ax = plt.subplots(figsize=(10,5), dpi=70)
    ax.plot(weights_sample)
    
    fig.tight_layout()

    if fig_name is None:
        fig_name = 'weights'

#     fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample) + '\nRRout: {:.3f} RRlabel: {:.3f} AE: {:.3f} [BPM]'.format(
#         RR_model[i_sample].squeeze(), RR_label[i_sample].squeeze(), error_abs[i_sample]), fontsize=15)
    
    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

    plt.show()
    
    
    
def check_weights(model, dataloaders, training_params, fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
    data, out, label, RR_model, RR_label, activation, meta = get_data_activation(model, dataloaders, training_params, verbose=False)

    weights = activation['atten_block_weights']
#     fig, ax = plt.subplots(figsize=(10,5), dpi=70)
#     ax.plot(weights_sample)

    gs0 = gridspec.GridSpec(2,2, width_ratios=[0.1, 10])
    fig = plt.figure(figsize=(weights.shape[0]/3,3), dpi=100)

    ax = fig.add_subplot(gs0[0,1])
    cax = fig.add_subplot(gs0[0,0])

    vmin = weights.min()
    vmax = weights.max()

    im = ax.imshow(weights.T , cmap='viridis', aspect="auto", origin='lower', vmin=vmin, vmax=vmax, interpolation='none',)

    ax.set_xticks(np.arange(0, weights.shape[0], 1))
    ax.set_yticks(np.arange(0, weights.shape[1], 1))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, weights.shape[0]+1, 1))
    ax.set_yticklabels(np.arange(1, weights.shape[1]+1, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, weights.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, weights.shape[1], 1), minor=True)

    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    fig.colorbar(im, cax=cax)
    
    ax_task = fig.add_subplot(gs0[1,1])
    t_arr = np.arange(1, weights.shape[0]+1, 1)
#     print(t_arr)
    ax_task.plot(t_arr, meta[:,1])
    ax_task.scatter(t_arr, meta[:,1])
    
    ax_task.grid(which='minor', color='w', linestyle='-', linewidth=0.5)

    ax_task.set_xticks(np.arange(1, weights.shape[0]+1, 1))
#     ax_task.set_xticklabels(np.arange(1, weights.shape[0]+1, 1))
#     ax_task.set_xticks(np.arange(-.5, weights.shape[0], 1), minor=True)
    ax_task.set_xlim(0.5, weights.shape[0]+0.5)
    fig.tight_layout()

    if fig_name is None:
        fig_name = 'weights'

#     fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample) + '\nRRout: {:.3f} RRlabel: {:.3f} AE: {:.3f} [BPM]'.format(
#         RR_model[i_sample].squeeze(), RR_label[i_sample].squeeze(), error_abs[i_sample]), fontsize=15)
    
    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

    plt.show()

def check_featuremap(model, dataloaders, training_params, mode='random', input_choice=None, fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
    # after model is trained, this function will plot the feature maps (all intermediate output)
    
    data, out, label, RR_model, RR_label, activation, meta = get_data_activation(model, dataloaders, training_params, verbose=False)
    
    b, c, h, w = data.shape    
    error_abs = np.abs(RR_model - RR_label).squeeze()

    if mode=='worst':
        i_sample = np.argmax(error_abs)
    if mode=='best':
        i_sample = np.argmin(error_abs)
    if mode=='random':
        # check one sample only
        i_sample = np.random.randint(b)
        
        


    fig = plt.figure(figsize=(15,30), constrained_layout=False, dpi=70)
    gs = gridspec.GridSpec(16, 9)
    fontsize = 8

    ts = np.arange(h)*3 # 1Hz
    RR_range = label_range_dict['RR']

    extent = [ts[0], ts[-1], RR_range[0], RR_range[-1]]

    # 1. plot the input
    vmin = data[i_sample,:,:,:].min()
    vmax = data[i_sample,:,:,:].max()
    for i_input, input_name in enumerate(training_params['input_names']):
        if input_choice is not None:
            if input_choice!=input_name:
                continue

        ax = fig.add_subplot(gs[i_input, 0])
        ax.imshow(data[i_sample,i_input,:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(input_name, fontsize=fontsize)

    #     'Conv1', 'Conv2', 'Conv3', 'Up_conv3', 'Up_conv2'
    # 2. plot the 1st average pooling output in SE block
#     print(activation['atten_block_out'].shape)
#     print(activation['atten_block_weights'].shape)

    variant_name = training_params['variant']
    if variant_name=='AT_block':
        weights_sample = activation['atten_block_weights'][i_sample,:]
        weight_string = ''
        for input_name, weight in zip(training_params['input_names'], weights_sample):
            weight_string = weight_string + 'P_{}={:.2f}; '.format(input_name, weight)
    else:
        weight_string = ''

    for i_layer, layer_name in enumerate(['Conv1', 'Conv2', 'Conv3', 'Up_conv3', 'Up_conv2', 'Conv']):
#         if training_params['model_name'] == 'UNet':
        if training_params['variant'] == 'baseline':
            key = layer_name
#         elif training_params['model_name'] == 'Late_UNet':
        elif training_params['variant'] == 'Late_UNet':
            key = input_choice + '_' + layer_name

        vmin = activation[key][i_sample,:,:,:].min()
        vmax = activation[key][i_sample,:,:,:].max()
        for i_ch in range(activation[key].shape[1]):
            ax = fig.add_subplot(gs[i_ch, i_layer+1])
            ax.imshow(activation[key][i_sample,i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
            ax.set_title('i_ch: {}\n{}'.format(i_ch, key), fontsize=fontsize)


    # 6. plot output
    vmin = out[i_sample,:,:,:].min()
    vmax = out[i_sample,:,:,:].max()
    #     print(vmin, vmax)
    for i_out in range(out.shape[1]):
        ax = fig.add_subplot(gs[i_out, -2])
        ax.imshow(out[i_sample, i_out,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title('out ch: {}'.format(i_out), fontsize=fontsize)

    vmin = label[i_sample,:2,:,:].min()
    vmax = label[i_sample,:2,:,:].max()
    for i_label in range(label.shape[1]-1):
        ax = fig.add_subplot(gs[i_label, -1])
        ax.imshow(label[i_sample, i_label,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title('label ch: {}'.format(i_label), fontsize=fontsize)

    fig.subplots_adjust(wspace=0.2, hspace=0.8)

    if fig_name is None:
        fig_name = 'featuremap'

    fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample) + '\nRRout: {:.3f} RRlabel: {:.3f} AE: {:.3f} [BPM]\n{}'.format(
        RR_model[i_sample].squeeze(), RR_label[i_sample].squeeze(), error_abs[i_sample], weight_string), fontsize=15)
    
    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

    plt.show()
    
    
    
def check_attention(model, dataloaders, training_params, mode='random', fig_name=None, show_plot=False, outputdir=None, log_wandb=False, verbose=False):
    # after model is trained, this function will plot the SE block attention

    SE_name = 'SE'
    
    data, out, label, RR_model, RR_label, activation, meta = get_data_activation(model, dataloaders, training_params, verbose=False)
    b, c, h, w = data.shape    
    error_abs = np.abs(RR_model - RR_label).squeeze()


    if mode=='worst':
        i_sample = np.argmax(error_abs)
    if mode=='best':
        i_sample = np.argmin(error_abs)
    if mode=='random':
        # check one sample only
    #         np.random.seed(0)
        i_sample = np.random.randint(b)

    # print(error_abs)
    # print(error_abs[i_sample])
    # print(error_abs.min(), error_abs.max())



    fig = plt.figure(figsize=(25,12), constrained_layout=False, dpi=80)
    gs = gridspec.GridSpec(4, 10)

    # gs.update(wspace = 0.5, hspace = 0.8)
    # i_sample = 50


    ts = np.arange(h)*3 # 1Hz
    RR_range = label_range_dict['RR']

    extent = [ts[0], ts[-1], RR_range[0], RR_range[-1]]

    # 1. plot the input
    vmin = data[i_sample,:,:,:].min()
    vmax = data[i_sample,:,:,:].max()
    for i_input, input_name in enumerate(training_params['input_names']):
        ax = fig.add_subplot(gs[i_input, 0])
        ax.imshow(data[i_sample,i_input,:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(input_name+'\nvmin={:.2e}\nvmax={:.2e}'.format(vmin, vmax))

#     if training_params['model_name']=='UNet':
    if training_params['variant']=='baseline':
        pass

#         for in_ch, input_name in enumerate(training_params['input_names']):
#         #     input_out = activation['{}_out'.format(input_name)]
#             input_out = activation['Conv0'][:, in_ch, :, :]

#             vmin = input_out[i_sample,:,:].min()
#             vmax = input_out[i_sample,:,:].max()

#             ax = fig.add_subplot(gs[ in_ch, 1 ])
#             ax.imshow(input_out[i_sample,:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#             ax.set_title('Conv0 {}\nvmin={:.2e}\nvmax={:.2e}'.format(input_name, input_out[i_sample,:,:].min(), input_out[i_sample,:,:].max()))

#     elif training_params['model_name']=='Late_UNet':
    elif training_params['variant']=='Late_UNet':


        for i_input, input_name in enumerate(training_params['input_names']):
#             ax = fig.add_subplot( gs[ i_input, 1 ] )

            act_data = activation['{}_out'.format(input_name)][i_sample,:,:,:].squeeze()
#             act_data = sigmoid(act_data)
    #     for i_input, input_name in enumerate(training_params['input_names']):
            act_data = scipy.special.softmax(act_data, axis=0)

            vmin = act_data.min()
            vmax = act_data.max()

#             ax.imshow(act_data[:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
#             ax.set_title(input_name+'out\nvmin={:.2e}\nvmax={:.2e}'.format(act_data[:,:].min(), act_data[:,:].max()))


            for i_ch in range(act_data.shape[0]):
                ax = fig.add_subplot( gs[ i_input*2 + i_ch, 1 ] )
                ax.imshow(act_data[i_ch,:,:].T,  cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
                ax.set_title(input_name+'out\nvmin={:.2e}\nvmax={:.2e}'.format(act_data[i_ch,:,:].min(), act_data[i_ch,:,:].max()))

#             plt.show()
#             sys.exit()

    # # 2. plot the 1st average pooling output in SE block
    # act_data = activation['{}_conv1'.format(SE_name)][i_sample,:,:,:]
    # vmin = act_data.min()
    # vmax = act_data.max()
    # # for i_input, input_name in enumerate(training_params['input_names']):
    # for i_ch in range(act_data.shape[0]):
    #     ax = fig.add_subplot(gs[i_ch, 2])
    #     ax.imshow(act_data[i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
    #     ax.set_title('i_ch:{} {}_conv1\nvmin={:.2e}\nvmax={:.2e}'.format(i_ch, SE_name, act_data[i_ch,:,:].min(), act_data[i_ch,:,:].max()))

    # # 2. plot the 1st average pooling output in SE block
    # act_data = activation['{}_conv2'.format(SE_name)][i_sample,:,:,:]
    # vmin = act_data[:,:,:].min()
    # vmax = act_data[:,:,:].max()
    # # for i_input, input_name in enumerate(training_params['input_names']):
    # for i_ch in range(act_data.shape[0]):
    #     ax = fig.add_subplot(gs[i_ch, 3])
    #     ax.imshow(act_data[i_ch,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
    #     ax.set_title('i_ch:{} {}_conv2\nvmin={:.2e}\nvmax={:.2e}'.format(i_ch, SE_name, act_data[i_ch,:,:].min(), act_data[i_ch,:,:].max()))

    
    
    
    if training_params['variant']=='SE_block_final':
        # 5. plot the weights for each input
        act_data = activation['{}_avg_pool'.format(SE_name)].squeeze()[i_sample,:]
        ax = fig.add_subplot(gs[:, 4])
        # ax.barh(training_params['input_names'][::-1], act_data[::-1], height=0.1)
        ax.barh(np.arange(act_data.shape[0]), act_data[::-1], height=0.1)
        ax.set_title('{}_avg_pool'.format(SE_name))
        ax_no_top_right(ax)

        #     5. plot the weights for each input
        act_data = activation['{}_fc1'.format(SE_name)][i_sample,:]
        ax = fig.add_subplot(gs[:, 5])
        #     ax.barh(training_params['input_names'][::-1], activation['SE1_fc1'][i_sample,:][::-1], height=0.1)
        ax.barh(np.arange(act_data.shape[0]), act_data[::-1], height=0.1)
        ax.set_title('{}_fc1'.format(SE_name))
        ax_no_top_right(ax)

        # 5. plot the weights for each input
        act_data = activation['{}_fc2'.format(SE_name)][i_sample,:]
        ax = fig.add_subplot(gs[:, 6])
        ax.barh(np.arange(act_data.shape[0]), act_data[::-1], height=0.1)
        ax.set_title('{}_fc2'.format(SE_name))
        ax_no_top_right(ax)

        # 6. plot the weights for each input
        act_data = activation['{}_weight'.format(SE_name)][i_sample,:]
        ax = fig.add_subplot(gs[:, 7])
        ax.barh(np.arange(act_data.shape[0]), act_data[::-1], height=0.1)
        ax.set_title('{}_weight'.format(SE_name))
        ax_no_top_right(ax)

    # 6. plot output
    vmin = out[i_sample,:,:,:].min()
    vmax = out[i_sample,:,:,:].max()
    #     print(vmin, vmax)
    for i_out in range(out.shape[1]):
        ax = fig.add_subplot(gs[i_out, -2])
        ax.imshow(out[i_sample, i_out,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title('out ch: {}\nvmin={:.2e}\nvmax={:.2e}'.format(i_out, vmin, vmax))

    vmin = label[i_sample,:2,:,:].min()
    vmax = label[i_sample,:2,:,:].max()
    for i_label in range(label.shape[1]-1):
        ax = fig.add_subplot(gs[i_label, -1])
        ax.imshow(label[i_sample, i_label,:,:].T, cmap='viridis', aspect="auto", origin='lower',  extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title('label ch: {}\nvmin={:.2e}, vmax={:.2e}'.format(i_label, vmin, vmax))


    
    fig.subplots_adjust(wspace=0.2, hspace=0.8)

    if fig_name is None:
        fig_name = 'IO_weights'

    fig.suptitle(fig_name+'\ni_sample:{}'.format(i_sample) + '\nRRout: {:.3f} RRlabel: {:.3f} AE: {:.3f} [BPM]'.format(
        RR_model[i_sample].squeeze(), RR_label[i_sample].squeeze(), error_abs[i_sample]), fontsize=15)
    
    if log_wandb:
        wandb.log({fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')

    plt.show()