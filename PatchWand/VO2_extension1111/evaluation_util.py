import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from tqdm import trange
import numpy as np
from sklearn.metrics import r2_score
from scipy.interpolate import griddata


import sys
sys.path.append('../') # add this line so Data and data are visible in this file
from evaluate import *
from plotting_tools import *
from setting import *
from spectral_module import *
# from dataset_util import *
from VO2_extension1111.dataset_util import *
from VO2_extension1111.training_util import *

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
from matplotlib import colors
import matplotlib.gridspec as gs
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from matplotlib import cm # colour map



def plot_losses(CV_dict, show_plot=False, outputdir=None, fig_name=None):
    
    df_losses_train = CV_dict['df_losses_train']
    df_losses_val = CV_dict['df_losses_val']
#     total_loss_train = np.sqrt(CV_dict['total_loss_train'])
#     total_loss_val = np.sqrt(CV_dict['total_loss_val'])
    
    metric_names = list(df_losses_train.columns)
    fig, axes = plt.subplots(len(metric_names),1, figsize=(5, len(metric_names)*3), dpi=80)
    
    fontsize = 10

    for ax, metric_name in zip(axes, metric_names):
        ax.plot(df_losses_train[metric_name].values, 'r', label='train')
        ax.plot(df_losses_val[metric_name].values,'b', label='val')
        ax.legend(loc='upper right', frameon=True, fontsize=fontsize*0.8)

        ax.set_xlabel('epoch', fontsize=fontsize)
        ax.set_ylabel(metric_name, fontsize=fontsize)
        ax_no_top_right(ax)

    fig.tight_layout()

    subject_id = CV_dict['subject_id_val']
    
    if fig_name is None:
        fig_name = 'loss_CV{}'.format(subject_id)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')


def check_featuremap(model, training_params, dataloaders, mode='worst', plot_STFT=True, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):
    
    # print('check featuremap...')
    inputdir = training_params['inputdir']
    device = training_params['device']
    # dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    dataloader = dataloaders['val']
    regression_names = training_params['regression_names']
    channel_n = training_params['channel_n']

    # 1. set up the hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    ecg = torch.from_numpy(dataloader.dataset.ecg)
    scg = torch.from_numpy(dataloader.dataset.scg)
    ppg = torch.from_numpy(dataloader.dataset.ppg)
    
    feature = torch.from_numpy(dataloader.dataset.feature)
    ecg = ecg.to(device).float()
    scg = scg.to(device).float()
    ppg = ppg.to(device).float()
    feature = feature.to(device).float()
    label = dataloader.dataset.label
    meta = dataloader.dataset.meta
    #     model_hooking(model, training_params)

    model_name = training_params['model_name']
    
    layer_names = []

    # 3. define the layers that I want to look at
    if model_name=='FeatureExtractor_CNN2':
        key = list(model.feature_extractors.keys())[0]
        model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
        model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
        model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
        model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
        model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
        model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))

        layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc1', 'fc2']

    elif model_name=='FeatureExtractor_CNN':
        key = list(model.feature_extractors.keys())[0]
        model.feature_extractors[key].layer1.register_forward_hook(get_activation('layer1'))
        model.feature_extractors[key].layer2.register_forward_hook(get_activation('layer2'))
        model.feature_extractors[key].layer3.register_forward_hook(get_activation('layer3'))
        model.feature_extractors[key].layer4.register_forward_hook(get_activation('layer4'))
        model.regressors.EE_cosmed.fc1.register_forward_hook(get_activation('fc1'))
        model.regressors.EE_cosmed.fc2.register_forward_hook(get_activation('fc2'))
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'fc1', 'fc2']

    elif model_name=='CNNlight':
#         key = list(model.feature_extractors.keys())[0]        
        # for input_name in list(model.feature_extractors.keys()):
        # for input_name in training_params['input_names']:
        for input_name in list(model.feature_extractors.keys()):
            model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'-layer_last'))
            layer_names = layer_names + [input_name+'-layer_last']

        for input_name in list(model.feature_extractors.keys()):
            model.feature_extractors[input_name].basicblock_list[1].ch_pooling.register_forward_hook(get_activation(input_name+'-layer_middle'))
            layer_names = layer_names + [input_name+'-layer_middle']

        for input_name in list(model.feature_extractors.keys()):
            model.feature_extractors[input_name].basicblock_list[0].ch_pooling.register_forward_hook(get_activation(input_name+'-layer_first'))
            layer_names = layer_names + [input_name+'-layer_first']
            


        # training_params['regression_names'][-1] is the main task
        # model.regressors[training_params['regression_names'][-1]].get_attention.register_forward_hook(get_activation('attention'))
        # layer_names = layer_names + ['attention']


    # print(layer_names)
    # sys.exit()
    model.eval()

    # 4. pass the data to the model and the hook will take care of the rest (output stored in activation), don't really need out
    # out, deep_feature, concat_feature = model(ecg, scg, ppg, feature)
    out, deep_feature, concat_feature, attention_dict = model(ecg, scg, ppg, feature)

    ecg = ecg.cpu().detach().numpy()
    scg = scg.cpu().detach().numpy()
    ppg = ppg.cpu().detach().numpy()

    label_dict = {}
    out_dict = {}
    

    main_task = training_params['output_names'][0]
    auxillary_tasks = training_params['auxillary_tasks']
    
    
    for output_name in training_params['regression_names']:
    # for output_name in training_params['criterion'].criterions.keys():

        if main_task in output_name:
            continue
        if 'domain' in output_name:
            continue

        # if training_params['dominantFreq_detect']=='expectation':
        #     out_freq = out[output_name].data.detach().cpu().numpy()
        # elif training_params['dominantFreq_detect']=='argmax':
        #     out_freq, indices_dominant = get_HR(deep_feature[output_name.split('-')[1]].data.detach().cpu().numpy(), training_params['xf_masked'])

        out_freq = out[output_name].data.detach().cpu().numpy()
        out_dict[output_name] = out_freq
        
        output_names = list(training_params['criterion'].criterions.keys())
        label_dict[output_name] = label[:, regression_names.index(output_name) ]

    # sys.exit()

    # 5. organize these activation layers

    for layer_name in activation.keys():
        activation[layer_name] = activation[layer_name].cpu().detach().numpy() # dim = (N_batch, channel_n, output_dim)

    for layer_name in activation.keys():
        if main_task in layer_name:
            continue 

        output_name = layer_name.split('-layer')[0]
        # if 'merged' in 
        if 'merged' not in layer_name:
            output_name = 'merged-'+output_name.split('-')[1]

        if output_name not in training_params['regression_names']:
            continue

        output = out_dict[output_name].squeeze()
        label = label_dict[output_name]

        error_abs = np.abs(output - label)

        if model_name == 'ResNet1D_LSTM':
            error_abs = error_abs.mean(axis=-1)

        if mode=='worst':
            i_sample = np.argmax(error_abs)
        if mode=='best':
            i_sample = np.argmin(error_abs)
        if mode=='random':
            # 2. check one sample only
            N_samples = error_abs.shape[0]
            np.random.seed(0)
            i_sample = np.random.randint(N_samples)

        [subject_id, task_id] = meta[i_sample, :2]


        if 'HR' in layer_name:
            xf_dict = training_params['HR_xf_dict']
            # print('using HR_xf_dict')
        elif 'RR' in layer_name:
            xf_dict = training_params['RR_xf_dict']
            # print('using RR_xf_dict')
        else:
            print('No HR_xf_dict or RR_xf_dict')

        xf_masked = xf_dict['xf_masked']
        mask = xf_dict['mask']
        raw_mask = xf_dict['raw_mask']
        FS_Extracted =  xf_dict['FS_Extracted']

        # prepare for FFT
        T = 1/FS_Extracted
        N =  activation[layer_name].shape[-1]

        data_layer = activation[layer_name]
        # print('\t working on', layer_name, data_layer.shape)

        if len(data_layer.shape)<3:
            data_layer = data_layer[:,:,None]


        # sig_name = training_params['input_names'][i_sig]   
        if '-' in layer_name:
            sig_name = layer_name.split('-')[0]
        else:
            sig_name = layer_name

        color = input_color_dict[sig_name]

        if 'ECG' in sig_name:
            unit = unit_dict['ecg']
            input_name = sig_name.split('_layer')[0]
            # i_sig = training_params['input_names'].index(input_name)
            data = ecg
            i_filter_start = 0

#             i_sig = training_params['input_names'].index('ECG')
        elif ('accel' in sig_name) or ('scg' in sig_name) or ('SCG' in sig_name):
            unit = unit_dict['accel']
            input_name = sig_name.split('_layer')[0]
            # i_sig = training_params['input_names'].index(input_name)
            data = scg
            i_filter_start = channel_n

#             i_sig = training_params['input_names'].index('accelZ')
        elif ('ppg' in sig_name) or ('PPG' in sig_name):
            unit = unit_dict['ppg']
            input_name = sig_name.split('_layer')[0]
            # i_sig = training_params['input_names'].index(input_name)
            data = ppg
            i_filter_start = 0

            if data.shape[1]>4:
                if 'HR' in layer_name:
                    data = data[:,4:,:]
                else:
                    data = data[:,:4,:]


        # print('\t\t input_name is', input_name)
        # N_sigs = len(training_params['input_names'])
        N_sigs = data.shape[1]


        N_axes = data_layer.shape[1]+N_sigs
        gs0 = gs.GridSpec(N_axes,3, width_ratios=[16, 2, 2])
        fig = plt.figure(figsize=(25,2*N_axes), dpi=120)
        axes = []
        faxes = []
        daxes = []
        
        for i_ch in range(N_axes):
            axes.append( fig.add_subplot(gs0[i_ch,0]) )
            faxes.append( fig.add_subplot(gs0[i_ch,1]) )
            daxes.append( fig.add_subplot(gs0[i_ch,2]) )

        if plot_STFT:
            fig_STFT, axes_STFT = plt.subplots(N_axes,1, figsize=(16,4*N_axes), dpi=300, facecolor='white', subplot_kw={'projection': '3d'})


        fontsize = 13

        FS_RESAMPLE_DL = training_params['FS_RESAMPLE_DL']
        t_arr = np.arange(data.shape[-1])/FS_RESAMPLE_DL


        # 1. plot one physio sig at the top row


        for i_sig in range(N_sigs):
            ax = axes[i_sig]
            fax = faxes[i_sig]
            # dax = daxes[i_sig]

            # plot raw time domain signals
            ax.plot(t_arr, data[i_sample,i_sig,:], color=color)
            ax.set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot
            ax.set_ylabel('{}\n[{}]'.format(sig_name, unit), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=20)

            ax.set_title('[{}] {}, i_sample={}\nout={:.2f}, label={:.2f}, AE={:.2f} [{}]'.format(subject_id, tasks_dict_reversed[task_id], i_sample, output[i_sample], label[i_sample], error_abs[i_sample], unit_dict[main_task.split('_')[0]]), fontsize=fontsize)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticklabels([])

            # plot raw FT domain            
            xf, yf_scaled = get_psd(data[i_sample,i_sig,:], FS_RESAMPLE_DL)
            
            some_mask = (xf*60 >= xf_masked.min()) & (xf*60 <= xf_masked.max()*2)
            fax.plot(xf[some_mask] * 60, yf_scaled[some_mask].T, color=color)

            # fax.plot(xf[raw_mask] * 60, yf_scaled[raw_mask].T, color=color)
            # fax.plot(xf, yf_scaled.T, color=color, zorder=2)
            
            if ('ppg' in sig_name) or ('PPG' in sig_name):
                fax.axvline(label[i_sample], color='black', linestyle='--', alpha=0.5)
            ax_no_top_right(fax)
            # fax.set_xticklabels([])

            # plot raw STFT
            if plot_STFT:
                ax_STFT = axes_STFT[i_sig]
                ax_STFT.pbaspect = [1, 4, 1]

                xf, y_matrix_scaled =  get_psd(data[:,i_sig,:], FS_RESAMPLE_DL)

                t_cat_arr = np.arange(label.shape[0]) * (t_arr.max()-t_arr.min()) / (label.shape[0]-1) + t_arr.min()

                # dim: (N_instance, N_spectral) -> (N_instance, N_spectral_masked)
                # ax_STFT.plot_surface(t_cat_arr[:, None], xf[raw_mask][None, :]*60, y_matrix_scaled[:, raw_mask], cmap=cm.coolwarm)
                # ax_STFT.set_yticks(xf[raw_mask][::10]*60)

                # f_size = xf.shape[1]
                # label_range_dict[]
                # print(xf_masked)
                some_mask = (xf*60 >= xf_masked.min()) & (xf*60 <= xf_masked.max()*2)
                ax_STFT.plot_surface(t_cat_arr[:, None], xf[None, some_mask]*60, y_matrix_scaled[:, some_mask], cmap=cm.Spectral)

#                 X = t_cat_arr
#                 Y = xf[some_mask]*60
#                 Z = y_matrix_scaled[:, some_mask]
                
#                 # print(X.min(),X.max(),len(Z)/3, Y.min(),Y.max(),len(Z)/3)

#                 xi = np.linspace(X.min(),X.max(),(len(Z)//3))
#                 yi = np.linspace(Y.min(),Y.max(),(len(Z)//3))
                
#                 print(X.shape, Y.shape, Z.shape, xi.shape, yi.shape)
#                 zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='nearest')
#                 xig, yig = np.meshgrid(xi, yi)
#                 ax_STFT.plot_surface(xig, yig, zi, cmap='gist_earth')


                
                
                
                
                # ax_STFT.set_ylim(1,xf[raw_mask])

                # ax_STFT.set_yticks(xf[::10]*60)
                # ax_STFT.set_xticks(t_arr[::300])
                # ax_STFT.set_xticks(t_arr[::300])
                
                ax_STFT.set_title('raw signal', fontsize=fontsize)

                ax_STFT.set_xlabel('t (s)')
                ax_STFT.set_ylabel('freq (bpm)')
                ax_STFT.plot(t_cat_arr, output, color='white', linestyle='--', alpha=0.9, linewidth=0.8)

        # 2. next, plot the feature map 
        # t_arr = np.arange(data_layer.shape[-1]) / FS_Extracted
        
        
        downsample_factor = round(data.shape[-1] / data_layer.shape[-1])
        FS_layer = FS_RESAMPLE_DL / downsample_factor
        t_arr = np.arange(data_layer.shape[-1]) / FS_layer

        for j_filter in range(data_layer.shape[1]):
            ax = axes[j_filter+N_sigs]
            fax = faxes[j_filter+N_sigs]
            dax = daxes[j_filter+N_sigs]
            
            # if j_filter!=data_layer.shape[1]-1:
            #     ax.set_xticklabels([])
            #     fax.set_xticklabels([])

            ax.plot(t_arr, data_layer[i_sample, j_filter, :].squeeze(), alpha=0.9, color=color)
            ax.set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot

            # remove some borders (top and right)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            ax.set_ylabel('filter {}'.format(j_filter), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=20)
            # set tick font size
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)


            # xf, yf_scaled = get_psd(data_layer[i_sample, j_filter, :].squeeze(), FS_Extracted)           
            xf, yf_scaled = get_psd(data_layer[i_sample, j_filter, :].squeeze(), FS_layer)           

            some_mask = (xf*60 >= xf_masked.min()) & (xf*60 <= xf_masked.max()*2)
            fax.plot(xf[some_mask] * 60, yf_scaled[some_mask], alpha=0.9, color=color, zorder=2)
            
            # fax.plot(xf[mask] * 60, yf_scaled[mask], alpha=0.9, color=color, zorder=2)
            # fax.set_xlim(0,200)
            ax_no_top_right(fax)
            
            # print(xf_masked)
            if 'HR' in layer_name:
                dax.plot(xf_masked, attention_dict['features_fft_HR_raw'][i_sample,i_filter_start+j_filter,:]*attention_dict['spectral_attention_HR'][i_sample,0,:], alpha=0.9, color=color, zorder=2)
            elif 'RR' in layer_name:
                dax.plot(xf_masked, attention_dict['features_fft_RR_raw'][i_sample,i_filter_start+j_filter,:]*attention_dict['spectral_attention_RR'][i_sample,0,:], alpha=0.9, color=color, zorder=2)
                
                

            # dax.set_xlim(0,200)
            dax.set_xlim(xf_masked.min(), xf_masked.max()*2)
            ax_no_top_right(dax)
            

            if j_filter==len(axes[1:])-1:
                ax.set_xlabel('time (sec)', fontsize=fontsize)
                ax.spines['bottom'].set_visible(True)
                fax.set_xlabel('freq (bpm)', fontsize=fontsize)
                fax.spines['bottom'].set_visible(True)

                fax.axvline(output[i_sample], color='firebrick', linestyle='--', alpha=0.5)
                fax.axvline(label[i_sample], color='black', linestyle='--', alpha=0.5)
                
                dax.axvline(output[i_sample], color='firebrick', linestyle='--', alpha=0.5)
                dax.axvline(label[i_sample], color='black', linestyle='--', alpha=0.5)
            
            
            
            
            if plot_STFT:
                ax_STFT = axes_STFT[j_filter+N_sigs]
                ax_STFT.pbaspect = [1, 4, 1]

                # y_matrix =  get_psd(data_layer[:, j_filter, :], N)
                # xf, y_matrix_scaled = get_psd(data_layer[:, j_filter, :], FS_Extracted)           
                xf, y_matrix_scaled = get_psd(data_layer[:, j_filter, :], FS_layer)           

                # dim: (N_instance, N_spectral) -> (N_instance, N_spectral_masked)
                t_cat_arr = np.arange(label.shape[0]) * (t_arr.max()-t_arr.min()) / (label.shape[0]-1) + t_arr.min()
                ax_STFT.plot(t_cat_arr, output, color='white', linestyle='--', alpha=0.9, linewidth=0.8)

                # ax_STFT.set_ylim(1,240)

                # ax_STFT.plot_surface(t_cat_arr[:, None], xf[mask][None, :]*60, y_matrix_scaled[:, mask], cmap=cm.coolwarm)
                # ax_STFT.set_yticks(xf[mask][::10]*60)

                some_mask = (xf*60 >= xf_masked.min()) & (xf*60 <= xf_masked.max()*2)
                ax_STFT.plot_surface(t_cat_arr[:, None], xf[None, some_mask]*60, y_matrix_scaled[:, some_mask], cmap=cm.coolwarm)
                # ax_STFT.plot_surface(t_cat_arr[:, None], xf[None, 1:]*60, y_matrix_scaled[:, 1:], cmap=cm.coolwarm)
                
                # ax_STFT.set_yticks(xf[::10]*60)
                # ax_STFT.set_ylim(5,120)


                ax_STFT.set_title('deep features', fontsize=fontsize)
                # ax_STFT.set_xticks(t_arr[::50])
                ax_STFT.set_xlabel('t (s)')
                ax_STFT.set_ylabel('freq (bpm)')

        if fig_name is None:
            fig_name = 'DL_activation'

        if log_wandb:
            wandb.log({fig_name: wandb.Image(fig)})

        if outputdir is not None:
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

            fig.savefig(outputdir + fig_name+ layer_name + '.png', facecolor=fig.get_facecolor())
            if plot_STFT:
                fig_STFT.savefig(outputdir + fig_name+ layer_name + '_STFT.png', facecolor=fig.get_facecolor())

        if show_plot == False:
            plt.close(fig)
            pyplot.close(fig)
            plt.close('all')
        plt.show()

        
        
        
        
def check_attention(CV_dict, task, training_params, mode='worst', fig_name=None, show_plot=False, outputdir=None, log_wandb=False):

    task_name = task.split('-')[-1].split('_')[0]

    label_est_val = CV_dict['performance_dict_val']['out_dict'][task]
    label_val = CV_dict['performance_dict_val']['label_dict'][task]

    error_abs = np.abs(label_est_val - label_val)

    if mode=='worst':
        i_sample = np.argmax(error_abs)
    if mode=='best':
        i_sample = np.argmin(error_abs)
    if mode=='random':
        # 2. check one sample only
        N_samples = error_abs.shape[0]
        np.random.seed(0)
        i_sample = np.random.randint(N_samples)

        
    attention_keys = CV_dict['performance_dict_val']['attention_dict_arr'].keys()

    attention_dicts = CV_dict['performance_dict_val']['attention_dict_arr']
#     attention_dicts = {}
#     for attention_key in attention_keys:
#         attention_dicts[attention_key] = []

#     for attention_key in attention_keys:
#         for i in range(len(CV_dict['performance_dict_val']['attention_dict_arr'])):
#             attention_dicts[attention_key].append(CV_dict['performance_dict_val']['attention_dict_arr'][i][attention_key])

#     for attention_key in attention_keys:
#         attention_dicts[attention_key] = np.concatenate(attention_dicts[attention_key], axis=0)
    
    fig, axes = plt.subplots(5,1,figsize=(10,8), dpi=100)

    attention_arr = attention_dicts['channel_attention_HR']
    # axes[0].plot(attention_arr[i_sample,:,0], '.-', color='gray', alpha=0.8)
    x_arr = np.arange(attention_arr.shape[1])

    
    channel_n = training_params['channel_n']
    colors = [input_color_dict['ECG']]*channel_n + [input_color_dict['SCG']]*channel_n
    axes[0].bar(x_arr, attention_arr[i_sample,:,0], color =colors, alpha=0.5, width = 0.5)
    axes[0].set_ylabel('HR ch atn')
    ax_no_top_right(axes[0])
    axes[1].imshow(attention_arr[i_sample,:,:].T, cmap='hot')
    axes[1].set_ylabel('HR ch atn')
    ax_no_top_right(axes[1])

    
    attention_arr = attention_dicts['channel_attention_RR']
    # axes[2].plot(attention_arr[i_sample,:,0], '.-', color='gray', alpha=0.8)
    x_arr = np.arange(attention_arr.shape[1])

    colors = [input_color_dict['PPG']]*channel_n + [input_color_dict['SCG']]*channel_n
    axes[2].bar(x_arr, attention_arr[i_sample,:,0], color =colors, alpha=0.5, width = 0.5)
    axes[2].set_ylabel('RR ch atn')
    ax_no_top_right(axes[2])
    axes[3].imshow(attention_arr[i_sample,:,:].T, cmap='hot')
    axes[3].set_ylabel('RR ch atn')


    axes[4].plot(training_params['HR_xf_dict']['xf_masked'], attention_dicts['spectral_attention_HR'][i_sample,0,:], color='firebrick', alpha=0.8, label='cardiac')
    axes[4].plot(training_params['RR_xf_dict']['xf_masked'], attention_dicts['spectral_attention_RR'][i_sample,0,:], color='steelblue', alpha=0.8, label='resp')

    axes[4].legend(loc='upper right')
    axes[4].set_ylabel('spectral attn')
    axes[4].set_xlabel('BPM/bpm')



    fig.tight_layout()

    if fig_name is None:
        fig_name = 'attention'

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
    
    
    
# def plot_feature_importances(feature_names, feature_importances, fig_name=None, outputdir=None, show_plot=False, log_wandb=False):

#     # fig, ax = plt.subplots(1,1, figsize=(5,feature_names.shape[0]/4), dpi=100)
#     fig, ax = plt.subplots(1,1, figsize=(5, 50), dpi=100)
#     fontsize = 12
#     ax.barh(feature_names, feature_importances)
#     ax.tick_params(axis='both', labelsize=fontsize)
#     ax_no_top_right(ax)

#     fig.tight_layout()
    
#     if outputdir is not None:
#         if not os.path.exists(outputdir):
#             os.makedirs(outputdir)
#         if fig_name is None:
#             fig_name = 'feature_importance'
#         else:
#             fig_name = fig_name

#         fig.savefig(outputdir + fig_name, bbox_inches='tight', transparent=False)

#     if log_wandb:
#         wandb.log({fig_name: wandb.Image(fig)})
        
#     if show_plot == False:
#         plt.close(fig)
#         pyplot.close(fig)
#         plt.close('all')