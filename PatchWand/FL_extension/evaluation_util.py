import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from tqdm import trange
import numpy as np
from sklearn.metrics import r2_score


import sys
sys.path.append('../') # add this line so Data and data are visible in this file
from evaluate import *
from plotting_tools import *
from setting import *
from spectral_module import *
# from dataset_util import *
from FL_extension.dataset_util import *
from FL_extension.training_util import *

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
matplotlib.rc( 'savefig', facecolor = 'white' )
from matplotlib import pyplot
from matplotlib import colors
import matplotlib.gridspec as gs


def plot_losses(CV_dict, show_plot=False, outputdir=None):
    
    df_losses_train = CV_dict['df_losses_train']
    df_losses_val = CV_dict['df_losses_val']
#     total_loss_train = np.sqrt(CV_dict['total_loss_train'])
#     total_loss_val = np.sqrt(CV_dict['total_loss_val'])
    
    metric_names = list(df_losses_train.columns)
    fig, axes = plt.subplots(len(metric_names),1, figsize=(5, len(metric_names)), dpi=80)
    
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
    #     fig_name = '{}_signl_{}'.format(title_str,subject_id)
    fig_name = 'loss_CV{}'.format(subject_id)

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
        


def check_featuremap(model, training_params, mode='worst', fig_name=None, show_plot=False, outputdir=None, log_wandb=False):

#     print(outputdir)
    inputdir = training_params['inputdir']
    device = training_params['device']
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)
    dataloader = dataloaders['val']
    

#     device = torch.device('cuda:{}'.format('cpu'))
#     device = torch.device('cpu')
#     model = model.to(device).float()

    # 1. set up the hook
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    data = torch.from_numpy(dataloader.dataset.data)
    feature = torch.from_numpy(dataloader.dataset.feature)
    data = data.to(device).float()
    feature = feature.to(device).float()

    label = dataloader.dataset.label

#     print(data.size(), feature.size(), label.shape)
#     print(data, feature, label)
    
    meta = dataloader.dataset.meta
#     print(meta.shape)

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

#     elif model_name=='CNN-DANN':
    elif model_name=='CNNlight-DANN':
#         key = list(model.feature_extractors.keys())[0]        
        for input_name in list(model.feature_extractors.keys()):
#             model.feature_extractors[input_name].basicblock_list[-1].ch_pooling.register_forward_hook(get_activation(input_name+'-layer_last'))
            model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'-layer_last'))
            layer_names = layer_names + [input_name+'-layer_last']

    elif model_name=='ResNet1D':

        try:
            for input_name in list(model.feature_extractors.keys()):
                model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'_layer_last'))
                model.feature_extractors[input_name].final_ch_pooling.register_forward_hook(get_activation(input_name+'_layer_pool'))
                layer_names = layer_names + [input_name+'_layer_last', input_name+'_layer_pool']
        except:
            for input_name in list(model.feature_extractors.keys()):
                model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'_layer_last'))
                layer_names = layer_names + [input_name+'_layer_last']
            for regressor_name in list(model.regressors.keys()):
                model.regressors[regressor_name].ch_pooling.register_forward_hook(get_activation(regressor_name+'_layer_pooling'))
                layer_names = layer_names + [regressor_name+'_layer_pooling']

    elif model_name == 'ResNet1D_LSTM':
        
        for input_name in list(model.feature_extractors.keys()):
            model.feature_extractors[input_name].basicblock_list[-1].register_forward_hook(get_activation(input_name+'_layer_last'))
            layer_names = layer_names + [input_name+'_layer_last']


    model.eval()

    # 4. pass the data to the model and the hook will take care of the rest (output stored in activation), don't really need out
    out, deep_feature = model(data, feature)
#     print(data.size())
#     sys.exit()
    data = data.cpu().detach().numpy()
#     output = output.cpu().detach().numpy()
#     label = {output_name: label[:, [training_params['output_names'].index( output_name.split('-')[0] )]] for output_name in output.keys()}

    label_dict = {}
    out_dict = {}
    for output_name in out.keys():
        if 'domain' in output_name:
            continue
        

        if training_params['dominantFreq_detect']=='expectation':
            out_HR = out[output_name].data.detach().cpu().numpy()
        elif training_params['dominantFreq_detect']=='argmax':
            out_HR, indices_dominant = get_HR(deep_feature[output_name.split('-')[1]].data.detach().cpu().numpy(), training_params['xf_masked'])

        out_dict[output_name] = out_HR
#         out_dict[output_name] = out[output_name].detach().cpu().numpy()

        label_dict[output_name] = label[:,training_params['output_names'].index(output_name.split('-')[0]) ]
#         else:
#         if 'domain' in output_name:
#             input_name = output_name.split('-')[1]
#             label_dict[output_name] = torch.ones(label.size()[0]).to(self.device) * self.modality_dict[input_name]
#         else:
#                 print(output_name)
#             label_dict[output_name] = label[:, [self.output_names.index( output_name.split('-')[0] )]].squeeze()
#             output_dict[output_name] = output[output_name].cpu().detach().numpy().squeeze()

#     print(data, feature, output, label)

    main_task = training_params['output_names'][0]

    # 5. organize these activation layers

    for layer_name in activation.keys():
#             print(layer_name, )
        activation[layer_name] = activation[layer_name].cpu().detach().numpy() # dim = (N_batch, channel_n, output_dim)
    
    # prepare for FFT
    Fs = training_params['FS_Extracted']
    T = 1/Fs
    N =  activation[layer_name].shape[-1]
#     xf = np.linspace(0.0, 1.0/2.0*Fs, N//2)*60
#     mask = (xf>=60) & (xf<=200)
    xf_masked = training_params['xf_masked']
    mask = training_params['mask']
    
    


    
    for output_name in label_dict.keys():
        if main_task not in output_name:
            continue
        output = out_dict[output_name].squeeze()
        label = label_dict[output_name]
        
        error_abs = np.abs(output - label)

        if model_name == 'ResNet1D_LSTM':
            error_abs = error_abs.mean(axis=-1)
    #     print(error_abs.shape)

        if mode=='worst':
            i_sample = np.argmax(error_abs)
        if mode=='best':
            i_sample = np.argmin(error_abs)
        if mode=='random':
            # 2. check one sample only
            N_samples = dataloader.dataset.data.shape[0]
            np.random.seed(0)
            i_sample = np.random.randint(N_samples)

        [subject_id, task_id] = meta[i_sample, :2]
#         print(out, label, meta)

        for layer_name in activation.keys():
        

        
        
            if layer_name.split('-')[0] != output_name.split('-')[1]:
                continue
#             print(layer_name.split('-')[0], output_name.split('-')[1], activation.keys(), label_dict.keys())
#             sys.exit()
#             if main_task not in output_name:
#                 continue
        
#             print('\t', layer_name, )

            data_layer = activation[layer_name]
    
            if len(data_layer.shape)<3:
                data_layer = data_layer[:,:,None]

            

            N_sigs = len(training_params['input_names'])

#             fig, axes = plt.subplots(data_layer.shape[1]+1,1, figsize=(10, data_layer.shape[1]+1), dpi=60, gridspec_kw = {'wspace':0, 'hspace':0}, facecolor='white', constrained_layout=True)
            
            
            N_axes = data_layer.shape[1]+1
            gs0 = gs.GridSpec(N_axes,2, width_ratios=[10, 2])
            fig = plt.figure(figsize=(13,2*N_axes), dpi=120)
            axes = []
            faxes = []
            for i_ch in range(N_axes):
                axes.append( fig.add_subplot(gs0[i_ch,0]) )
#                 if i_ch == N_axes-1:
#                     continue
                faxes.append( fig.add_subplot(gs0[i_ch,1]) )


#             fig.tight_layout()
            
            
            
            fontsize = 13

            FS_RESAMPLE_DL = training_params['FS_RESAMPLE_DL']
            t_arr = np.arange(data.shape[-1])/FS_RESAMPLE_DL

    #         for i, input_name in enumerate(training_params['input_names']):

            # 1. plot one physio sig at the top column
            ax = axes[0]
            fax = faxes[0]



            if '-' in layer_name:
                sig_name = layer_name.split('-')[0]
            else:
                sig_name = layer_name



            if 'ECG' in sig_name:
                unit = unit_dict['ecg']
                input_name = sig_name.split('_layer')[0]
                i_sig = training_params['input_names'].index(input_name)
    #             i_sig = training_params['input_names'].index('ECG')
            elif ('accel' in sig_name) or ('scg' in sig_name):
                unit = unit_dict['accel']
                input_name = sig_name.split('_layer')[0]
                i_sig = training_params['input_names'].index(input_name)
    #             i_sig = training_params['input_names'].index('accelZ')
            elif 'ppg' in sig_name:
                unit = unit_dict['ppg']
                input_name = sig_name.split('_layer')[0]
                i_sig = training_params['input_names'].index(input_name)
    #             i_sig = training_params['input_names'].index('ppg_g_1')

#             print(layer_name, sig_name, input_name, i_sig)

            sig_name = training_params['input_names'][i_sig]
    
            color = input_color_dict[sig_name]
    
    

    #         print(layer_name, i_sig)
    #         sys.exit()

            ax.plot(t_arr, data[i_sample,i_sig,:], color=color)
            ax.set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot
            ax.set_ylabel('{}\n[{}]'.format(sig_name, unit), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=20)

            ax.set_title('[{}] {}\nout={:.2f}, label={:.2f}, AE={:.2f} [{}]'.format(subject_id, tasks_dict_reversed[task_id], output[i_sample], label[i_sample], error_abs[i_sample], unit_dict[main_task.split('_')[0]]), fontsize=fontsize)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.set_xticklabels([])
            
            
            y = get_psd(data[i_sample,i_sig,:], N)
            y_scaled = 2.0/N * (np.abs(y[:N//2])**2)
#             fax.plot(xf[mask], y_scaled[mask])
            fax.plot(xf_masked, y_scaled[mask], color=color)
            
#             fax.axvline(output[i_sample], color='firebrick', linestyle='--', alpha=0.7)
            fax.axvline(label[i_sample], color='black', linestyle='--', alpha=0.7)

            ax_no_top_right(fax)
            fax.set_xticklabels([])

            
            
            
            
            fig_STFT, axes_STFT = plt.subplots(N_axes,1, figsize=(6, 5), dpi=100, facecolor='white')

            ax_STFT = axes_STFT[0]
            y_matrix =  get_psd(data[:,i_sig,:], N)
            # dim: (N_instance, N_spectral) -> (N_instance, N_spectral_masked)
            y_matrix_scaled= 2.0/N * (np.abs(y_matrix[:,:N//2])**2)[:, mask]
#             aaa_scaled = aaa_scaled[:, mask]

            ax_STFT.imshow( y_matrix_scaled.T[::-1, :], interpolation='nearest', aspect='auto' ,extent=[t_arr.min(),t_arr.max(), xf_masked.min(),xf_masked.max()], cmap='viridis')
            ax_STFT.set_title('raw signal', fontsize=fontsize)
            ax_STFT.set_yticks(xf_masked[::10])
            ax_STFT.set_xticks(t_arr[::300])
            
            
            
            
            t_cat_arr = np.arange(label.shape[0]) * (t_arr.max()-t_arr.min()) / (label.shape[0]-1) + t_arr.min()
#             ax_STFT.plot(t_cat_arr, label, color='white')
            ax_STFT.plot(t_cat_arr, output, color='firebrick', linestyle='--', alpha=0.7, linewidth=0.5)

#             fig_STFT.savefig(outputdir + fig_name+ layer_name + '.png', facecolor=fig.get_facecolor())


#             plt.show()
#             sys.exit()


            # 2. next, plot the feature map 
    #         t_arr = np.arange(data_layer.shape[-1]) / ( FS_RESAMPLE_DL / (2**(training_params['n_block_macro'] - 1)) )
#             t_arr = np.arange(data_layer.shape[-1]) / ( FS_RESAMPLE_DL / (training_params['stride']**(training_params['n_block']) ) )
            t_arr = np.arange(data_layer.shape[-1]) / training_params['FS_Extracted']

    #         t_arr = np.arange(data_layer.squeeze().shape[-1]) / ( FS_RESAMPLE_DL / (training_params['stride'] **(training_params['n_block_macro'] - 1)) )

#             for j_filter, ax in enumerate(axes[1:]):
            for j_filter in range(len(axes)-1):
#                 if j_filter==0:
#                     continue
#                 print(len(axes), len(faxes), j_filter)
                ax = axes[j_filter+1]
                fax = faxes[j_filter+1]
                if j_filter!=len(axes[1:])-1:
                    ax.set_xticklabels([])
                    fax.set_xticklabels([])

    #             print( data_layer[i_sample, j_filter, :].shape)

                ax.plot(t_arr, data_layer[i_sample, j_filter, :].squeeze(), alpha=0.9, color=color)
    #             ax.plot(t_arr, data_layer[i_sample, j_filter, :].T, alpha=1)

                ax.set_xlim(t_arr.min(), t_arr.max()) # remove the weird white space at the beg and end of the plot

                # remove some borders (top and right)
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['bottom'].set_visible(False)


                ax.set_ylabel('filter {}'.format(j_filter), fontsize=fontsize,rotation = 0,  va='center', ha='center',  labelpad=20)
                # set tick font size
                ax.tick_params(axis='both', which='major', labelsize=fontsize*0.8)

                if j_filter==len(axes[1:])-1:
                    ax.set_xlabel('time (sec)', fontsize=fontsize)
                    ax.spines['bottom'].set_visible(True)
                    fax.set_xlabel('freq (bpm)', fontsize=fontsize)
                    fax.spines['bottom'].set_visible(True)

                y = get_psd(data_layer[i_sample, j_filter, :].squeeze(), N)
                y_scaled = 2.0/N * (np.abs(y[:N//2])**2)
                

                fax.plot(xf_masked, y_scaled[mask].T, alpha=0.9, color=color)

#                 fax.plot(xf[mask], y_scaled[mask].T)

                fax.axvline(output[i_sample], color='firebrick', linestyle='--', alpha=0.7)
                fax.axvline(label[i_sample], color='black', linestyle='--', alpha=0.7)
                ax_no_top_right(fax)
            
            
            
            
            
                ax_STFT = axes_STFT[j_filter+1]
                
                y_matrix =  get_psd(data_layer[:, j_filter, :], N)
                # dim: (N_instance, N_spectral) -> (N_instance, N_spectral_masked)
                y_matrix_scaled= 2.0/N * (np.abs(y_matrix[:,:N//2])**2)[:, mask]

                ax_STFT.imshow( y_matrix_scaled.T[::-1, :], interpolation='nearest', aspect='auto' ,extent=[t_arr.min(),t_arr.max(), xf_masked.min(),xf_masked.max()], cmap='viridis')

                t_cat_arr = np.arange(label.shape[0]) * (t_arr.max()-t_arr.min()) / (label.shape[0]-1) + t_arr.min()
#                 ax_STFT.plot(t_cat_arr, label, color='white')
                ax_STFT.plot(t_cat_arr, output, color='firebrick', linestyle='--', alpha=0.7, linewidth=0.5)
                
                ax_STFT.set_title('deep features', fontsize=fontsize)

                ax_STFT.set_yticks(xf_masked[::10])
                ax_STFT.set_xticks(t_arr[::300])

    #         fig.subplots_adjust(wspace=0, hspace=0)

#             fig_STFT.tight_layout()

            if fig_name is None:
                fig_name = 'DL_activation'
    #             print('hihi', sig_name)
    #         fig_name = 'DL_activation_'+sig_name

            if log_wandb:
                wandb.log({fig_name: wandb.Image(fig)})

            if outputdir is not None:
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                    
#                 print('saving to', outputdir + fig_name+ layer_name + '.png')
                fig.savefig(outputdir + fig_name+ layer_name + '.png', facecolor=fig.get_facecolor())
                fig_STFT.savefig(outputdir + fig_name+ layer_name + '_STFT.png', facecolor=fig.get_facecolor())

            if show_plot == False:
                plt.close(fig)
                pyplot.close(fig)
                plt.close('all')
            plt.show()


    #         return fig