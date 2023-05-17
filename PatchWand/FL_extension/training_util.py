import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from dataIO import *
from stage3_preprocess import *
# from dataset_util import *
from FL_extension.dataset_util import *
from handy_tools import *
import wandb

# def train_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def train_dann(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)
#     total_loss = 0

    total_AE = dict.fromkeys(training_params['model_out_names'], 0)
    total_AE = {k:v for k,v in total_AE.items() if 'domain' not in k}

    total_losses = dict.fromkeys(training_params['model_out_names']+['total'], 0)


    model.train()
    
    for i, (data, feature, label, meta) in enumerate(dataloader):
#         print('epoch', i)
        # 1. get data
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out, deep_feature = model(data, feature)
        
#         print(deep_feature.size())
#         print(torch.argmax(deep_feature, axis=))
#         i_dominant = torch.argmax(deep_feature, axis=-1)

#         deep_feature = np.random.rand(20, 3, xf_masked.shape[0])
        

#         sys.exit()
#         out = model(data)
        
        # 3. loss function
        losses = criterion(out, label)

        # 3. Backward and optimize
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()

        # 4. accumulate the loss
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()
            
        # 5. compute metric
        for AE_names in total_AE.keys():
            
#             print('total', total_AE.keys())
#             print('deep_feature', deep_feature)
#             print('out', out)
#             sys.exit()
            
            main_task = AE_names.split('-')[0]
            label_AE = label[:, [training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
            

            if training_params['dominantFreq_detect']=='expectation':
                out_AE = out[AE_names].data.detach().cpu().numpy()
            elif training_params['dominantFreq_detect']=='argmax':
                out_AE, indices_dominant = get_HR(deep_feature[AE_names.split('-')[1]].data.detach().cpu().numpy(), training_params['xf_masked'])
#             print(out_argmax, out)           
            total_AE[AE_names] += np.sum(np.abs(out_AE.squeeze()-label_AE.squeeze()).squeeze())

            
    for loss_name in total_losses.keys():
        total_losses[loss_name] == total_losses[loss_name]/dataset_size
            
    subject_id = training_params['CV_config']['subject_id']
    
    # don't log training metrics
#     log_dict = {}
#     for AE_names in total_AE.keys():
#         log_dict['[{}] train_{}'.format(subject_id, AE_names)] = total_AE[AE_names]/dataset_size

#     log_dict['epoch'] = epoch
    
#     if training_params['wandb']==True:
#         # W&B
#         wandb.log(log_dict)

    performance_dict = total_losses

    return performance_dict


# def eval_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def eval_dann(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

#     total_loss = 0

    total_losses = dict.fromkeys(training_params['model_out_names']+['total'], 0)
        
    total_AE = dict.fromkeys(training_params['model_out_names'], 0)
    total_AE = {k:v for k,v in total_AE.items() if 'domain' not in k}
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, feature, label, meta) in enumerate(dataloader):
        # 1. get data        
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out, deep_feature = model(data, feature)
            
        # 3. loss function
        losses = criterion(out, label)
#         loss = losses['total']

        # 4. accumulate the loss
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()

        for AE_names in total_AE.keys():
            main_task = AE_names.split('-')[0]
            label_AE = label[:, [training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
#             out_AE = out[AE_names].data.detach().cpu().numpy()
            if training_params['dominantFreq_detect']=='expectation':
                out_AE = out[AE_names].data.detach().cpu().numpy()
            elif training_params['dominantFreq_detect']=='argmax':
                out_AE, indices_dominant = get_HR(deep_feature[AE_names.split('-')[1]].data.detach().cpu().numpy(), training_params['xf_masked'])

            total_AE[AE_names] += np.sum(np.abs(out_AE.squeeze()-label_AE.squeeze()).squeeze())

#         sys.exit()
        
    for loss_name in total_losses.keys():
        total_losses[loss_name] == total_losses[loss_name]/dataset_size
            
    subject_id = training_params['CV_config']['subject_id']
    
    log_dict = {}
    for AE_names in total_AE.keys():
        log_dict['[{}] val_{}'.format(subject_id, AE_names)] = total_AE[AE_names]/dataset_size

    log_dict['epoch'] = epoch
    
    if training_params['wandb']==True:
        # W&B
        wandb.log(log_dict)
    
#     performance_dict = {'total_loss': total_loss,
#                        }
    performance_dict = total_losses
    
    return performance_dict




# TODO: export deep features too
def pred_dann(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

#     total_loss = 0
    total_losses = dict.fromkeys(training_params['model_out_names']+['total'], 0)

    out_dict = {}
    for model_out_name in training_params['model_out_names']:
        out_dict[model_out_name] = []
    # don't do the following since it's a shallow copy (copy the address only)
#     out_dict = dict.fromkeys(training_params['model_out_names'], list() )

    label_dict = {}
    for model_out_name in training_params['model_out_names']:
        label_dict[model_out_name] = []


    feature_arr = []
    meta_arr = []
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, feature, label, meta) in enumerate(dataloader):
        # 1. get data
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out, deep_feature = model(data, feature)
#         out = model(data)
            
        # 3. loss function
        losses = criterion(out, label)

        # 4. accumulate the loss
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()
            


        # TODO: fix this block (out_dict)
#         for output_name in out.keys():
#         print(label)

        for output_name in out_dict.keys():
#             print(output_name)
#             print('===== out =====')
#             out_dict[output_name].append(out[output_name].detach().cpu().numpy())


            if 'domain' in output_name or training_params['dominantFreq_detect']=='expectation':
#             if training_params['dominantFreq_detect']=='argmax':
                out_dict[output_name].append(out[output_name].detach().cpu().numpy())
            else:
#             elif training_params['dominantFreq_detect']=='argmax':
                out_HR, indices_dominant = get_HR(deep_feature[output_name.split('-')[1]].data.detach().cpu().numpy(), training_params['xf_masked'])
                out_dict[output_name].append(out_HR)

#             if 'domain' in output_name:
#                 print(output_name, out_dict[output_name])
#                 sys.exit()
#             print(output_name, out_dict[output_name], out[output_name].detach().cpu().numpy().shape)

#             out_dict[output_name].append(output_name].detach().cpu().numpy())

#             print('===== label =====')
            if 'domain' in output_name:
                input_name = output_name.split('-')[1]
                label_dict[output_name].append( np.ones(label.size()[0]) * training_params['modality_dict'][input_name] )

#                 label_int = np.ones(label.size()[0]).astype(int)  * training_params['modality_dict'][input_name]
#                 label_onehot = np.zeros((label_int.size, len(training_params['input_names'])))
#                 label_onehot[np.arange(label_int.size),label_int] = 1

# #                 print(output_name,  label_dict[output_name], label_onehot.shape)

#                 label_dict[output_name].append(label_onehot)
#                 print( label_dict[output_name], label_onehot.shape)
#                 sys.exit()
            else:
                label_dict[output_name].append( label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy() )
#                 print(output_name, label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy().shape)


#             if len(out[output_name].size())<2:
#             if 'LSTM' not in training_params['model_name']:
#                 print(output_name, out_dict[output_name].shape, out[output_name].detach().cpu().numpy().shape)
        
#                 out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy()]
#             else: # if lstm, take the average of the output
#                 out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze().mean(axis=-1)]
    
# #             print(label.size())
# #             print(training_params['output_names'].index(output_name.split('-')[0]))
# #             print(label[:,training_params['output_names'].index(output_name.split('-')[0]) ])
# #             sys.exit()
# #             print(len(label.size()), label.size())
#             if 'LSTM' not in training_params['model_name']:
#                 print(label_dict[output_name].shape, label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy().shape)
#                 label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy() ]
#             else:
#                 label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]), :].detach().cpu().numpy().squeeze().mean(axis=-1) ]

#             label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['tasks'].index(output_name.split('-')[0])].detach().cpu().numpy().squeeze() ]
#             label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['tasks'].index('_'.join(output_name.split('_')[:2]))].detach().cpu().numpy().squeeze()]
        

#         print(meta.shape)
        feature_arr.append( feature.detach().cpu().numpy())
        meta_arr.append( meta.detach().cpu().numpy())




    for output_name in out_dict.keys():
        out_dict[output_name] = np.concatenate(out_dict[output_name]).squeeze()
#         print('out', output_name, out_dict[output_name].shape)
#         print(output_name)
#         for aaa in out_dict[output_name]:
#             print('\t', aaa.shape)
    for output_name in label_dict.keys():
        label_dict[output_name] = np.concatenate(label_dict[output_name])
#         print('label', output_name, label_dict[output_name].shape)


#         print(output_name)
#         for aaa in label_dict[output_name]:
#             print('\t', aaa.shape)
#     sys.exit()
        
    
    feature_arr = np.concatenate(feature_arr,axis=0)
    meta_arr = np.concatenate(meta_arr,axis=0)
    
#     print(meta_arr.shape, feature_arr.shape, out_dict, label_dict)
        
    # TODO: plot feature maps and filters
    performance_dict = {'out_dict': out_dict,
                        'label_dict': label_dict,
                        'meta_arr': meta_arr,
                        'feature_arr': feature_arr,
                       }
    

    for loss_name in total_losses.keys():
        total_losses[loss_name] == total_losses[loss_name]/dataset_size
    
    performance_dict = Merge(performance_dict, total_losses)

    
    return performance_dict


    
def plot_DR_features(ax, df_features, training_params, fig_name=None):

    # features = (n_samples, n_features)
    features = StandardScaler().fit_transform(df_features[training_params['xf_masked']].values) # normalizing the features

#     print('show standardize mean and std:', np.mean(features, axis=0),np.std(features, axis=0))
#     print('show standardize mean and std:', np.mean(features),np.std(features))

    if training_params['DR_mode']=='tSNE':
        RANDOM_STATE = 0
        tsne = TSNE(n_components=3, perplexity=30, random_state=RANDOM_STATE)
        DR_features = tsne.fit_transform(features)
    
        plt_title = fig_name + '_' + training_params['DR_mode']

    elif training_params['DR_mode']=='PCA':
        pca_features = PCA(n_components=3)
        DR_features = pca_features.fit_transform(features)
        var_pca = np.cumsum(np.round(pca_features.explained_variance_ratio_, decimals=3)*100)
    #     print('PCA var:', var_pca)
        explained_var = var_pca[1]
        
        plt_title = fig_name + '_' + '{}\n{} (explained_var: {:.2f}%)'.format(training_params['DR_mode'], df_features['input_name'].unique(), explained_var)
        

    df_features.loc[:,'DR1'] = DR_features[:, 0]
    df_features.loc[:,'DR2'] = DR_features[:, 1]
    df_features.loc[:,'DR3'] = DR_features[:, 2]



    sns.scatterplot(data=df_features, x="DR1", y="DR2", hue="input_name", ax=ax, palette=input_color_dict, alpha=0.5, legend = False)

    ax.set_xlabel('Dimension - 1',fontsize=12)
    ax.set_ylabel('Dimension - 2',fontsize=12)
    ax.set_title(plt_title,fontsize=15)
    # ax.set_title('PCA of features extracted by Gf ({})'.format(col_name),fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
#     ax.legend(loc='upper right', prop={'size': 15})
    ax_no_top_right(ax)

    
def plot_spectral_features(ax, df_features, training_params, fig_name=None):
    df = pd.melt(df_features, id_vars=['input_name'], value_vars=training_params['xf_masked']).rename(columns={'variable':'xf_masked','value':'psd'})
    sns.lineplot(ax=ax, data=df, x="xf_masked", y="psd", hue="input_name", palette=input_color_dict)
    
    ax.set_xlabel('HR (bpm)',fontsize=12)
    ax.set_ylabel('PSD (norm=1)',fontsize=12)
    ax.set_title('spectral_{}'.format(fig_name),fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='upper right', prop={'size': 15})
    ax_no_top_right(ax)
    
# TODO: implement this based on model_features_diagnosis
# ref: https://github.com/chanmi168/Fall-Detection-DAT/blob/master/falldetect/eval_util.py
def visualize_latent(model, training_params, fig_name=None, show_plot=False, outputdir=None, log_wandb=False):

    dataloaders, dataset_sizes = get_loaders(training_params['inputdir'], training_params)
    data = dataloaders['val'].dataset.data
    feature = dataloaders['val'].dataset.feature
    meta = dataloaders['val'].dataset.meta
    label = dataloaders['val'].dataset.label

    data = torch.from_numpy(data)
    feature = torch.from_numpy(feature)

    data = data.to(device=training_params['device'], dtype=torch.float)
    feature = feature.to(device=training_params['device'], dtype=torch.float)
    
    # 2. infer by net
    out, feature_out = model(data, feature)
    
    df_features = []
    
    for input_name in feature_out.keys():
        feature_sig = feature_out[input_name].cpu().detach().numpy()
        df = pd.DataFrame(data=feature_sig, index=None, columns=training_params['xf_masked'])
        df['input_name'] = input_name
        df_features.append(df)
        
    df_features = pd.concat(df_features)
    
    
    if fig_name is None:
        fig_name = 'features'
    else:
        fig_name = 'features_'+fig_name

    fig, axes = plt.subplots(1,2, figsize=(10, 5), dpi=80)
    ax = axes[0]
    plot_DR_features(ax, df_features, training_params, fig_name=fig_name)

    ax = axes[1]
    plot_spectral_features(ax, df_features, training_params, fig_name=fig_name)

    fig.tight_layout()

    if log_wandb:
        wandb.log({'DR_'+fig_name: wandb.Image(fig)})

    if outputdir is not None:
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        fig.savefig(outputdir + fig_name + '.png', facecolor=fig.get_facecolor())

    if show_plot == False:
        plt.close(fig)
        pyplot.close(fig)
        plt.close('all')
    plt.show()


    
    plt.show()
#   src_feature, src_class_out, src_domain_out = model(src_data)
    
    return


def train_model(model, training_params, dataloaders, trainer, evaler, preder):

#     inputdir = training_params['inputdir']
    
#     dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    
#     total_loss_train = np.zeros(training_params['num_epochs'])
#     total_loss_val = np.zeros(training_params['num_epochs'])
    total_losses_train = dict.fromkeys(training_params['model_out_names']+['total'], np.zeros(training_params['num_epochs']))
    total_losses_val = dict.fromkeys(training_params['model_out_names']+['total'], np.zeros(training_params['num_epochs']))

#     if training_params['wandb']==True:
#         # tell wandb to watch what the model gets up to: gradients, weights, and more!
#         wandb.watch(model, log="all", log_freq=10)
    
    subject_id = training_params['CV_config']['subject_id']
    print('\t start training.....')
    print('\t validate on subject {}'.format(subject_id))

    df_losses_train = pd.DataFrame()
    df_losses_val = pd.DataFrame()
        
    for epoch in range(training_params['num_epochs']):
        if epoch%1000==1:
            print('\t[{}th epoch]'.format(epoch))
        training_params['epoch'] = epoch

        ##### model training mode ####
        performance_dict_train = trainer(model, dataloaders['train'], training_params)
        
        df_losses_train = df_losses_train.append(  pd.DataFrame(performance_dict_train, index=[0]), ignore_index=True )

#         print(performance_dict_train)
#         total_loss_train[epoch] = performance_dict_train['total']

        performance_dict_val = evaler(model, dataloaders['val'], training_params)
        df_losses_val = df_losses_val.append(  pd.DataFrame(performance_dict_val, index=[0]), ignore_index=True )
        
        if training_params['regressor_type']=='DominantFreqRegression':
            if epoch==1 or epoch==10 or epoch==training_params['num_epochs']//2 or epoch==training_params['num_epochs']-1:
                visualize_latent(model, training_params, fig_name='epoch{}'.format(epoch), show_plot=False, outputdir=training_params['outputdir_feature']+'{}/'.format(subject_id), log_wandb=False)
                
                
#         total_loss_val[epoch] = performance_dict_val['total']

    print('\t done with training.....')
    
#     print(df_losses_train)
#     print(df_losses_val)

#     sys.exit()
    performance_dict_train = preder(model, dataloaders['train'], training_params)
    performance_dict_val = preder(model, dataloaders['val'], training_params)

    
    CV_dict = {
        'performance_dict_train': performance_dict_train,
#         'total_loss_train': df_losses_train['total'].values,
        'df_losses_train': df_losses_train,
        'performance_dict_val': performance_dict_val,
#         'total_loss_val': df_losses_val['total'].values,
        'df_losses_val': df_losses_val,
        'model': model,
        'subject_id_val': training_params['CV_config']['subject_id'], 
    }
    
    return CV_dict


def change_output_dim(training_params):
    # TODO: return output_dim instead of training_params
    input_dim = training_params['data_dimensions'][1]
    output_dim = input_dim

    for i_macro in range(training_params['n_block_macro']-1):
        output_dim = np.ceil(output_dim/training_params['stride'])

    output_dim = int(output_dim)
    training_params['output_dim'] = output_dim
    return training_params


def update_freq_meta(training_params):
    # change the FS_Extracted at the last layer of conv net based on n_block
    training_params['FS_Extracted'] = training_params['FS_RESAMPLE_DL'] / (training_params['stride']**training_params['n_block'])

    # compute last layer dimension based on n_block
    last_layer_dim = training_params['data_dimensions'][-1]
    for n in range(training_params['n_block']):
        last_layer_dim = round(last_layer_dim/training_params['stride'])

    training_params['last_layer_dim'] = last_layer_dim

    # compute xf based on FS_Extracted and a mask + xf_masked using xf and label_range_dict['HR_DL']
    xf = np.linspace(0.0, 1.0/2.0*training_params['FS_Extracted'] , training_params['last_layer_dim']//2)*60    
    mask = (xf>=label_range_dict['HR_DL'][0]) & (xf<=label_range_dict['HR_DL'][1])

    training_params['xf'] = xf
    training_params['xf_masked'] = xf[mask]
    training_params['mask'] = mask
    
    return training_params




def get_HR(deep_feature, xf_masked, debug=False):
    # deep_feature has the dimension of (N_instances, N_samples in a window)
    # xf_masked has the dimension of N_samples in a window (in freq domain, bpm)
    
    # compute the dominant frequency (assuming the last dimension store the spectral distribution)
    indices_dominant = np.argmax(deep_feature, axis=-1)
    # print(indices_dominant.shape)
    
    # repeat xf_masked so it has a dimension of (N_samples, N_instances)
    xf_repeated = np.tile(xf_masked, (indices_dominant.shape+(1,)))
    
#     # reshape xf_repeated so it has a dimension of (N_instances, N_sig, N_samples)
#     xf_repeated = xf_repeated.reshape(-1, xf_repeated.shape[-1])

    # map indices_dominant the right HR based on xf_repeated 
    # dim of xf_mapped: (N_instances, N_sig, N_samples)
    xf_mapped = xf_repeated[ range(xf_repeated.shape[0]), indices_dominant]
    
#     # reshape xf_mapped so it has the dimension of  (N_instances,)
#     xf_mapped = xf_mapped.reshape(deep_feature[:,0].shape) # in bpm, or whatever frequency xf_masked has
    
#     indices_dominant = indices_dominant.reshape(deep_feature[:,:,0].shape) # integers
    
    if debug:
        j = 15
#         h = 1
        i = np.argmax(deep_feature[j,:])
    #     print(xf_masked[i])
        i_mapped = indices_dominant.reshape(deep_feature[:,0].shape)[j]
        print('i is {}, i_max is {}'.format(i, i_mapped) )
    
    return xf_mapped, indices_dominant

# def get_HR(deep_feature, xf_masked, debug=False):
#     # deep_feature has the dimension of (N_instances, N_sig, N_samples in a window)
#     # xf_masked has the dimension of N_samples in a window (in freq domain, bpm)
    
#     # compute the dominant frequency (assuming the last dimension store the spectral distribution)
#     indices_dominant = np.argmax(deep_feature, axis=-1)

#     # reshape it into a vector so it's easier to use
#     indices_dominant = indices_dominant.reshape(-1)
#     # print(indices_dominant.shape)
    
#     # repeat xf_masked so it has a dimension of (N_samples, N_instances x N_sig)
#     xf_repeated = np.tile(xf_masked, (indices_dominant.shape+(1,)))
    
# #     # reshape xf_repeated so it has a dimension of (N_instances, N_sig, N_samples)
# #     xf_repeated = xf_repeated.reshape(-1, xf_repeated.shape[-1])

#     # map indices_dominant the right HR based on xf_repeated
#     xf_mapped = xf_repeated[ range(xf_repeated.shape[0]), indices_dominant]
    
#     # reshape xf_mapped so it has the dimension of  (N_instances, N_sig)
#     xf_mapped = xf_mapped.reshape(deep_feature[:,:,0].shape) # in bpm, or whatever frequency xf_masked has
    
#     indices_dominant = indices_dominant.reshape(deep_feature[:,:,0].shape) # integers
    
#     if debug:
#         j = 15
#         h = 1
#         i = np.argmax(deep_feature[j,h,:])
#     #     print(xf_masked[i])
#         i_mapped = indices_dominant.reshape(deep_feature[:,:,0].shape)[j,h]
#         print('i is {}, i_max is {}'.format(i, i_mapped) )
    
#     return xf_mapped, indices_dominant