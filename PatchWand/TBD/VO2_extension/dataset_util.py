import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from dataIO import *
from filters import *

# subject independent
def get_samples(inputdir, set_name, training_params):
    input_names = training_params['input_names']  
    feature_names = training_params['feature_names']
    # output_names = training_params['output_names']
    
    if 'regression_names' not in training_params:
        output_names = training_params['output_names']
    else:
        output_names = training_params['regression_names']
    meta_names = training_params['meta_names']
    
    list_feature = training_params['list_feature']
    list_signal = training_params['list_signal']
    list_output = training_params['list_output']
    list_meta = training_params['list_meta']

    indices_sig = []
    for sig_name in input_names:
        i_sig = list_signal.index(sig_name)
        indices_sig.append(i_sig)

    indices_feature = []
    for feature_name in feature_names:
        i_feature = list_feature.index(feature_name)
        indices_feature.append(i_feature)
        
    indices_label = []
    for label_name in output_names:
        i_label = list_output.index(label_name.split('-')[0])
        # print(i_label, label_name)
        indices_label.append(i_label)

    indices_meta = []
    for meta_name in meta_names:
        i_meta = list_meta.index(meta_name)
        indices_meta.append(i_meta)

    data_all = data_loader('data', inputdir)[:,indices_sig,:]
    feature_all = data_loader('feature', inputdir)[:, indices_feature]
    label_all = data_loader('label', inputdir)[:, indices_label]
    
    
    
    if training_params['sequence']:
        if 'output_dim' in training_params:
            feature_all, downsample_factor = reduce_data_dim(feature_all, training_params)

            label_all, downsample_factor = reduce_data_dim(label_all, training_params)

            training_params['downsample_factor'] = downsample_factor

            if training_params['output_dim']==1:
                feature_all = feature_all[:,:,0]
                label_all = label_all[:,:,0]
#     else:
        

    meta_all = data_loader('meta', inputdir)[:, indices_meta]

    subject_id = training_params['CV_config']['subject_id']
    task_ids = training_params['CV_config']['task_ids']
    training_mode = training_params['training_mode']

#     if 'train' in set_name:
#         mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]!=task_id)
#     else:
#         mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]==task_id)

#     if ['training_mode']

    # select the subject(s)
    if training_mode == 'subject_ind':
        if 'train' in set_name:
            mask_set = (meta_all[:,0]!=subject_id)
        else:
            mask_set = (meta_all[:,0]==subject_id)
    elif training_mode == 'subject_specific':
        if 'train' in set_name:
            mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]!=task_id) # test on task_id, use the rest for training
        else:
            mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]==task_id) # test on task_id

    # select the task(s)
    mask_task = np.in1d(meta_all[:,1], task_ids)
    mask_set = (mask_set) & (mask_task)

    if 'reject_subject_id' in training_params['CV_config']:
        reject_subject_id = training_params['CV_config']['reject_subject_id']
        mask_set = (mask_set) & (~np.isin(meta_all[:,0], reject_subject_id))

    
            
    data_set = data_all[mask_set,:,:]
    feature_set = feature_all[mask_set,:]
    label_set = label_all[mask_set,:]
    meta_set = meta_all[mask_set,:]


    
#     print(data_set.shape, label_set.shape, feature_set.shape, meta_set.shape)

    return data_set, feature_set, label_set, meta_set

def reduce_data_dim(data, training_params):
    # data has a dimension of (N_instances, N_feature/signals, N_samples)
    # data_reduced has a dimension of (N_instances, N_feature/signals, output_size)
    # downsample_factor = N_samples // output_size

    data_reduced = np.zeros((data.shape[0], data.shape[1], training_params['output_dim']))
    
    downsample_factor = data.shape[-1] // training_params['output_dim']

    for i_instance in range(data_reduced.shape[0]):
        for i_signal in range(data_reduced.shape[1]):
            sig = data[i_instance, i_signal, :]
            sig_smoothed = get_smooth(sig, N=int(training_params['FS_RESAMPLE_DL']*5))
            sig_smoothed = sig_smoothed[downsample_factor//2:][::downsample_factor]
            data_reduced[i_instance, i_signal, :] = sig_smoothed
    
    return data_reduced, downsample_factor

# def get_samples(inputdir, set_name):
    
#     inputdir_set = inputdir+set_name
    
#     input_names = training_params['input_names']
#     output_names = training_params['output_names']
    
    
#     data = data_loader('data', inputdir_set)[:,0,:][:,None,:] # get ECG only
#     label = data_loader('label', inputdir_set)[:, [0, -1]] # get RR and task
    
#     return data, label


class FS_Dataset(Dataset):
    def __init__(self, data, feature, label, meta, training_params, transform=None):
        self.data = data
        self.feature = feature
        self.label = label
        self.meta = meta
        
    def __getitem__(self, index):
        data = self.data[index,:,:]
#         label = np.asarray(self.label[index,0]).astype(float)
        feature = self.feature[index, :]
        label = self.label[index, :].astype(float)
        meta = self.meta[index,:].astype(float)
        
#         print(meta)
        
        data = torch.from_numpy(data)
        feature = torch.from_numpy(feature)
        label = torch.from_numpy(label)
        meta = torch.from_numpy(meta)

        
        return data, feature, label, meta

    def __len__(self):
#         return len(self.data)
        return self.data.shape[0]



def clip_to_std(data, N_std=2):
    
    data = data-np.mean(data, axis=-1)[:,:,None]
    clip_thre = np.std(data, axis=-1)[:,:,None]*N_std

    data = np.clip(data, -clip_thre, clip_thre)
    return data, clip_thre

def get_loaders(inputdir, training_params):

    data_train, feature_train, label_train, meta_train = get_samples(inputdir, 'train/', training_params)
    data_val, feature_val, label_val, meta_val = get_samples(inputdir, 'val/', training_params)
    
    # normalize SCG and PPG signals
    if 'normalization' in training_params:

        if training_params['normalization'] == 'input_normed_clipped':   
            clipping_thre = 10
            for input_name in training_params['input_names']:
                # i_scg = training_params['input_names'].index('scgZ')
                if 'scg' not in input_name:
                    continue
                # dimension of data is (N_instances, N_sig, N_samples)
                i_scg = training_params['input_names'].index(input_name)

                data_train[:,i_scg, :] = np.clip(data_train[:,i_scg, :], -clipping_thre, clipping_thre) # scg can only be in +/- 10mg
                data_val[:,i_scg, :] = np.clip(data_val[:,i_scg, :], -clipping_thre, clipping_thre) # scg can only be in +/- 10mg
            

        if training_params['normalization'] == 'input_normed_clipped2std':
            clipping_thre = 5
            
            for input_name in training_params['input_names']:
                # i_scg = training_params['input_names'].index('scgZ')
                if 'ECG' in input_name:
                    continue
                i_sig = training_params['input_names'].index(input_name)

                data_clipped, clip_thre = clip_to_std(data_train[:,[i_sig], :], N_std=clipping_thre)
                data_train[:,[i_sig], :] = data_clipped

                data_clipped, clip_thre = clip_to_std( data_val[:,[i_sig], :], N_std=clipping_thre)
                data_val[:,[i_sig], :] = data_clipped
            
#             i_scg = training_params['input_names'].index('scgZ')
#             data_clipped, clip_thre = clip_to_std(data_train[:,[i_scg], :], N_std=clipping_thre)
#             data_train[:,[i_scg], :] = data_clipped
            
#             data_clipped, clip_thre = clip_to_std( data_val[:,[i_scg], :], N_std=clipping_thre)
#             data_val[:,[i_scg], :] = data_clipped
            
        # normalizae the 60s signal so it has 0 mean and unit variance
        if training_params['normalization'] == 'input_normed':   
            data_mean = np.mean(data_train, axis=-1, keepdims=True)
            data_std = np.std(data_train, axis=-1, keepdims=True)
            data_train = (data_train-data_mean)/data_std

            data_mean = np.mean(data_val, axis=-1, keepdims=True)
            data_std = np.std(data_val, axis=-1, keepdims=True)
            data_val = (data_val-data_mean)/data_std
            
            
    
    
    # zero mean unit variance
    feature_mean = np.mean(feature_train, axis=0)
    feature_std = np.std(feature_train, axis=0)

    feature_train = (feature_train-feature_mean)/feature_std
    feature_val = (feature_val-feature_mean)/feature_std

#     print(feature_mean, feature_std)

    train_dataset = FS_Dataset(data_train, feature_train, label_train, meta_train, training_params)
    val_dataset = FS_Dataset(data_val, feature_val, label_val, meta_val, training_params)
#         test_dataset = PTBXL_Dataset(data_test, label_test, training_params, transform=transforms)


    FS_datasets = {
        'train': train_dataset, 'val': val_dataset
    }
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=0),
        'train_eval': DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0),
        'feature_mean': feature_mean,
        'feature_std': feature_std,
    }
    

    dataset_sizes = {
        x: len(FS_datasets[x]) for x in FS_datasets.keys()
    }
    
    return dataloaders, dataset_sizes

