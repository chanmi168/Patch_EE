import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import pandas as pd

import sys
# sys.path.append('../../') # add this line so Data and data are visible in this file
sys.path.append('../') # add this line so Data and data are visible in this file

from dataIO import *
from setting import *

from imgaug import augmenters as iaa
import imgaug as ia

from resp_module import *

def get_samples(inputdir, training_params, set_name='train'):
#     inputdir_set = inputdir+set_name+'/'

    data_all = data_loader('data', inputdir)
    label_all = data_loader('label', inputdir)
    meta_all = data_loader('meta', inputdir)
    ts_all = data_loader('ts', inputdir)
    raw_all = data_loader('raw', inputdir)
#     hr = data_loader('hr', inputdir_set)



    list_input = list(training_params['surrogate_names'])
    input_names = training_params['input_names']

    if ('CDC_dataset' in training_params['outputdir']) & training_params['select_channel']: # select the best ppg channel
        list_data = []
        indices_ppg = get_indices_ppg(data_all, meta_all, list_input)

        for input_name in input_names:
#             if 'PPG' in input_name:
            if input_name=='PPG': # the old fasion way
                d = data_all[np.arange(data_all.shape[0]),indices_ppg, :,:]
                d = d[:,None,:,:]
                
                
            elif 'select' in input_name: # the new way to select PPG/SCG channel 
                if input_name=='PPG_select':
                    list_valid = ['ppg_g_1_resp', 'ppg_g_2_resp', 'ppg_ir_1_resp', 'ppg_ir_2_resp']
                elif input_name=='SCG_select':
                    list_valid = ['SCG_AMpt', 'SCGy_AMpt', 'SCGxyz_AMpt']

                indices_channel_select = []
                for channel_name in list_valid:
                    indices_channel_select.append(list_input.index(channel_name))

                print('signal indices for {} are: {}'.format(input_name, indices_channel_select) )

                fft_matrix = data_all[:,indices_channel_select, :,:] # (N_sample, 4, 20, 58)

#                 i_mid = fft_matrix.shape[-2]//2 # the frequency dimension
#                 fft_matrix_avg = fft_matrix[:,:,i_mid-1:i_mid+1,:].mean(axis=-2) # (N_sample, 4, 58)
                fft_matrix_avg = fft_matrix.mean(axis=-2) # (N_sample, 4, 58)
                fft_matrix_avg = fft_matrix_avg.transpose(0, 2, 1) # (N_sample, 58, 4)

                RQI_fft_arr = get_RQI_fft(fft_matrix_avg) # compute RQIfft on dim 1 -> (N_sample, 4)

                indices_channel = np.argmax(RQI_fft_arr, axis=-1)
                d = fft_matrix[np.arange(fft_matrix.shape[0]),indices_channel, :,:]
                d = d[:,None,:,:]
                
            else:
                d = data_all[:,[ list_input.index(input_name) ],:,:] # (N x 1 x 20 x 58)
#                 list_data.append( data_all[:,[ list_input.index(input_name) ],:,:] ) # (N x 1 x 20 x 58)
#             print(d.shape)
            list_data.append( d )    

#         sys.exit()
        data_all = np.concatenate(list_data, axis=1)
#         print(data_all.shape)

    
    else:
        indices_input = []
        for input_name in input_names:
            i_input = list_input.index(input_name)
            indices_input.append(i_input)
        data_all = data_all[:,indices_input,:,:]
    
    indices_raw = []
    raw_all = raw_all[:,indices_raw,:,:]

#     meta_all = data_loader('meta', inputdir)[:, indices_meta]

    subject_id = training_params['CV_config']['subject_id']
    task_id = training_params['CV_config']['task_id']
#     training_mode = training_params['training_mode']

#     if 'train' in set_name:
#         mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]!=task_id)
#     else:
#         mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]==task_id)

#     if ['training_mode']

#     if training_mode == 'subject_ind':


#     mask_set = np.full(data_all.shape[0], True, dtype=bool)
    mask_set = np.isin(meta_all[:,1], task_id)
    
#     mask_set = meta_all[:,1]!=5 # don't use task_id=5 (Recovery Run)

#     if (training_params['TF_type']=='pretrain') or (training_params['TF_type']=='prepare'):
    if training_params['TF_type']=='prepare':
        if 'train' in set_name:
            mask_task = np.in1d(meta_all[:,1], training_params['task_ids_train'])
            mask_set = (mask_set) & (mask_task)

#             if training_params['domain']=='CDC_dataset':
#                 # don't put 6MWT (task_id=6) in the dataset
#                 mask_task = np.in1d(meta_all[:,1], training_params['task_ids_train'])
#                 mask_set = (mask_set) & (mask_task)
#             elif training_params['domain']=='GT_dataset':
#                 pass
#                 mask_set = meta_all[:,0]>=0 # all subjects
        else:
            mask_task = np.in1d(meta_all[:,1], training_params['task_ids_val'])
            mask_set = (mask_set) & (mask_task)
#             if training_params['domain']=='CDC_dataset':
#                 mask_task = np.in1d(meta_all[:,1], training_params['task_ids_val'])
#                 mask_set = (mask_set) & (mask_task)
#             elif training_params['domain']=='GT_dataset':
#                 pass
#             mask_set = meta_all[:,0]>=0 # all subjects

    else:
        if 'train' in set_name:
            mask_set = (mask_set) & (meta_all[:,0]!=subject_id)
            # don't put 6MWT (task_id=6) in the dataset
            mask_task = np.in1d(meta_all[:,1], training_params['task_ids_train'])
            mask_set = (mask_set) & (mask_task)
#             mask_set = (mask_set) & (meta_all[:,1]!=5) &  (meta_all[:,1]!=6) & (meta_all[:,1]!=10)
#             mask_set = (mask_set) & (meta_all[:,1]!=10)
#                 include all (including walking)
        else:
            mask_set = (mask_set) & (meta_all[:,0]==subject_id)
            mask_task = np.in1d(meta_all[:,1], training_params['task_ids_val'])
            mask_set = (mask_set) & (mask_task)
            
#             mask_set = (meta_all[:,0]==subject_id) & (np.isin(meta_all[:,1], task_id))
#             mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]<6)

    if 'reject_subject_id' in training_params['CV_config']:
        # add False to mask_set for subject in reject_subject_id
        reject_subject_id = training_params['CV_config']['reject_subject_id']
#         print(reject_subject_id)
#         print(~np.isin(meta_all[:,0], reject_subject_id))
#         print(np.isin(meta_all[:,0], reject_subject_id).shape)
#         print(mask_set.shape)
        mask_set = (mask_set) & (~np.isin(meta_all[:,0], reject_subject_id))


#     elif training_mode == 'subject_specific':
#         if 'train' in set_name:
#             mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]!=task_id) # test on task_id, use the rest for training
#         else:
#             mask_set = (meta_all[:,0]==subject_id) & (meta_all[:,1]==task_id) # test on task_id

    label_set = label_all[mask_set,:,:]
    data_set = data_all[mask_set,:,:,:]
    raw_set = raw_all[mask_set,:,:,:]
    meta_set = meta_all[mask_set,:]
    ts_set = ts_all[mask_set,:]

    # print(label_all.shape, data_all.shape, raw_all.shape, meta_all.shape, ts_all.shape)
    # (1978, 20, 46) (1978, 13, 20, 46) (1978, 13, 20, 300) (1978, 2) (1978, 20)
#     return data, label, meta, ts, raw
#     print(data_set.shape, label_set.shape, meta_set.shape, ts_set.shape, raw_set.shape)
    return data_set, label_set, meta_set, ts_set, raw_set

def get_indices_ppg(data_all, meta_all, list_input):    
    # get the ppg_name selected for each subject, each task_name
    
    outputdir_df_ppg_selected =  '../../data/stage3/cardiac/df_ppg_selected.feather'
    df_ppg_selected = pd.read_feather(outputdir_df_ppg_selected)
    df_ppg_selected['subject_id'] = df_ppg_selected['subject_id'].astype(int)

    indices_ppg = -1*np.ones(data_all.shape[0]).astype(int)

    for subject_id in np.unique(meta_all[:,0]):
    #     print('==================')
        for task_id in np.unique(meta_all[:,1]):     
            
            mask = (meta_all[:,0]==subject_id) & (meta_all[:,1]==task_id)
    # mask = (meta_all[:,0]==101) & (meta_all[:,1]==tasks_dict['Baseline'])
    #         print('{}\t{}\t\t{}'.format(subject_id, tasks_dict_reversed[task_id], mask.sum()) ) 

            ppg_selected = df_ppg_selected[(df_ppg_selected['subject_id']==subject_id) & (df_ppg_selected['task_name']==tasks_dict_reversed[task_id])]['ppg_selected'].values[0]
            ppg_selected = ppg_selected
            indices_ppg[mask] = list_input.index(ppg_selected)

    return indices_ppg


def binarize_axis(im, axis=1):
    im = (im == im.max(axis=axis)[:,:,:,None]).astype(int)
    return im


# ref: https://imgaug.readthedocs.io/en/latest/source/overview/segmentation.html#superpixels
class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.OneOf([
            iaa.Identity(),
            iaa.Superpixels(p_replace=0.3, n_segments=128),
            iaa.GaussianBlur(sigma=(0, 3.0)),
#             iaa.Fliplr(1),
            iaa.SaltAndPepper(0.01, per_channel=False)
        ])

    def __call__(self, img):
        # img.shape = (N_ch, length, height)
        img = np.array(img.transpose(1, 2, 0)) # (length, height, N_ch)
        return self.aug.augment_image(img).transpose(2, 0, 1) # (batch_size, N_ch, length, height)
    



class GTPatchDataset(Dataset):
    def __init__(self, data, label, meta, raw, ts, training_params, transform=None):
        self.data = data
        self.label = label
        self.meta = meta
        self.raw = raw
        self.ts = ts
        self.transform = transform
        self.device = training_params['device']
        self.normalization = training_params['normalization']

    def __getitem__(self, index):
        data = self.data[index,:,:,:]
        label = self.label[index,:,:,:]
        meta = self.meta[index,:]
        ts = self.ts[index,:]
        raw = self.raw[index,:]
        
        if self.transform:
            data = self.transform(data)

        data = torch.from_numpy(data).to(self.device)
        label = torch.from_numpy(label).to(self.device)
        
        if self.normalization=='input_normed_minmax':
            data = ( data - data.amin(dim=(-2,-1), keepdim=True) ) / ( data.amax(dim=(-2,-1), keepdim=True) - data.amin(dim=(-2,-1), keepdim=True) ) * 256
        
#         return data, label, meta, ts, raw
        return data, label, meta, ts, raw

    def __len__(self):
        return len(self.data)
    

# class GTPatchDataset(Dataset):
#     def __init__(self, data, label, meta, training_params, transform=None):
#         device = training_params['device']
#         self.data = torch.from_numpy(data).to(device)
#         self.label = torch.from_numpy(label).to(device)
#         self.meta = meta
# #         self.transform = transform


#     def __getitem__(self, index):
#         data = self.data[index,:,:,:]
#         label = self.label[index,:,:,:]
#         meta = self.meta[index]
        
# #         data = torch.from_numpy(data)
# #         label = torch.from_numpy(label)

#         return data, label, meta

#     def __len__(self):
#         return len(self.data)
    

def get_loaders(inputdir, training_params):
#     inputdir_repCV = inputdir+'rep{}/CV{}/'.format(i_rep,i_CV)


    data_train, label_train, meta_train, ts_train, raw_train = get_samples(inputdir, training_params, set_name='train')
    data_val, label_val, meta_val, ts_val, raw_val = get_samples(inputdir, training_params, set_name='val')
    data_test, label_test, meta_test, ts_test, raw_test = get_samples(inputdir, training_params, set_name='TEST')


#     data_train = data_train[:,None,:,:]
#     data_val = data_val[:,None,:,:]
#     data_test = data_test[:,None,:,:]

    # each axis=2 can only has one 1, the rest are 0
#     label_train_bin = label_train[:,None,:,:]
#     label_val_bin = label_val[:,None,:,:]
#     label_test_bin = label_test[:,None,:,:]
#     print(label_train[:,None,:,:].shape)

    
    label_train_bin = binarize_axis(label_train[:,None,:,:], axis=-1)
    label_val_bin = binarize_axis(label_val[:,None,:,:], axis=-1)
    label_test_bin = binarize_axis(label_test[:,None,:,:], axis=-1)
    
#     print(label_train_bin.shape)

#     label_train_bin = binarize_axis(label_train, axis=-1)
#     label_val_bin = binarize_axis(label_val, axis=-1)
#     label_test_bin = binarize_axis(label_test, axis=-1)

#     label_train_bin = label_train_bin[:,None,:,:]
#     label_val_bin = label_val_bin[:,None,:,:]
#     label_test_bin = label_test_bin[:,None,:,:]


    # label_train.shape = (N, 2(0=mostly ones, 1=mostly zeors), seq_len, N_freq)
    # axis1=0: class0 means pixel belongs to white(0)
    # axis1=1: class1 means pixel belongs to black(1)
    # axis1=2: raw label pixel belongs to [0,1]
    label_train = np.concatenate((1-label_train_bin, label_train_bin, label_train[:,None,:,:]), axis=1)
    label_val = np.concatenate((1-label_val_bin, label_val_bin, label_val[:,None,:,:]), axis=1)
    label_test = np.concatenate((1-label_test_bin, label_test_bin, label_test[:,None,:,:]), axis=1)



    if not training_params.get('augmentation'):
        transforms = ImgAugTransform()
    else:
        transforms = None
    
    train_dataset = GTPatchDataset(data_train, label_train, meta_train, raw_train, ts_train, training_params, transform=transforms)
    train_eval_dataset = GTPatchDataset(data_train, label_train, meta_train, raw_train, ts_train, training_params)
    val_dataset = GTPatchDataset(data_val, label_val, meta_val, raw_val, ts_val, training_params)
#     test_dataset = GTPatchDataset(data_test, label_test, meta_test, training_params)


#     patch_datasets = {
#         'train': train_dataset, 'val': val_dataset, 'test': test_dataset
#     }
    patch_datasets = {
        'train': train_dataset, 'val': val_dataset
    }

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, num_workers=0, drop_last=True),
        'val': DataLoader(val_dataset, batch_size=20000, shuffle=False, num_workers=0),
#         'test': DataLoader(test_dataset, batch_size=20000, shuffle=False, num_workers=0),
        'train_eval': DataLoader(train_eval_dataset, batch_size=20000, shuffle=False, num_workers=0),
    }

    dataset_sizes = {
        x: len(patch_datasets[x]) for x in patch_datasets.keys()
    }
    
    return dataloaders, dataset_sizes
    

