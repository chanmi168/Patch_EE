import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import numpy as np

from dataIO import *
from stage3_preprocess import *
from EE_extension.dataset_util import *

import wandb

# def train_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def train_resnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)
    total_loss = 0
    
    total_AE = 0

    model.train()
    
    for i, (data, feature, label, meta) in enumerate(dataloader):
#         print('epoch', i)
        # 1. get data
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)
        
#         data = data.to(device).float()
# #         label = label.to(device=device, dtype=torch.float)
#         label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out = model(data, feature)
#         out = model(data)
        
        # 3. loss function
#         print(out.squeeze().size(), label.size())

#         print(out, label)
#         sys.exit()
#         print(out, label, training_params['regressor_names'])

        losses = criterion(out, label)
        loss = losses['total']
        


        # 3. accumulate the loss
        total_loss += loss.data.detach().cpu().numpy()

        # 4. Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        main_task = training_params['output_names'][0]
        out = out[main_task].data.detach().cpu().numpy()
        label = label[:, [training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
#         out = out['EE_cosmed'].data.detach().cpu().numpy()
#         label = label[:, [training_params['tasks'].index('EE_cosmed')]].data.detach().cpu().numpy()
            
#         print(out, label, main_task)
#         sys.exit()

    
#         total_AE += np.sum(np.abs(out-label).squeeze())
        total_AE += np.sum(np.abs(out.squeeze()-label.squeeze()).squeeze())

#         label = {task: label[:, [self.tasks.index(task)]] for task in self.tasks}

#         print(out.shape, label.shape, out-label)
#         sys.exit()

    total_loss = total_loss/dataset_size
    MAE = total_AE/dataset_size
    
    subject_id = training_params['CV_config']['subject_id']
    if training_params['wandb']==True:
        # W&B
        wandb.log({
#             '[{}] train_loss'.format(subject_id): total_loss, 
            '[{}] train_MAE'.format(subject_id): MAE, 
            'epoch': epoch, })
    
    
    # # TODO: remove performance_dict
    performance_dict = {
        'total_loss': total_loss,
    }
    return performance_dict


# def eval_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def eval_resnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

    total_loss = 0
    total_AE = 0
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, feature, label, meta) in enumerate(dataloader):
        # 1. get data        
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out = model(data, feature)
#         out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(out, label)
        loss = losses['total']

        # 3. accumulate the loss
        total_loss += loss.data.detach().cpu().numpy()
        
        main_task = training_params['output_names'][0]
        out = out[main_task].data.detach().cpu().numpy()
        label = label[:, [ training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
#         out = out['EE_cosmed'].data.detach().cpu().numpy()
#         label = label[:, [training_params['tasks'].index('EE_cosmed')]].data.detach().cpu().numpy()
        total_AE += np.sum(np.abs(out.squeeze()-label.squeeze()).squeeze())
    
#         print((out-label).shape)
#         print(out.shape)
#         print(label.shape)

#         sys.exit()
        
    total_loss = total_loss/dataset_size
    MAE = total_AE/dataset_size



    subject_id = training_params['CV_config']['subject_id']
    if training_params['wandb']==True:
        # W&B

        wandb.log({
#             '[{}] val_loss'.format(subject_id): total_loss, 
            '[{}] val_MAE'.format(subject_id): MAE, 
            'epoch': epoch, })
#     if training_params['wandb']==True:
#         # W&B
#         wandb.log({'val_loss': total_loss, 
#                    'val_MAE': MAE, 
#                    'epoch': epoch, })
    
    performance_dict = {'total_loss': total_loss,
                       }
    
    return performance_dict




# def pred_resnet(model, dataloader, criterion, epoch, training_params):
def pred_resnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')

    dataset_size = len(dataloader.dataset)

    total_loss = 0
    
#     out_arr = np.empty(0)
#     label_arr = np.empty(0)
    out_dict = {}
    label_dict= {}
    
#     for task in training_params['tasks']:
#         out_dict[task] = np.empty(0)
#         label_dict[task] = np.empty(0)

    feature_arr = []
    meta_arr = []
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (data, feature, label, meta) in enumerate(dataloader):
        # 1. get data
        data = data.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)
#         data = data.to(device).float()
#         label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out = model(data, feature)
#         out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(out, label)
        loss = losses['total']
        
        # 4. compute the class loss of features
        total_loss += loss.data.detach().cpu().numpy()


        # editted 3/19
        for output_name in out.keys():
            
            if i == 0:
                out_dict[output_name] = np.empty(0)
                label_dict[output_name] = np.empty(0)

            
#             print(out, out_dict)
#             print(out, out_dict)
#             print(out[output_name].size(), out_dict[output_name].shape)
#             sys.exit()

#             out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze()]

#             if len(out[output_name].size())<2:
            if 'LSTM' not in training_params['model_name']:
                out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze()]
            else: # if lstm, take the average of the output
                out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze().mean(axis=-1)]
    
#             print(label.size())
#             print(training_params['output_names'].index(output_name.split('-')[0]))
#             print(label[:,training_params['output_names'].index(output_name.split('-')[0]) ])
#             sys.exit()
#             print(len(label.size()), label.size())
            if 'LSTM' not in training_params['model_name']:
                label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy().squeeze() ]
#                 print(label_dict[output_name].shape)
            else:
                label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]), :].detach().cpu().numpy().squeeze().mean(axis=-1) ]

#             label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['tasks'].index(output_name.split('-')[0])].detach().cpu().numpy().squeeze() ]
#             label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['tasks'].index('_'.join(output_name.split('_')[:2]))].detach().cpu().numpy().squeeze()]
        
        
    
#         for task in training_params['tasks']:
# #             print( out[task].detach().cpu().numpy().shape)
# #             print(label[:,training_params['tasks'].index(task)].detach().cpu().numpy().shape)
#             out_dict[task] = np.r_[out_dict[task], out[task].detach().cpu().numpy().squeeze()]
#             label_dict[task] = np.r_[label_dict[task], label[:,training_params['tasks'].index(task)].detach().cpu().numpy().squeeze()]

            
            
            
#         out_arr = np.r_[out_arr, out]
# #         sys.exit()
#         label_arr = np.r_[label_arr, label]
    
#         print(meta.shape)
        feature_arr.append( feature.detach().cpu().numpy())
        meta_arr.append( meta.detach().cpu().numpy())


    meta_arr = np.concatenate(meta_arr,axis=0)
        
    # TODO: plot feature maps and filters
        
    total_loss = total_loss/dataset_size
    performance_dict = {'total_loss': total_loss,
                        'out_dict': out_dict,
                        'label_dict': label_dict,
                        'meta_arr': meta_arr,
                        'feature_arr': feature_arr,
                       }
    
    return performance_dict



def train_model(model, training_params, trainer, evaler, preder):

    inputdir = training_params['inputdir']
    
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    total_loss_train = np.zeros(training_params['num_epochs'])
    total_loss_val = np.zeros(training_params['num_epochs'])

#     if training_params['wandb']==True:
#         # tell wandb to watch what the model gets up to: gradients, weights, and more!
#         wandb.watch(model, log="all", log_freq=10)
    
    
    print('\t start training.....')

    for epoch in range(training_params['num_epochs']):
        if epoch%10==1:
            print(epoch)
        training_params['epoch'] = epoch

        ##### model training mode ####
        performance_dict_train = trainer(model, dataloaders['train'], training_params)
        total_loss_train[epoch] = performance_dict_train['total_loss']

        performance_dict_val = evaler(model, dataloaders['val'], training_params)
        total_loss_val[epoch] = performance_dict_val['total_loss']

    print('\t done with training.....')

    performance_dict_train = preder(model, dataloaders['train'], training_params)
    performance_dict_val = preder(model, dataloaders['val'], training_params)

    
    CV_dict = {
        'performance_dict_train': performance_dict_train,
        'total_loss_train': total_loss_train,
        'performance_dict_val': performance_dict_val,
        'total_loss_val': total_loss_val,
        'model': model,
        'subject_id_val': training_params['CV_config']['subject_id'], 
    }
    
    return CV_dict


def change_output_dim(training_params):
    input_dim = training_params['data_dimensions'][1]
    output_dim = input_dim

    for i_macro in range(training_params['n_block_macro']-1):
        output_dim = np.ceil(output_dim/training_params['stride'])

    output_dim = int(output_dim)
    training_params['output_dim'] = output_dim
    return training_params

