import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

import numpy as np

from dataIO import *
from stage3_preprocess import *
from VO2_extension1111.dataset_util import *
from VO2_extension1111.models import *
import copy

import wandb



# def train_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def train_mtlnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    # device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
    device = training_params['device']

    dataset_size = len(dataloader.dataset)
    # total_loss = 0
    # training_params['regression_names']
    # total_AE = 0    
    # total_losses = dict.fromkeys(training_params['regression_names']+['total'], 0)
    # total_losses = dict.fromkeys(list(criterion.criterions.keys())+['total'], 0)

    output_names = list(criterion.criterions.keys())
    total_losses = dict.fromkeys(output_names+['total'], 0)
    # concat_feature_arr = []

    model.train()
    
    for i, (ecg, scg, ppg, feature, label, meta) in enumerate(dataloader):

        # don't train on the batch that only has one instance
        if ecg.size()[0] == 1:
            continue

        # 1. get data
        ecg = ecg.to(device=device, dtype=torch.float)
        scg = scg.to(device=device, dtype=torch.float)
        ppg = ppg.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        

        if training_params['optimizer_name'] == 'Adam':
            # 2. infer by net
            out, deep_feature, concat_feature, attention_dict = model.to(device)(ecg, scg, ppg, feature)

            losses = criterion(out, label)

            # 3. Backward and optimize
            optimizer.zero_grad()
            # print(losses['total'])
            # print(concat_feature.size())
            # print(losses['total'].size())
            losses['total'].backward()
            optimizer.step()

        elif training_params['optimizer_name'] == 'LBFGS':
            
            def closure():
                out, deep_feature, concat_feature, attention_dict = model(ecg, scg, ppg, feature)
                losses = criterion(out, label)

                # 3. Backward and optimize
                optimizer.zero_grad()
                losses['total'].backward()
                return losses['total']
            
            # Update weights
            optimizer.step(closure)

            out, deep_feature, concat_feature, attention_dict = model(ecg, scg, ppg, feature)
            losses = criterion(out, label)

        # loss = closure()

        
        
        # 4. accumulate the loss
        for loss_name in losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()


#         # 3. accumulate the loss
#         total_loss += loss.data.detach().cpu().numpy()

#         # 4. Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

        
        main_task = training_params['regression_names'][-1]
        # print(main_task, training_params['regression_names'], label)
        out = out[main_task].data.detach().cpu().numpy()
        label = label[:, [ training_params['regression_names'].index(main_task) ]].data.detach().cpu().numpy()
#         out = out['EE_cosmed'].data.detach().cpu().numpy()
#         label = label[:, [training_params['tasks'].index('EE_cosmed')]].data.detach().cpu().numpy()
        # concat_feature_arr.append(concat_feature.data.detach().cpu().numpy())


        # print(deep_feature.size(), out.shape, )
        # sys.exit()







#     total_loss = total_loss/dataset_size
    for loss_name in total_losses.keys():
        total_losses[loss_name] == total_losses[loss_name]/dataset_size

    
    if training_params['adaptive_loss_name']=='awl':
        loss_weights_dict = {}

        for i, key in enumerate(criterion.criterions.keys()):
            loss_weights_dict['weight_'+key] = criterion.awl.params.data.detach().cpu().numpy()[i]

        total_losses = merge_dict(total_losses, loss_weights_dict)

    
    subject_id = training_params['CV_config']['subject_id']
    # if training_params['wandb']==True:
    #     # W&B
    #     wandb.log({
    #         '[{}] train_loss'.format(subject_id): total_losses['total'], 
    #         # '[{}] train_MAE'.format(subject_id): MAE, 
    #         'epoch': epoch, })

    
    # # TODO: remove performance_dict
    performance_dict = {
        'total_losses': total_losses,
    }
    

    
    return performance_dict


# def eval_resnet(model, dataloader, optimizer, criterion, epoch, training_params):
def eval_mtlnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    
    # device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
    device = training_params['device']

    dataset_size = len(dataloader.dataset)

    # total_loss = 0
    # total_AE = 0
    # total_losses = dict.fromkeys(training_params['regression_names']+['total'], 0)

    output_names = list(criterion.criterions.keys())
    total_losses = dict.fromkeys(output_names+['total'], 0)
    
    # total_losses = dict.fromkeys(list(criterion.criterions.keys())+['total'], 0)

    model.eval()
#     print('\t\tswitch model to eval')

    for i, (ecg, scg, ppg, feature, label, meta) in enumerate(dataloader):
        # 1. get data        
        # data = data.to(device=device, dtype=torch.float)
        ecg = ecg.to(device=device, dtype=torch.float)
        scg = scg.to(device=device, dtype=torch.float)
        ppg = ppg.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out, deep_feature, concat_feature, attention_dict = model.to(device)(ecg, scg, ppg, feature)
#         out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(out, label)
        loss = losses['total']

        # 3. accumulate the loss
        # total_loss += loss.data.detach().cpu().numpy()
        for loss_name in losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()

        
        
        # main_task = training_params['output_names'][0]
        # out = out[main_task].data.detach().cpu().numpy()
        # label = label[:, [ training_params['output_names'].index(main_task.split('-')[0]) ]].data.detach().cpu().numpy()
        # total_AE += np.sum(np.abs(out.squeeze()-label.squeeze()).squeeze())



        main_task = training_params['regression_names'][-1]
        out = out[main_task].data.detach().cpu().numpy()
        label = label[:, [ training_params['regression_names'].index(main_task) ]].data.detach().cpu().numpy()
#         out = out['EE_cosmed'].data.detach().cpu().numpy()
#         label = label[:, [training_params['tasks'].index('EE_cosmed')]].data.detach().cpu().numpy()
            


#     total_loss = total_loss/dataset_size
    for loss_name in total_losses.keys():
        total_losses[loss_name] == total_losses[loss_name]/dataset_size


    if training_params['adaptive_loss_name']=='awl':
        loss_weights_dict = {}
        for i, key in enumerate(criterion.criterions.keys()):
            loss_weights_dict['weight_'+key] = criterion.awl.params.data.detach().cpu().numpy()[i]

        total_losses = merge_dict(total_losses, loss_weights_dict)


    subject_id = training_params['CV_config']['subject_id']
    # if training_params['wandb']==True:
    #     # W&B
    #     wandb.log({
    #         '[{}] val_loss'.format(subject_id): total_losses['total'], 
    #         'epoch': epoch, })

    
    performance_dict = {'total_losses': total_losses, 
                       }
    
    return performance_dict




# def pred_resnet(model, dataloader, criterion, epoch, training_params):
def pred_mtlnet(model, dataloader, training_params):

    optimizer = training_params['optimizer']
    criterion = training_params['criterion']
    epoch = training_params['epoch']
    regression_names = training_params['regression_names']
    
    # device = torch.device('cuda:{}'.format(int(training_params['cuda_i'])) if torch.cuda.is_available() else 'cpu')
    device = training_params['device']

    dataset_size = len(dataloader.dataset)

    # total_loss = 0
    
    output_names = list(criterion.criterions.keys())
    # total_losses = dict.fromkeys(training_params['regression_names']+['total'], 0)
    total_losses = dict.fromkeys(output_names+['total'], 0)

#     out_arr = np.empty(0)
#     label_arr = np.empty(0)
    # out_dict = {}
    # label_dict= {}
    

    out_dict = {}
    for model_out_name in output_names:
        out_dict[model_out_name] = []
    label_dict = {}
    for model_out_name in output_names:
        label_dict[model_out_name] = []

    deep_feature_dict = {}
    # for input_name in training_params['input_names']:
    for input_name in list(model.feature_extractors.keys()):
        deep_feature_dict[input_name] = []

    

#     for task in training_params['tasks']:
#         out_dict[task] = np.empty(0)
#         label_dict[task] = np.empty(0)

    meta_arr = []
    feature_arr = []
    concat_feature_arr = []
    attention_dict_arr = []
    
    model.eval()
#     print('\t\tswitch model to eval')

    for i, (ecg, scg, ppg, feature, label, meta) in enumerate(dataloader):
        # 1. get data
        ecg = ecg.to(device=device, dtype=torch.float)
        scg = scg.to(device=device, dtype=torch.float)
        ppg = ppg.to(device=device, dtype=torch.float)
        feature = feature.to(device=device, dtype=torch.float)
        label = label.to(device=device, dtype=torch.float)
#         data = data.to(device).float()
#         label = label.to(device=device, dtype=torch.float)

        # 2. infer by net
        out, deep_feature, concat_feature, attention_dict = model.to(device)(ecg, scg, ppg, feature)
#         out = model(data)
            
        # 3. loss function
#         loss = criterion(out, label)
        losses = criterion(out, label)
        loss = losses['total']
        
        # 4. compute the class loss of features
        # total_loss += loss.data.detach().cpu().numpy()
        for loss_name in total_losses.keys():
            total_losses[loss_name] += losses[loss_name].data.detach().cpu().numpy()

        # main_task = training_params['regression_names'][-1]
        # out = out[main_task].data.detach().cpu().numpy()
        # label = label[:, [ training_params['regression_names'].index(main_task) ]].data.detach().cpu().numpy()
#         out = out['EE_cosmed'].data.detach().cpu().numpy()
#         label = label[:, [training_params['tasks'].index('EE_cosmed')]].data.detach().cpu().numpy()
            

        for output_name in out_dict.keys():
            # out_dict[output_name] = np.r_[ out_dict[output_name],  out[output_name].data.detach().cpu().numpy().squeeze() ]
            # label_dict[output_name] = np.r_[ label_dict[output_name], label[:, [ training_params['regression_names'].index(output_name) ]].data.detach().cpu().numpy().squeeze() ]
            # print('out', output_name, out_dict, out)
            # print('label', output_name, label_dict, label)
            
            # if using xgb_regressor, will replace predicted output with xgb_regressor's output (using concat features)
            # if 'xgb_regressor' in training_params:
            #     out_xgb = training_params['xgb_regressor'].predict(concat_feature.data.detach().cpu().numpy())
            #     out_dict[output_name].append(out_xgb)
            # else:
            #     out_dict[output_name].append(out[output_name].detach().cpu().numpy())

            # print(output_name, out[output_name], label[:,  [output_names.index(output_name)] ])
            # print(output_name, out_dict.keys(), out.keys())
            out_dict[output_name].append(out[output_name].detach().cpu().numpy())
            
            # regression_names = ['merged-HR_patch', 'SCG-HR_patch', 'merged-RR_cosmed', 'SCG-RR_cosmed', 'VO2_cosmed']
            label_dict[output_name].append(label[:,  [regression_names.index(output_name)] ].detach().cpu().numpy())

        for input_name in deep_feature.keys():
            deep_feature_dict[input_name].append(deep_feature[input_name].data.detach().cpu().numpy())
            # print(input_name)
            # print(deep_feature)
            # print(deep_feature_dict)


        feature_arr.append( feature.detach().cpu().numpy())
        meta_arr.append( meta.detach().cpu().numpy())

        concat_feature_arr.append(concat_feature.data.detach().cpu().numpy())

        attention_dict_arr.append( attention_dict)

        
        
        

    # if training_params['regressor_type']=='CardioRespXGBRegression':
    #     label_main = dataloader.dataset.label[:, [ training_params['regression_names'].index(main_task) ]]
    #     concat_feature_arr = np.concatenate(concat_feature_arr, axis=0)
    #     training_params['xgb_regressor'].predict(concat_feature_arr)



        
        
        

#         # editted 3/19
#         for output_name in out.keys():
            
#             if i == 0:
#                 out_dict[output_name] = np.empty(0)
#                 label_dict[output_name] = np.empty(0)

            
# #             print(out, out_dict)
# #             print(out, out_dict)
# #             print(out[output_name].size(), out_dict[output_name].shape)
# #             sys.exit()

# #             out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze()]

# #             if len(out[output_name].size())<2:
#             if 'LSTM' not in training_params['model_name']:
#                 out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze()]
#             else: # if lstm, take the average of the output
#                 out_dict[output_name] = np.r_[out_dict[output_name], out[output_name].detach().cpu().numpy().squeeze().mean(axis=-1)]
    
# #             print(label.size())
# #             print(training_params['output_names'].index(output_name.split('-')[0]))
# #             print(label[:,training_params['output_names'].index(output_name.split('-')[0]) ])
# #             sys.exit()
# #             print(len(label.size()), label.size())
#             if 'LSTM' not in training_params['model_name']:
#                 label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]) ].detach().cpu().numpy().squeeze() ]
# #                 print(label_dict[output_name].shape)
#             else:
#                 label_dict[output_name] = np.r_[label_dict[output_name], label[:,training_params['output_names'].index(output_name.split('-')[0]), :].detach().cpu().numpy().squeeze().mean(axis=-1) ]

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
        # feature_arr.append( feature.detach().cpu().numpy())
        # meta_arr.append( meta.detach().cpu().numpy())


    meta_arr = np.concatenate(meta_arr,axis=0)      
    feature_arr = np.concatenate(feature_arr,axis=0)
    concat_feature_arr = np.concatenate(concat_feature_arr,axis=0)


    # print('meta_arr', meta_arr.shape)
    # print('feature_arr', feature_arr.shape)
    
    for output_name in out_dict.keys():
        out_dict[output_name] = np.concatenate(out_dict[output_name]).squeeze()
        label_dict[output_name] = np.concatenate(label_dict[output_name]).squeeze()
        # print(output_name, out_dict[output_name].shape)
        # print(output_name, label_dict[output_name].shape)

    for input_name in deep_feature_dict.keys():
        deep_feature_dict[input_name] = np.concatenate(deep_feature_dict[input_name])
        # print(input_name, deep_feature_dict[input_name].shape)
    # TODO: plot feature maps and filters
    

    df_attention = pd.DataFrame(attention_dict_arr)
    attention_dict_arr = {}
    for atn_name in df_attention.keys():
        attention_dict_arr[atn_name] = np.concatenate(df_attention[atn_name].values)

    
#     for atn_name in attention_dict_arr.keys():
#         attention_dict_arr[atn_name] = np.concatenate(attention_dict_arr[atn_name]).squeeze()
    # total_loss = total_loss/dataset_size

    for loss_name in total_losses.keys():
        total_losses[loss_name] == total_losses[loss_name]/dataset_size


    performance_dict = {'total_losses': total_losses,
                        'out_dict': out_dict,
                        'label_dict': label_dict,
                        'deep_feature_dict': deep_feature_dict,
                        'meta_arr': meta_arr,
                        'feature_arr': feature_arr,
                        'concat_feature_arr': concat_feature_arr,
                        'attention_dict_arr': attention_dict_arr
                       }
    
    return performance_dict



def train_model(model, training_params, trainer, evaler, preder):
    
    inputdir = training_params['inputdir']
    dataloaders, dataset_sizes = get_loaders(inputdir, training_params)

    feature_train = dataloaders['train'].dataset.feature
    feature_val = dataloaders['val'].dataset.feature

    # print(feature_train.shape, feature_val.shape)
    
    # total_loss_train = np.zeros(training_params['num_epochs'])
    # total_loss_val = np.zeros(training_params['num_epochs'])

#     if training_params['wandb']==True:
#         # tell wandb to watch what the model gets up to: gradients, weights, and more!
#         wandb.watch(model, log="all", log_freq=10)
    
    # total_loss = 0
    total_losses_train = dict.fromkeys(training_params['regression_names']+['total'], np.zeros(training_params['num_epochs']))
    total_losses_val = dict.fromkeys(training_params['regression_names']+['total'], np.zeros(training_params['num_epochs']))

    print('\t start training.....')

    df_losses_train = pd.DataFrame()
    df_losses_val = pd.DataFrame()

    for epoch in range(training_params['num_epochs']):
        if epoch%(math.ceil(training_params['num_epochs']/4))==0:
            print(epoch)
        training_params['epoch'] = epoch
        
        # # at the half of the training, focus all gd on main task
        # if epoch==training_params['num_epochs']//2:
        #     training_params_copy = copy.deepcopy(training_params)
        #     training_params_copy['loss_weights']['auxillary_task'] = 0
        #     criterion = MultiTaskLoss(training_params_copy)
        #     training_params['criterion'] = criterion

        ##### model training mode ####
        performance_dict_train = trainer(model, dataloaders['train'], training_params)
        df_losses_train = df_losses_train.append(  pd.DataFrame(performance_dict_train['total_losses'], index=[0]), ignore_index=True )

#         for loss_name in performance_dict_train['total_losses'].keys():
            
#             total_losses_train[loss_name][epoch] = performance_dict_train['total_losses'][loss_name]
        # print('performance_dict_train', performance_dict_train)
        # total_loss_train[epoch] = performance_dict_train['total_losses']['total']

        performance_dict_val = evaler(model, dataloaders['val'], training_params)
        df_losses_val = df_losses_val.append(  pd.DataFrame(performance_dict_val['total_losses'], index=[0]), ignore_index=True )

        # for loss_name in performance_dict_val['total_losses'].keys():
        #     total_losses_val[loss_name][epoch] = performance_dict_val['total_losses'][loss_name]
        # total_loss_val[epoch] = performance_dict_val['total_losses']['total']

    print('\t done with training.....')

    performance_dict_train = preder(model, dataloaders['train'], training_params)
    performance_dict_val = preder(model, dataloaders['val'], training_params)

    
    main_task = training_params['regression_names'][-1]
    
    # print(main_task)

    if training_params['regressor_type']=='CardioRespXGBRegression':
        # label_train = dataloaders['train'].dataset.label[:,  -1]
        label_train = performance_dict_train['label_dict'][main_task]
        # print(main_task, label_train, performance_dict_train['out_dict'][main_task])

        all_feature_train = np.concatenate([performance_dict_train['concat_feature_arr'], performance_dict_train['feature_arr'][:, training_params['n_demographic']:]], axis=-1)
        all_feature_val = np.concatenate([performance_dict_val['concat_feature_arr'], performance_dict_val['feature_arr'][:, training_params['n_demographic']:]], axis=-1)
        # print(performance_dict_train['concat_feature_arr'].shape, performance_dict_train['feature_arr'].shape)
        training_params['xgb_regressor'] = training_params['xgb_regressor'].fit(all_feature_train, label_train)        
        performance_dict_train['out_dict'][main_task] = training_params['xgb_regressor'].predict(all_feature_train)
        performance_dict_val['out_dict'][main_task] = training_params['xgb_regressor'].predict(all_feature_val)

        
        # print(main_task, label_train, performance_dict_train['out_dict'][main_task])
        
        # training_params['xgb_regressor'] = training_params['xgb_regressor'].fit(performance_dict_train['concat_feature_arr'], label_train)        
        # performance_dict_train['out_dict'][main_task] = training_params['xgb_regressor'].predict(performance_dict_train['concat_feature_arr'])
        # performance_dict_val['out_dict'][main_task] = training_params['xgb_regressor'].predict(performance_dict_val['concat_feature_arr'])
        # sys.exit()
        
    
    CV_dict = {
        'performance_dict_train': performance_dict_train,
        'df_losses_train': df_losses_train,
        # 'total_losses_train': total_losses_train,
        'performance_dict_val': performance_dict_val,
        'df_losses_val': df_losses_val,
        # 'total_losses_val': total_losses_val,
        'model': model,
        'subject_id_val': training_params['CV_config']['subject_id'], 
    }
    
    del dataloaders
    torch.cuda.empty_cache()
    
    return CV_dict


def change_output_dim(training_params):
    input_dim = training_params['data_dimensions'][1]
    output_dim = input_dim

    for i_macro in range(training_params['n_block_macro']-1):
        output_dim = np.ceil(output_dim/training_params['stride'])

    output_dim = int(output_dim)
    training_params['output_dim'] = output_dim
    return training_params



# Note this may specific to VO2 Estimation
def get_regression_names(training_params):

    # regression_names = ['merged-HR_patch', 'merged-RR_cosmed', 'VO2_cosmed']

    regression_names = []
    for aux_task in training_params['auxillary_tasks']:
        regression_names = regression_names + ['merged-'+aux_task]

    regression_names = regression_names + training_params['main_task']

    return regression_names