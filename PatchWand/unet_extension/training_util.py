import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import math
from tqdm import trange
import numpy as np

# sys.path.append('../') # add this line so Data and data are visible in this file
# from models import *


from unet_extension.dataset_util import *
from unet_extension.evaluation_util import *
from unet_extension.models import *
# from unet_extension.training_util import *

import wandb


def train_model(model, dataloaders, training_params):
    # initialize model, loss function, optimizer
    device = training_params['device']

#     model= U_Net(in_ch=training_params['data_dimensions'][0], out_ch=2, n1 = training_params['N_channels']).to(device).float()
    # criterion = nn.CrossEntropyLoss()
    
    if training_params['regressor']=='DominantFreq_regressor':
        criterion = DiceBCERRLoss()
    else:
        criterion = DiceBCELoss()

#     tr = trange(training_params['num_epochs'], desc='', leave=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'], weight_decay=0.01)

    freeze_model(model, training_params)

    training_params['criterion'] = criterion
    training_params['optimizer'] = optimizer

    total_loss_train = np.zeros(training_params['num_epochs'])
    total_loss_val = np.zeros(training_params['num_epochs'])

    if training_params['wandb']==True:
        # tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, criterion, log="all", log_freq=5)
    
    for epoch in range(training_params['num_epochs']):
        ##### model training mode ####
        performance_dict_train = train_unet(model, dataloaders['train'], training_params)
        
#         performance_dict_train = eval_unet(model, dataloaders['train_val'], training_params)
        total_loss_train[epoch] = performance_dict_train['total_loss']

        performance_dict_val = eval_unet(model, dataloaders['val'], training_params)
        total_loss_val[epoch] = performance_dict_val['total_loss']
        
        
        if training_params['wandb']==True:
            # W&B
            wandb.log({'train_loss': performance_dict_train['total_loss'], 'val_loss': performance_dict_val['total_loss']})


    total_loss_train = total_loss_train / len(dataloaders['train'])
    total_loss_val = total_loss_val / len(dataloaders['val'])
    
    return model, total_loss_train, total_loss_val



def train_unet(model, dataloader, training_params):
    device = training_params['device']
    criterion = training_params['criterion']
    optimizer = training_params['optimizer']
    flipper = training_params['flipper']
    total_loss = 0

    model.train()
    
#     data, label, meta, ts, raw
    for i, (data, label, meta, _, _) in enumerate(dataloader):
        # 1. get data
        data = data.to(device).float()
        meta = meta.to(device).float()
        label = label.to(device=device, dtype=torch.float)[:,0:2,:,:]
    
        if flipper:
            data_flipped = torch.flip(data, [3])
            data = torch.cat((data, data_flipped), 3)
            label_flipped = torch.flip(label, [3])
            label = torch.cat((label, label_flipped), 3)

        # 2. infer by net
#         out, RR_est = model(data)
        out = model(data)
    
#         print(out.size(), label.size())
#         print(out, label)
#         sys.exit()

        if training_params['regressor'] == 'DominantFreq_regressor':
            RR_label = meta[:, training_params['i_RR']]
            losses = criterion(out, label, RR_est, RR_label)
        else:
            losses = criterion(out, label)
#             print(out, label, losses)
#         print(loss)

        # 3. compute the class loss of features
        total_loss += losses['total'].data.detach().cpu().numpy()

        # 4. Backward and optimize
        optimizer.zero_grad()
        losses['total'].backward()
        if training_params['grad_clip']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()


    performance_dict = {'total_loss': total_loss,
                       }
    return performance_dict

def eval_unet(model, dataloader, training_params):
    device = training_params['device']
    criterion = training_params['criterion']
    flipper = training_params['flipper']

    total_loss = 0

    model.eval()
#     for i, (data, label, meta, _) in enumerate(dataloader):
    for i, (data, label, meta, _, _) in enumerate(dataloader):
        # 1. get data
        data = data.to(device).float()
        meta = meta.to(device).float()
        label = label.to(device=device, dtype=torch.float)[:,0:2,:,:]

        if flipper:
            data_flipped = torch.flip(data, [3])
            data = torch.cat((data, data_flipped), 3)
            label_flipped = torch.flip(label, [3])
            label = torch.cat((label, label_flipped), 3)
        
        # 2. infer by net
#         out, RR_est = model(data)
        out = model(data)
            
        if training_params['regressor'] == 'DominantFreq_regressor':
            RR_label = meta[:, training_params['i_RR']]
            losses = criterion(out, label, RR_est, RR_label)
        else:
            losses = criterion(out, label)

        # 3. compute the class loss of features
        total_loss += losses['total'].data.detach().cpu().numpy()
        
    
    
#     outputdir = training_params['outputdir']
#     # Save the model in the exchangeable ONNX format
#     torch.onnx.export(model, data,outputdir+ 'model.onnx', opset_version=11)
#     wandb.save(outputdir+'model.onnx')
    
    
#     model_out_dict = get_model_out(model, dataloader, training_params)
#     RR_label = model_out_dict['RR_label']
#     RR_model = model_out_dict['RR_model']
#     MAE_mean_model, MAE_std_model = get_MAE(RR_label, RR_model)

        
    performance_dict = {'total_loss': total_loss,
#                         'MAE_mean': MAE_mean
                       }
    return performance_dict


def freeze_model(model, training_params, debug=False):

    if training_params['TF_type']=='FT_top':

        for param in model.parameters():
            param.requires_grad = False

        model.Conv1.conv[0].weight.requires_grad = True
        model.Conv1.conv[0].bias.requires_grad = True
        # model.Conv1.conv[0]
        if debug:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)

    elif training_params['TF_type']=='FT_top2':

        for param in model.parameters():
            param.requires_grad = False

        model.Conv1.conv[0].weight.requires_grad = True
        model.Conv1.conv[0].bias.requires_grad = True
        model.Conv1.conv[3].weight.requires_grad = True
        model.Conv1.conv[3].bias.requires_grad = True
        # model.Conv1.conv[0]
        
        
        if debug:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(name)
                
    return model