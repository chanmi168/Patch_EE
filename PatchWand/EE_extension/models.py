import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys


# from EE_extension.models_CNN import *
# from EE_extension.models_CNN2 import *
from EE_extension.models_resnet import *

class RespiratoryRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, feature_dim=0):
        super(RespiratoryRegression, self).__init__()

        self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
        self.fc1 = nn.Linear(input_dim+feature_dim, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        
#         if feature is not None:
#             print(x.size(), feature.size())
        x = torch.cat((x, feature), 1)
        
#         print(x.size(), feature.size())

        out = self.bn1(x)
        out = self.relu(out)
        out = self.fc1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
#         out = torch.cat((out, feature), 1)
        out = self.fc2(out)
        
        out = self.relu(out)

        return out

# class RespiratoryRegression(nn.Module):
#     def __init__(self, num_classes=10, input_dim=50, feature_dim=0):
#         super(RespiratoryRegression, self).__init__()

#         self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
# #         self.bn2 = nn.BatchNorm1d(50)
#         self.relu = nn.ReLU()

# #         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
#         self.fc1 = nn.Linear(input_dim+feature_dim, 1)

# #         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x, feature):  
#         x = torch.cat((x, feature), 1)

#         out = self.bn1(x)
#         out = self.relu(out)
#         out = self.fc1(out)
        
# #         out = self.bn2(out)
# #         out = self.relu(out)
# # #         out = torch.cat((out, feature), 1)
# #         out = self.fc2(out)
        
#         out = self.relu(out)

#         return out



class resp_multiverse(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(resp_multiverse, self).__init__()
        
#         training_params = get_regressor_names(training_params)
        
        input_dim = training_params['data_dimensions'][1]
        self.input_names = training_params['input_names']

        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        kernel_size = training_params['kernel_size']
#         self.tasks = training_params['tasks']
#         self.class_N = training_params['n_classes'][0]
        self.n_classes = 1

#         self.tasks = training_params['tasks']
        self.output_names = training_params['output_names']
        self.fusion_type = training_params['fusion_type']
        
#         featrue_extractor = training_params['featrue_extractor']

        if training_params['model_name'] == 'FeatureExtractor_CNN':
            featrue_extractor = FeatureExtractor_CNN
        elif training_params['model_name'] == 'ResNet1D':
            featrue_extractor = ResNet1D
        elif training_params['model_name'] == 'FeatureExtractor_CNN2':
            featrue_extractor = FeatureExtractor_CNN2


        
        
        
        if self.fusion_type=='late':
            self.feature_extractors = nn.ModuleDict(
                [[input_name, featrue_extractor(training_params=training_params, input_channel=1)] for input_name in self.input_names]
            )
        elif self.fusion_type=='early':
            self.feature_extractors = nn.ModuleDict(
                [[input_name, featrue_extractor(training_params=training_params)] for input_name in ['early_fusion']]
            )
#             self.feature_extractors = nn.ModuleDict(

#              self.feature_extractors = featrue_extractor(training_params=training_params)
    
        self.N_features = len(training_params['feature_names'])
    
        feature_out_dim = 0
#         for input_name in self.input_names:
        for input_name in self.feature_extractors.keys():
            feature_out_dim += self.feature_extractors[input_name].feature_out_dim
    
#         feature_out_dim = self.feature_extractor.feature_out_dim
    
        
        # edited 3/22

        self.regressors = []
        
#         print( self.tasks)
#         main_task = self.tasks[0]
        self.main_task = self.output_names[0]
        
#         sys.exit()
        for task in self.output_names:
            if task != self.main_task:
#             if 'EE' not in task:
                for input_name in self.feature_extractors.keys():
#                     print(task+'_'+input_name)
                    self.regressors.append([task+'-'+input_name,  RespiratoryRegression(num_classes=self.n_classes, input_dim=self.feature_extractors[input_name].feature_out_dim, feature_dim=self.N_features)])
        
#         self.regressors.append(['EE_cosmed', RespiratoryRegression(num_classes=self.n_classes, input_dim=feature_out_dim, feature_dim=self.N_features )])
        self.regressors.append([self.main_task, RespiratoryRegression(num_classes=self.n_classes, input_dim=feature_out_dim, feature_dim=self.N_features )])
        
#         self.regressors = []
#         for task in self.tasks:
#             if 'EE' not in task:
#                 self.regressors.append([task+'_'+input_name,  RespiratoryRegression(num_classes=self.n_classes, input_dim=self.feature_extractors[input_name].feature_out_dim, feature_dim=self.N_features)])
        
#         out_channels = len(self.feature_extractors.keys())
#         self.final_ch_pooling = torch.nn.Conv1d(out_channels, 1, 1) # out_channels->1 channel, kernel size = 1

#         self.regressors.append(['EE_cosmed', RespiratoryRegression(num_classes=self.n_classes, input_dim=self.feature_extractors[input_name].feature_out_dim, feature_dim=self.N_features )])
        
        
        
        
        
        
        
        
        

        
        
        self.regressors = nn.ModuleDict(self.regressors)
        
#         print(self.regressors)
#         self.regressors = nn.ModuleDict(
#             [[task, RespiratoryRegression(num_classes=self.n_classes, input_dim=feature_out_dim, feature_dim=self.N_features )] for task in self.tasks]
#         )
            
            
#         self.regressors = nn.ModuleDict(
#             [[task, RespiratoryRegression(num_classes=self.n_classes, input_dim=feature_out_dim, feature_dim=self.N_features )] for task in self.tasks]
#         )
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x, feature):
#         print(x.size())


        feature_out = {}
    
#         print(self.feature_extractors)
#         sys.exit()
    
        # edited 3/22
        feature_out['concat'] = []
        for i, input_name in enumerate(self.feature_extractors.keys()):
            feature_out[input_name] = (self.feature_extractors[input_name](x[:, [i], :]))
#             feature_out['concat']
            if self.fusion_type=='late':
                feature_out['concat'].append(self.feature_extractors[input_name](x[:, [i], :]))
            elif self.fusion_type=='early':
                feature_out['concat'].append(self.feature_extractors[input_name](x))
#             .append(self.feature_extractors[input_name](x[:, [i], :]))

#         print( self.feature_extractors[input_name].feature_out_dim)
#         print(len(feature_out['concat']), feature_out['concat'][0].size())

#         feature_out['concat'] = torch.cat(feature_out['concat'], -1)
        feature_out['concat'] = torch.cat(feature_out['concat'], -1)

    
#         feature_out['concat'] = self.final_ch_pooling()
    
    
#         # reduce the number of channels to 1
# #         out = self.final_ch_pooling(out)
#         out = torch.squeeze(out, 1) # out dim = (batch_size, N_feature)


    
    
#         print(self.regressors, feature_out['concat'].size())
#         sys.exit()
        if len(feature.size())==3:
            feature = feature[:,:,0]
    
#         feature_out = torch.cat((feature_out, feature), 1)
        output = {}
        for regressor_name in self.regressors.keys():
#             if 'EE' not in regressor_name:
                
            if self.main_task not in regressor_name:
#                 print(regressor_name, regressor_name.split('-'))
                output[regressor_name] = self.regressors[regressor_name](feature_out[regressor_name.split('-')[-1]], feature)
            else:
                output[regressor_name] = self.regressors[regressor_name](feature_out['concat'], feature)
            
    
    
    
#         feature_out = []
# #         for i, input_name in enumerate(self.input_names):
#         for i, input_name in enumerate(self.feature_extractors.keys()):
# #             feature_out.append(self.feature_extractors[input_name](x[:, [i], :]))
            
#             if self.fusion_type=='late':
#                 feature_out.append(self.feature_extractors[input_name](x[:, [i], :]))
#             elif self.fusion_type=='early':
#                 feature_out.append(self.feature_extractors[input_name](x))

        
        
#         feature_out = torch.cat(feature_out, -1)
#         print(feature_out.size())

        
        
#         if self.fusion_type = 'late':
#             feature_out = []
#             for i, input_name in enumerate(self.input_names):
#                 feature_out.append(self.feature_extractors[input_name](x[:, [i], :]))

#             feature_out = torch.cat(feature_out, -1)
            
#         elif self.fusion_type = 'early':
#             feature_out = self.feature_extractors(x)

        
        
#         if len(feature.size())==3:
#             feature = feature[:,:,0]
    
# #         feature_out = torch.cat((feature_out, feature), 1)
#         output = {}
#         for task in self.tasks:
#             output[task] = self.regressors[task](feature_out, feature)
            
#         print(output, output.keys())
#         sys.exit()
        return output




class MultiTaskLoss(nn.Module):
    def __init__(self, training_params):
        super(MultiTaskLoss, self).__init__()
#         assert(set(training_params['tasks']) == set(training_params['criterions'].keys()))
#         assert(set(training_params['tasks']) == set(training_params['loss_weights'].keys()))
        self.output_names = training_params['output_names']
#         self.criterions = training_params['criterions']
#         self.loss_weights = training_params['loss_weights']
        self.device = training_params['device']
        

        
#         aaa = {'main_task': 1, 'auxillary_task': 0.5}

        self.loss_weights  = {}
        self.criterions = {}
        # main_task = 'EE_cosmed'
#         output_names = ['EE_cosmed', 'RR_cosmed', 'HR_patch']
        main_task = self.output_names[0]

#         for task in self.tasks:
        for task in training_params['regressor_names']:
            self.criterions[task] = torch.nn.MSELoss()
            if main_task in task:
                self.loss_weights[task] = training_params['loss_weights']['main_task']
            else:
                
                N_aux_tasks = len(self.output_names)-1
                if N_aux_tasks==0:
                    self.loss_weights[task] = 0
                else:
                    self.loss_weights[task] = training_params['loss_weights']['auxillary_task']/N_aux_tasks
        
        

    def forward(self, output, label):
#         print(self.tasks, output, label)
#         print(self.tasks.index( list(output.keys())[1].split('-')[0] ))
#         sys.exit()
        label = {output_name: label[:, [self.output_names.index( output_name.split('-')[0] )]] for output_name in output.keys()}
#         label = {task: label[:, [self.tasks.index(task)]] for task in self.tasks}

#         print(output, label)
#         sys.exit()
#         losses = {output_name: self.criterions[ output_name.split('-')[0] ](output[output_name].squeeze(), label[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}
        losses = {output_name: self.criterions[ output_name ](output[output_name].squeeze(), label[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}
    
#         print(losses, output, label, self.criterions, self.loss_weights)
#         sys.exit()
#         losses['total'] = torch.sum(torch.stack([self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys()]))
        losses['total'] = torch.sum(torch.stack([self.loss_weights[ output_name ] * losses[output_name] for output_name in output.keys()]))
#         print(losses)
#         sys.exit()
        return losses