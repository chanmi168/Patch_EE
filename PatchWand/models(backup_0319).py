import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys


from models_CNN import *
from models_CNN2 import *
from models_resnet import *

class RespiratoryRegression(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, feature_dim=0):
        super(RespiratoryRegression, self).__init__()

        self.bn = nn.BatchNorm1d(input_dim+feature_dim)
        self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
        self.fc1 = nn.Linear(input_dim+feature_dim, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        x = torch.cat((x, feature), 1)

        out = self.bn(x)
        out = self.relu(out)
        out = self.fc1(out)
        
#         out = torch.cat((out, feature), 1)
        out = self.fc2(out)
        
        return out

    
class resp_multiverse(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(resp_multiverse, self).__init__()
        
        
        input_dim = training_params['data_dimensions'][1]
        self.input_names = training_params['input_names']

        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
        kernel_size = training_params['kernel_size']
        self.tasks = training_params['tasks']
#         self.class_N = training_params['n_classes'][0]
        self.n_classes = 1

#         self.tasks = training_params['tasks']
        self.tasks = training_params['output_names']
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
    
        self.regressors = nn.ModuleDict(
            [[task, RespiratoryRegression(num_classes=self.n_classes, input_dim=feature_out_dim, feature_dim=self.N_features )] for task in self.tasks]
        )
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def forward(self, x, feature):
#         print(x.size())

        feature_out = []
#         for i, input_name in enumerate(self.input_names):
        for i, input_name in enumerate(self.feature_extractors.keys()):
#             feature_out.append(self.feature_extractors[input_name](x[:, [i], :]))
            
            if self.fusion_type=='late':
                feature_out.append(self.feature_extractors[input_name](x[:, [i], :]))
            elif self.fusion_type=='early':
                feature_out.append(self.feature_extractors[input_name](x))

            
        feature_out = torch.cat(feature_out, -1)
#         print(feature_out.size())

        
        
#         if self.fusion_type = 'late':
#             feature_out = []
#             for i, input_name in enumerate(self.input_names):
#                 feature_out.append(self.feature_extractors[input_name](x[:, [i], :]))

#             feature_out = torch.cat(feature_out, -1)
            
#         elif self.fusion_type = 'early':
#             feature_out = self.feature_extractors(x)

        
        
        if len(feature.size())==3:
            feature = feature[:,:,0]
    
#         feature_out = torch.cat((feature_out, feature), 1)
        output = {}
        for task in self.tasks:
            output[task] = self.regressors[task](feature_out, feature)
            
        return output




class MultiTaskLoss(nn.Module):
    def __init__(self, training_params):
        super(MultiTaskLoss, self).__init__()
        assert(set(training_params['tasks']) == set(training_params['criterions'].keys()))
        assert(set(training_params['tasks']) == set(training_params['loss_weights'].keys()))
        self.tasks = training_params['tasks']
        self.criterions = training_params['criterions']
        self.loss_weights = training_params['loss_weights']
        self.device = training_params['device']

    def forward(self, output, label):
        label = {task: label[:, [self.tasks.index(task)]] for task in self.tasks}

#         print(output, label)
#         sys.exit()
        losses = {task: self.criterions[task](output[task].squeeze(), label[task].to(device=self.device, dtype=torch.float).squeeze()) for task in self.tasks}
        losses['total'] = torch.sum(torch.stack([self.loss_weights[task] * losses[task] for task in self.tasks]))
#         print(losses)
#         sys.exit()
        return losses