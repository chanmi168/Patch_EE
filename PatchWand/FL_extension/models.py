import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import sys


# from models_CNN import *
# from models_CNN2 import *
# from models_resnet import *
from FL_extension.models_CNNlight import *

# high level arch
class resp_DANN(nn.Module):
    def __init__(self, class_N=1, training_params=None):
        super(resp_DANN, self).__init__()
        
        input_dim = training_params['data_dimensions'][1]
        self.input_names = training_params['input_names']

        input_channel = training_params['data_dimensions'][0]
        channel_n = training_params['channel_n']
#         kernel_size = training_params['kernel_size']
        self.n_classes = 1

        self.output_names = training_params['output_names']
        self.fusion_type = training_params['fusion_type']
        
        feature_extractor = training_params['feature_extractor']
        regressor = training_params['regressor']
        
        
        if self.fusion_type=='late':
            self.feature_extractors = nn.ModuleDict(
                [[input_name, feature_extractor(training_params=training_params, input_channel=1)] for input_name in self.input_names]
            )
        elif self.fusion_type=='early':
            self.feature_extractors = nn.ModuleDict(
                [[input_name, feature_extractor(training_params=training_params)] for input_name in ['early_fusion']]
            )

        self.N_features = len(training_params['feature_names'])

        self.dummy_param = nn.Parameter(torch.empty(0))

    
        feature_out_dim = 0
#         for input_name in self.input_names:
        for input_name in self.feature_extractors.keys():
#             feature_out_dim += self.feature_extractors[input_name].feature_out_dim
            feature_out_dim = self.feature_extractors[input_name].feature_out_dim

        # edited 3/22

        self.regressors = []
        
#         print( self.tasks)
#         main_task = self.tasks[0]
        self.main_task = self.output_names[0]
        
#         sys.exit()
        for task in self.output_names:
            if task != self.main_task:
                for input_name in self.feature_extractors.keys():
                    self.regressors.append([task+'-'+input_name,  regressor(training_params, num_classes=self.n_classes, input_dim=self.feature_extractors[input_name].feature_out_dim)])
        
        self.regressors.append([self.main_task, regressor(training_params, num_classes=self.n_classes, input_dim=feature_out_dim )])
        
        self.regressors = nn.ModuleDict(self.regressors)
        
        self.domain_classifier = DomainClassifier(num_classes=input_channel, input_dim=feature_out_dim)

        self.adversarial_weight = training_params['adversarial_weight']
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
        
    def forward(self, x, feature):
        
        if len(feature.size())==3:
            feature = feature[:,:,0]

#         print(x.size(), feature.size())

        output = {}
        feature_out = {}
#         device = self.dummy_param.device
        
        for i, input_name in enumerate(self.feature_extractors.keys()):
            feature_out[input_name] = (self.feature_extractors[input_name](x[:, [i], :]))

            for regressor_name in self.regressors.keys():
                output[regressor_name+'-{}'.format(input_name)] = self.regressors[regressor_name](feature_out[input_name], feature)

    
            output['domain-{}'.format(input_name)] = self.domain_classifier(feature_out[input_name], self.adversarial_weight)


        return output, feature_out
    
    
class RespiratoryRegression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, feature_dim=0):
        super(RespiratoryRegression, self).__init__()

#         self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
        self.fc1 = nn.Linear(input_dim+feature_dim, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        
#         if feature is not None:
#             print(x.size(), feature.size())
#         x = torch.cat((x, feature), 1)
        
#         print(x.size(), feature.size())

#         out = self.bn1(x)
        out = self.relu(x)
        out = self.fc1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
#         out = torch.cat((out, feature), 1)
        out = self.fc2(out)
        
#         out = self.relu(out)

        return out

class FFTRegression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, feature_dim=0):
        super(FFTRegression, self).__init__()

#         self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
        self.bn2 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()

#         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
        self.fc1 = nn.Linear(input_dim+feature_dim, 50)

        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
        
#         if feature is not None:
#             print(x.size(), feature.size())
#         x = torch.cat((x, feature), 1)
        
#         print(x.size(), feature.size())

#         out = self.bn1(x)
        out = self.relu(x)
        out = self.fc1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
#         out = torch.cat((out, feature), 1)
        out = self.fc2(out)
        
#         out = self.relu(out)

        return out
    
class DominantFreqRegression(nn.Module):
    def __init__(self, training_params, num_classes=10, input_dim=50, feature_dim=0):
        super(DominantFreqRegression, self).__init__()

    
#         self.xf = training_params['xf']
#         self.xf_masked = training_params['xf_masked']
        xf_masked = torch.from_numpy(training_params['xf_masked'])
        self.xf_masked = xf_masked.float()
        
        self.dominantFreq_detect = training_params['dominantFreq_detect']
#         self.xf_masked = xf_masked.to(device=training_params['device'], dtype=torch.float)
        
#         self.bn1 = nn.BatchNorm1d(input_dim+feature_dim)
#         self.bn2 = nn.BatchNorm1d(50)
#         self.relu = nn.ReLU()

# #         self.fc1 = nn.Linear(input_dim, 50+feature_dim)
#         self.fc1 = nn.Linear(input_dim+feature_dim, 50)

#         self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x, feature):  
#         out = out / torch.sum(out, axis=1)[:,None]
#         x = x / torch.sum(x, axis=1)[:,None]

        out = torch.sum(x * self.xf_masked.to(x.device), axis=1)[:,None]

    
#         # argmax is to be deprecated
#         if self.dominantFreq_detect=='argmax':
#             index_dominant = torch.argmax(x,axis=1).squeeze()
# #             print('index_dominant', index_dominant.size())
# #             print('self.xf_masked', self.xf_masked.size())
#             xf_repeated = torch.tile(self.xf_masked.to(x.device), (x.shape[0], 1)).T
# #             print('xf_repeated', xf_repeated.size())
#             out = xf_repeated[index_dominant,  range(xf_repeated.shape[1])][:,None]

    
#         elif self.dominantFreq_detect=='expectation':
# #             x_normed = x / torch.sum(x, axis=1)[:,None]
# #             out = torch.sum(x_normed * self.xf_masked.to(x.device), axis=1)[:,None]
#             out = torch.sum(x * self.xf_masked.to(x.device), axis=1)[:,None]

        return out
    
# GRL
class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


# domain classifier neural network (fc layers)
class DomainClassifier(nn.Module):
    def __init__(self, num_classes=10, input_dim=50, hidden_dim=50, p_dropout=0.5):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(p=p_dropout)
        self.relu = nn.ReLU()
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #       print('DomainClassifier_total_params:', pytorch_total_params)

    def forward(self, x, constant):
        out1 = GradReverse.grad_reverse(x.float(), constant)
#         out1 = self.relu(self.fc1(out1))
        out1 = self.relu(self.drop(self.fc1(out1)))
        out2 = self.fc2(out1)
        return out2

    
    
# loss function
class AdversarialLoss(nn.Module):
    def __init__(self, training_params):
        super(AdversarialLoss, self).__init__()
        self.output_names = training_params['output_names']
        self.device = training_params['device']

        self.loss_weights  = {}
        self.criterions = {}

        main_task = self.output_names[0]

#         for task in self.tasks:
        for task in training_params['regressor_names']:
        # for task in training_params['model_out_names']:
#             print(task, main_task)
#             sys.exit()
            self.criterions[task] = torch.nn.MSELoss()
            if main_task in task:
                self.loss_weights[task] = training_params['loss_weights']['main_task']
#                 print(self.loss_weights[task] )
            else:
#                 print(self.loss_weights[task] )

                N_aux_tasks = len(self.output_names)-1
                if N_aux_tasks==0:
                    self.loss_weights[task] = 0
                else:
                    self.loss_weights[task] = training_params['loss_weights']['auxillary_task']/N_aux_tasks
            

        
        task = 'domain'
        self.criterions[task] = torch.nn.CrossEntropyLoss()
        self.loss_weights[task] = training_params['loss_weights']['auxillary_task']
        self.modality_dict = training_params['modality_dict']
        
        
#         print(self.loss_weights)

#         print(self.modality_dict)
        
    def forward(self, output, label):
        

#         label = {output_name: label[:, [self.output_names.index( output_name.split('-')[0] )]] for output_name in output.keys()}

        label_dict = {}
        for output_name in output.keys():
            if 'domain' in output_name:
                input_name = output_name.split('-')[1]
                label_dict[output_name] = torch.ones(label.size()[0]).to(self.device) * self.modality_dict[input_name]
            else:
#                 print(output_name)
                label_dict[output_name] = label[:, [self.output_names.index( output_name.split('-')[0] )]]

#         for input_name in self.modality_dict:
#             output['domain-'+input_name] = torch.ones(label.size()[0]).to(self.device).float() * self.modality_dict[input_name]

#         print(output)

        losses = {}
        for output_name in output.keys():


            if 'domain' in output_name:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float).long()).squeeze().float()            
            else:
                losses[output_name] = self.criterions[ output_name.split('-')[0] ](output[output_name], label_dict[output_name].to(device=self.device, dtype=torch.float)).squeeze().float()
    
        
    
        list_loss = []
        
        for output_name in output.keys():
#             print(output_name)
#             print(losses[output_name] )
            
            l = self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] 
            list_loss.append(l)
            
#         sys.exit()
#             list_loss.append(self.loss_weights[ output_name.split('-')[0] ] * losses[output_name] for output_name in output.keys())

        losses['total'] = torch.sum(torch.stack(list_loss))

#         losses = {output_name: self.criterions[ output_name ](output[output_name].squeeze(), label_dict[output_name].to(device=self.device, dtype=torch.float).squeeze()) for output_name in output.keys()}


        return losses
