import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def reset_weights(model):
    if isinstance(model, nn.Linear):
        model.reset_parameters()


class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.task_specific_layers = nn.ModuleList() # task specific layers

        self.shared_layers = nn.Sequential(nn.Linear(kwargs['input_dim'], kwargs['hidden1_dim']), 
                                           nn.ReLU(),
                                           nn.Dropout(p=kwargs['drop_rate']),
                                           nn.Linear(kwargs['hidden1_dim'], kwargs['hidden2_dim']),
                                           nn.ReLU(),
                                           nn.Dropout(p=kwargs['drop_rate']))

        for i in range(kwargs['num_tasks']):
            fc_last = nn.Linear(kwargs['hidden2_dim'], kwargs['output_dim']) # [neg=0, pos=1]
            self.task_specific_layers.append(fc_last)

    def forward(self, inputs, task_idx): 
        x = self.shared_layers(inputs)
        logits = self.task_specific_layers[task_idx](x)
        return logits

    
class SingleNet(nn.Module):
    def __init__(self, **kwargs):
        super(SingleNet, self).__init__()
        self.shared_layers = nn.Sequential(nn.Linear(kwargs['input_dim'], kwargs['hidden1_dim']), 
                                           nn.ReLU(),
                                           nn.Dropout(p=kwargs['drop_rate']),
                                           nn.Linear(kwargs['hidden1_dim'], kwargs['hidden2_dim']),
                                           nn.ReLU(), 
                                           nn.Dropout(p=kwargs['drop_rate']))

        self.fc_last = nn.Linear(kwargs['hidden2_dim'], kwargs['output_dim']) # [neg=0, pos=1]
        
    def forward(self, inputs): 
        x = self.shared_layers(inputs)
        logits = self.fc_last(x)
        return logits
    

    
