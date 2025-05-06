import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, **kwargs):
        super(Net, self).__init__()
        self.fc = nn.Sequential(nn.Linear(kwargs['input_dim'], kwargs['hidden1_dim']), 
                                nn.ReLU(), 
                                nn.Dropout(p=kwargs['drop_rate']),
                                nn.Linear(kwargs['hidden1_dim'], kwargs['hidden2_dim']),
                                nn.ReLU(),
                                nn.Dropout(p=kwargs['drop_rate']),
                                nn.Linear(kwargs['hidden2_dim'], kwargs['output_dim']))

    def forward(self, x):
        out = self.fc(x)
        return out
    


