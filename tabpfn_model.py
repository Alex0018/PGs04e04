'''
TABPFN (TabPFN is a Prior-Data Fitted Network) is an advanced deep learning model designed specifically 
for tabular data analysis. By leveraging a combination of fully connected layers, batch normalization, 
ReLU activation functions, and dropout regularization, TABPFN effectively captures complex patterns 
and relationships within the data. 

Components of the TABPFN Model:
1. Input Layer: Transforms input features into a suitable representation for the subsequent layers.
2. Hidden Layers: These layers consist of 
        - fully connected layers (Linear), 
        - batch normalization (BatchNorm1d), 
        - ReLU activation functions, and dropout regularization. 
        The number of hidden layers and their dimensions are determined through hyperparameter optimization.
3. Output Layer: Maps the learned representations from the hidden layers to the final prediction.
'''

import torch
import torch.nn as nn


class TABPFNModel(nn.Module):
    def __init__(self, 
                 in_features, 
                 hidden_dim_list, 
                 dropout):
        super(TABPFNModel, self).__init__()
        
        layers = []
        prev_dim = in_features
        for dim in hidden_dim_list:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)
    

    def forward(self, x):
        return self.model(x).squeeze(1)
    
