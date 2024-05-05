'''
SAINT (Self-Attention and Intersample Attention Transformer) is an advanced deep learning model 
that integrates self-attention and intersample attention mechanisms to effectively capture 
intricate relationships within tabular data.

Components of the SAINT Model:
1. Input Layer: Transforms input features into a hidden representation, preparing them for subsequent processing.
2. Transformer Layers: These layers incorporate self-attention and intersample attention mechanisms 
                       to capture relationships between features and samples, enabling the model 
                       to discern complex patterns within the data.
3. Output Layer: Maps the learned representations to the final prediction.
'''

import torch
import torch.nn as nn


class SAINTModel(nn.Module):
    def __init__(self, in_features, hidden_dim, num_attention_heads, num_layers, dropout):
        super(SAINTModel, self).__init__()
        self.input_dim = in_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_attention_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer( hidden_dim, 
                                        num_attention_heads, 
                                        hidden_dim, 
                                        dropout) 
                    for _ in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, 1)
    


    def forward(self, x):
       
        x = self.embedding(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        output = self.fc(x)
        
        return output.squeeze(1)

