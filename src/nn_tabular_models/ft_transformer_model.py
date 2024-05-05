# NB read: where is the Fourier???


'''
The FT-Transformer (Fourier Transform Transformer) model is a variant of the Transformer architecture 
that incorporates Fourier transforms to enhance its capability to capture long-range dependencies in the input data. 

Components:
1. Embedding Layer: Converts the input features into a dense representation 
                    that is more suitable for processing by neural networks.
2. Transformer Encoder Layers: Utilizes the self-attention mechanism to effectively capture 
                               long-range dependencies within the input sequence.
3. Fully Connected Layer: Transforms the outputs from the transformer encoder 
                          into the final prediction.
'''

import torch
import torch.nn as nn


class FTTransformerModel(nn.Module):

    def __init__(self, in_features, hidden_dim, num_layers, num_attention_heads, dropout):
        super(FTTransformerModel, self).__init__()
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 
                                       num_attention_heads, 
                                       dim_feedforward=hidden_dim*4,  # ???
                                       dropout=dropout),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)
    

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x.unsqueeze(1)).squeeze(1)
        x = self.fc(x).squeeze(1)
        return x
