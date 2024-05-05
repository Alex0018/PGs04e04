'''
The AutoInt (Automatic Feature Interaction Learning) model is a sophisticated deep learning approach 
tailored for tabular data, which excels in automatically learning feature interactions. 
By integrating self-attention mechanisms with residual connections, AutoInt efficiently captures high-order feature interactions.


Components of the AutoInt Model:
1. Embedding Layer: Transforms the input features into dense vector representations, preparing them for deeper analysis.
2. Multi-head Self-Attention Layers: These layers employ self-attention mechanisms to explore feature interactions 
                                     at multiple scales, using several attention heads to capture a diverse range 
                                     of interaction patterns.
3. Residual Connections: Enhance the flow of information and gradients through the network, 
                         facilitating the training of deeper models by learning additional residual information.
4. Layer Normalization and Activation Functions: Stabilize the neural network's training by normalizing layer outputs 
                        and introducing non-linearities, enhancing the model's ability to learn complex patterns.
5. Fully Connected Layers: Convert the processed feature interactions into the final prediction.
'''

import torch
import torch.nn as nn


class AutoIntModel(nn.Module):

    def __init__(self, in_features, 
                 embedding_dim, 
                 num_heads, 
                 num_layers, 
                 dropout, 
                 use_residual=True, 
                 use_normalization=True, 
                 use_activation=True):
        super(AutoIntModel, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.Dropout(dropout)
        )
        self.use_residual = use_residual
        self.use_normalization = use_normalization
        self.use_activation = use_activation
        
        activation = nn.GELU() if use_activation else nn.Identity()
        normalization = nn.LayerNorm(embedding_dim) if use_normalization else nn.Identity()
        
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True),
                normalization,
                activation,
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1)
        )
    

    def forward(self, x):

        x = self.embedding(x)
        
        for attention_layer in self.attention_layers:
            
            residual = x
            x, _ = attention_layer[0](x, x, x)
            x = attention_layer[1:](x)
            
            if self.use_residual:
                x = x + residual
        
        x = self.fc(x).squeeze(1)
        return x
