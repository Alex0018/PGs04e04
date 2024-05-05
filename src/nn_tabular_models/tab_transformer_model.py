'''
The TabTransformer model incorporates a unique structure that includes a class token (CLS) and positional embeddings.

Components of the TabTransformer Model:
1. Embedding Layer: Transforms input features into dense vector representations.
2. Class Token (CLS): A learnable token that aggregates information across the input sequence 
                      for use in the final prediction.
3. Positional Embeddings: Learnable vectors that represent the position of each feature 
                          within the input sequence, aiding the model in maintaining the order of input data.
4. Transformer Encoder Layers: These layers apply the self-attention mechanism to evaluate 
                               and capture the interactions among the features.
5. Fully Connected Layer: This layer converts the outputs of the transformer encoders into the final prediction.
'''

import torch
import torch.nn as nn


class TabTransformerModel(nn.Module):
    def __init__(self, in_features, hidden_dim, num_layers, num_attention_heads, dropout):
        super(TabTransformerModel, self).__init__()
        self.embedding = nn.Linear(in_features, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, in_features + 1, hidden_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 
                                       num_attention_heads, 
                                       dim_feedforward=hidden_dim*4, 
                                       dropout=dropout, 
                                       batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(hidden_dim, 1)
    

    def forward(self, x):

        x = self.embedding(x)

        cls_tokens = self.cls_token.repeat(x.size(0), 1, 1)

        x = torch.cat([cls_tokens, x.unsqueeze(1)], dim=1)

        x = x + self.pos_embedding[:, :x.size(1), :]

        x = self.transformer_encoder(x)

        x = x[:, 0, :] # is it squeeze(1) ???

        x = self.fc(x).squeeze(1)

        return x
