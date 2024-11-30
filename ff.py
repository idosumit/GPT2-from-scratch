import torch
import torch.nn as nn
from gelu import GELU

'''
This file contains the FeedForward class, which is the feed forward block of the
transformer block. Feed forward classes are important for the transformer model
because they allow the model to learn non-linear relationships between inputs and
outputs. As a whole, this helps the model to generalize better and enrich the
representations of the data.
'''

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']), # linear layer
            GELU(), # gelu activation
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']), # linear layer
        )
    
    def forward(self, x):
        return self.layers(x)