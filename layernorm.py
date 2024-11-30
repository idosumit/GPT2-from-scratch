import torch
import torch.nn as nn

'''
This file contains the LayerNorm class, which is the layer normalization class.
Layer normalization is used in the transformer model to normalize the input to
the transformer block. Specifically speaking, it normalizes the input to have an
uniform mean and variance so that it streamlines the training and evaluation
process.
'''

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # epsilon, added to the variance to prevent division by zero during normalization
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False) # False as in dividing by `n` and not `n-1`
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift