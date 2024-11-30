import torch
import torch.nn as nn
from block import TransformerBlock
from layernorm import LayerNorm

'''
This file contains the GPT2 class, which is the transformer model.

As of Nov 30, 2024 (happy birthday, ChatGPT!):
This is currently configured for the 124M model, and is outputting logits.
In the future, it will need to have a loss function and optimizer applied to it so that it can learn.
'''


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embed = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])] # real transformer block
        )
        self.final_norm = LayerNorm(cfg['emb_dim']) # real layer norm class
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.token_embed(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits