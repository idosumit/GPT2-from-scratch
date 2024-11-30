import torch
import torch.nn as nn
import tiktoken

'''
This file is just for me to see the output logits (numbers before softmax) of
the model, you can run it to get a feel for the model. Not used for training.

Logits are the raw output of the model before the softmax activation function is applied.
'''

from model import GPT2
tokenizer = tiktoken.get_encoding('gpt2')
batch = []
txt1 = 'another day of waking' # sample text 1
txt2 = 'up with a privilege' # sample text 2

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(46893023)
model = GPT2(GPT_CONFIG)

out = model(batch) # get logits

print(f"\nInput batch:\n{batch}\n\nOutput shape:\n{out.shape}\n\nOut:\n{out}")
