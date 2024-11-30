import torch
import torch.nn as nn

'''
This file contains the MultiHeadAttention class, which is the multihead attention class.

Multihead attention is used in the transformer model to allow the model to jointly attend to information from different
representation subspaces at different positions. This is done by splitting the input into multiple heads, computing the
attention scores for each head, and then concatenating the results.

Some important bits about this code:

1. The mask is used to prevent the model from attending to future positions. This is an implementation of the
causal mask, which is a triangular matrix filled with -inf (infinity) values.

2. The keys, queries, and values are all linear transformations of the input. The keys and queries are used to compute the
attention scores, and the values are used to compute the context vector.

3. The attention scores are computed by taking the dot product of the queries and keys, and then scaling by the square
root of the key dimension. This is to prevent the scores from becoming too large or too small, which can cause numerical
stability issues.

4. The attention scores are then passed through a softmax function to turn them into probabilities. This is done to ensure
that the attention scores sum to 1, which is a requirement for the attention mechanism.

5. The context vector is computed by taking the attention scores, multiplying them by the values, and then concatenating
the results. This is done for each head, and the results are then concatenated and passed through a linear transformation
and a dropout layer to produce the final output, which is the context vector.

6. Context vector is the output of the multihead attention block, which is the weighted sum of the values, weighted by the
attention scores.
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out HAS TO BE divisible by num_heads for this to work"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec