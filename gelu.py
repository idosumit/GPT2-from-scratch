import torch
import torch.nn as nn

'''
This file contains the GELU class, which is the GELU activation function.
GELU is a smooth approximation to the RELU function and is used in the feed forward block of the transformer model.
It works better than RELU in the transformer model because it is smoother and allows for better gradient flow.
RELU can sometimes "kill" the gradient when the values are very close to 0, causing dead neurons because RELU is
always 0 when the value is less than 0. GELU is a smooth approximation to RELU and does not have this problem.
'''

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))
        ))