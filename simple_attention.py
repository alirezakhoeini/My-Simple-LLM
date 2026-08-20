
import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, input):
        query = self.W_q(input)
        key = self.W_k(input)
        value = self.W_v(input)

        atten_scores = query @ key.T
        atten_weights = torch.softmax(atten_scores / (key.shape[-1] ** 0.5), dim=-1)
        print("Softmax Normalized Attention weights:\n", atten_weights)
        context_vec = atten_weights @ value

        return context_vec