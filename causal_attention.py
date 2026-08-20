import torch.nn as nn
import torch
class CausalAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_q = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self,input_batches):
        batch_num,number_of_tokens,d_in = input_batches.shape
        keys = self.W_k(input_batches)
        queries = self.W_q(input_batches)
        values = self.W_v(input_batches)
        atten_scores = queries @ keys.transpose(1,2)
        atten_scores.masked_fill_(
            self.mask.bool()[:number_of_tokens,:number_of_tokens],
            -torch.inf
        )
        atten_weights = torch.softmax(atten_scores/keys.shape[-1]**0.5,dim=-1)
        atten_weights = self.dropout(atten_weights)
        contex_vec = atten_weights@values
        return contex_vec

