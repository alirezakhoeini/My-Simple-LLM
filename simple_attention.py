
import torch


class SimpleAttention:
    def __init__(self, d_in, d_out):
        self.W_q = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_k = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_v = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, input):
        query = input @ self.W_q
        key = input @ self.W_k
        value = input @ self.W_v

        atten_scores = query @ key.T
        atten_weights = torch.softmax(atten_scores / (key.shape[-1] ** 0.5), dim=-1)
        print("Softmax Normalized Attention weights:\n", atten_weights)
        context_vec = atten_weights @ value

        return context_vec