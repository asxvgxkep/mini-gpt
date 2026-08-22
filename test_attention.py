import torch

from model.attention import CausalSelfAttention


batch_size = 2
seq_len = 8
hidden_dim = 128


x = torch.randn(
    batch_size,
    seq_len,
    hidden_dim
)


attention = CausalSelfAttention(
    hidden_dim=hidden_dim,
    n_heads=4,
    context_length=128
)


output = attention(x)


print(output.shape)