import torch

from model.transformer import TransformerBlock


x = torch.randn(
    2,
    8,
    128
)


block = TransformerBlock(
    hidden_dim=128,
    n_heads=4,
    context_length=128
)


out = block(x)


print(out.shape)