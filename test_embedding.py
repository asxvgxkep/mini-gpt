import torch

from model.embedding import (
    TokenEmbedding,
    PositionEmbedding
)


token = torch.tensor([
    [1,2,3,4]
])


token_emb = TokenEmbedding(
    vocab_size=5000,
    hidden_dim=128
)


pos_emb = PositionEmbedding(
    context_length=128,
    hidden_dim=128
)


x = token_emb(token)

p = pos_emb(token)


print(x.shape)
print(p.shape)