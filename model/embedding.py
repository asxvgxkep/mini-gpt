import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

    def forward(self, x):
        return self.embedding(x)


class PositionEmbedding(nn.Module):
    def __init__(self, context_length, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            context_length,
            hidden_dim
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        return self.embedding(positions)