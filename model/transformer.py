import torch.nn as nn

from .attention import CausalSelfAttention
from .mlp import FeedForward



class TransformerBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()


        self.ln1 = nn.LayerNorm(
            hidden_dim
        )


        self.attention = CausalSelfAttention(
            hidden_dim,
            n_heads,
            context_length,
            dropout
        )


        self.ln2 = nn.LayerNorm(
            hidden_dim
        )


        self.ffn = FeedForward(
            hidden_dim
        )


        self.dropout = nn.Dropout(
            dropout
        )


    def forward(self, x):

        x = x + self.attention(
            self.ln1(x)
        )


        x = x + self.ffn(
            self.ln2(x)
        )


        return self.dropout(x)