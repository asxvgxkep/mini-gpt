import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        n_heads,
        context_length,
        dropout=0.1
    ):
        super().__init__()

        assert hidden_dim % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.qkv = nn.Linear(
            hidden_dim,
            hidden_dim * 3
        )

        self.proj = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(
                    context_length,
                    context_length
                )
            )
        )


    def forward(self, x):

        batch, seq_len, hidden = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)


        q = q.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1,2)


        k = k.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1,2)


        v = v.view(
            batch,
            seq_len,
            self.n_heads,
            self.head_dim
        ).transpose(1,2)


        attention = (
            q @ k.transpose(-2,-1)
        )

        attention = attention / (
            self.head_dim ** 0.5
        )


        mask = self.mask[:seq_len,:seq_len]


        attention = attention.masked_fill(
            mask == 0,
            float("-inf")
        )


        attention = torch.softmax(
            attention,
            dim=-1
        )


        attention = self.dropout(attention)


        output = attention @ v


        output = output.transpose(1,2)

        output = output.contiguous().view(
            batch,
            seq_len,
            hidden
        )


        return self.proj(output)