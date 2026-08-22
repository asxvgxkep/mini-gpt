import torch.nn as nn

from .embedding import (
    TokenEmbedding,
    PositionEmbedding
)

from .transformer import TransformerBlock


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()


        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.hidden_dim
        )


        self.position_embedding = PositionEmbedding(
            config.context_length,
            config.hidden_dim
        )


        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.hidden_dim,
                    config.n_heads,
                    config.context_length,
                    config.dropout
                )
                for _ in range(config.n_layers)
            ]
        )


        self.ln = nn.LayerNorm(
            config.hidden_dim
        )


        self.lm_head = nn.Linear(
            config.hidden_dim,
            config.vocab_size,
            bias=False
        )


    def forward(self, x):

        token_emb = self.token_embedding(x)

        pos_emb = self.position_embedding(x)


        x = token_emb + pos_emb


        for block in self.blocks:
            x = block(x)


        x = self.ln(x)


        logits = self.lm_head(x)


        return logits