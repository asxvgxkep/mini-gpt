from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 5000
    context_length: int = 128

    n_layers: int = 2
    n_heads: int = 4
    hidden_dim: int = 128

    dropout: float = 0.1