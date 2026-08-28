import torch

from model.transformer import TransformerBlock


def test_transformer_block_output_shape():
    block = TransformerBlock(
        hidden_dim=128,
        n_heads=4,
        context_length=128,
        dropout=0.0
    )

    x = torch.randn(
        2,
        8,
        128
    )

    output = block(x)

    assert output.shape == x.shape


def test_transformer_block_accepts_full_context_length():
    block = TransformerBlock(
        hidden_dim=64,
        n_heads=4,
        context_length=32,
        dropout=0.0
    )

    x = torch.randn(
        1,
        32,
        64
    )

    output = block(x)

    assert output.shape == (
        1,
        32,
        64
    )
