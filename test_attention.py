import torch

from model.attention import CausalSelfAttention


def test_attention_output_shape():
    attention = CausalSelfAttention(
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

    output = attention(x)

    assert output.shape == x.shape


def test_attention_is_causal():
    torch.manual_seed(0)

    attention = CausalSelfAttention(
        hidden_dim=32,
        n_heads=4,
        context_length=16,
        dropout=0.0
    )

    attention.eval()

    x = torch.randn(
        1,
        6,
        32
    )

    modified = x.clone()

    modified[:, 4:, :] += 100.0

    output = attention(x)

    modified_output = attention(
        modified
    )

    torch.testing.assert_close(
        output[:, :4, :],
        modified_output[:, :4, :],
        rtol=1e-5,
        atol=1e-6
    )
