import torch

from model.config import GPTConfig
from model.gpt import GPT


def test_gpt_output_shape():
    config = GPTConfig(
        vocab_size=100,
        context_length=16,
        n_layers=2,
        n_heads=4,
        hidden_dim=32,
        dropout=0.0
    )

    model = GPT(config)

    tokens = torch.randint(
        0,
        config.vocab_size,
        (2, 8)
    )

    logits = model(tokens)

    assert logits.shape == (
        2,
        8,
        config.vocab_size
    )


def test_gpt_is_causal_at_full_context_length():
    torch.manual_seed(0)

    config = GPTConfig(
        vocab_size=100,
        context_length=16,
        n_layers=2,
        n_heads=4,
        hidden_dim=32,
        dropout=0.0
    )

    model = GPT(config)
    model.eval()

    tokens = torch.randint(
        0,
        config.vocab_size,
        (1, config.context_length)
    )

    modified = tokens.clone()
    modified[:, 8:] = (
        modified[:, 8:] + 1
    ) % config.vocab_size

    logits = model(tokens)
    modified_logits = model(modified)

    torch.testing.assert_close(
        logits[:, :8, :],
        modified_logits[:, :8, :],
        rtol=1e-5,
        atol=1e-6
    )
