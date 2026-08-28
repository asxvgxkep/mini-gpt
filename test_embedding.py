import torch

from model.embedding import (
    PositionEmbedding,
    TokenEmbedding,
)


def test_token_embedding_shape():
    embedding = TokenEmbedding(
        vocab_size=5000,
        hidden_dim=128
    )

    tokens = torch.tensor(
        [
            [1, 2, 3, 4],
            [4, 3, 2, 1],
        ]
    )

    output = embedding(tokens)

    assert output.shape == (
        2,
        4,
        128
    )


def test_position_embedding_shape():
    embedding = PositionEmbedding(
        context_length=128,
        hidden_dim=128
    )

    tokens = torch.tensor(
        [
            [1, 2, 3, 4],
            [9, 8, 7, 6],
        ]
    )

    output = embedding(tokens)

    assert output.shape == (
        4,
        128
    )


def test_position_embedding_is_independent_of_token_values():
    embedding = PositionEmbedding(
        context_length=16,
        hidden_dim=32
    )

    first = torch.tensor(
        [[1, 2, 3, 4]]
    )

    second = torch.tensor(
        [[9, 9, 9, 9]]
    )

    first_output = embedding(first)
    second_output = embedding(second)

    torch.testing.assert_close(
        first_output,
        second_output
    )
