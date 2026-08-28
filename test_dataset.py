import pytest
import torch

from train.dataset import TextDataset


def test_dataset_returns_shifted_sequences():
    tokens = [
        1, 2, 3, 4,
        5, 6, 7
    ]

    dataset = TextDataset(
        tokens,
        context_length=3
    )

    x, y = dataset[0]

    torch.testing.assert_close(
        x,
        torch.tensor([1, 2, 3])
    )

    torch.testing.assert_close(
        y,
        torch.tensor([2, 3, 4])
    )


def test_dataset_length():
    dataset = TextDataset(
        list(range(10)),
        context_length=3
    )

    assert len(dataset) == 3


def test_dataset_raises_index_error():
    dataset = TextDataset(
        [1, 2, 3, 4],
        context_length=2
    )

    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
