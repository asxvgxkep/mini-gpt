import torch


class TextDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        tokens,
        context_length
    ):
        self.tokens = tokens
        self.context_length = context_length


    def __len__(self):
        return (
            len(self.tokens) - 1
        ) // self.context_length


    def __getitem__(self, idx):

        if idx >= len(self):
            raise IndexError

        start = idx * self.context_length

        x = self.tokens[
            start:
            start + self.context_length
        ]

        y = self.tokens[
            start + 1:
            start + self.context_length + 1
        ]

        return (
            torch.tensor(x),
            torch.tensor(y)
        )