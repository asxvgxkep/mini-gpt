import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                hidden_dim,
                hidden_dim * 4
            ),

            nn.GELU(),

            nn.Linear(
                hidden_dim * 4,
                hidden_dim
            )
        )


    def forward(self, x):
        return self.net(x)