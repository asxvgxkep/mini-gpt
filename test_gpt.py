import torch

from model.config import GPTConfig
from model.gpt import GPT


config = GPTConfig()


model = GPT(config)


x = torch.randint(
    0,
    config.vocab_size,
    (2,8)
)


logits = model(x)


print(logits.shape)