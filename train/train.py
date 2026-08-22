import yaml

import torch

from torch.utils.data import DataLoader

from model.config import GPTConfig

from model.gpt import GPT

from train.dataset import TextDataset

from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        config_dict = yaml.safe_load(f)

    return GPTConfig(**config_dict)


print(
    "Training device:",
    device
)


# =========================
# 1. 准备数据
# =========================

with open(
    "data/train.txt",
    "r",
    encoding="utf-8"
) as f:

    # 第一次实验先读取 2MB
    text = f.read(100_000_000)


tokenizer = SentencePieceTokenizer(
    "tokenizer/tiny.model"
)


print(
    "Encoding text..."
)


tokens = tokenizer.encode(text)


print(
    "Token count:",
    len(tokens)
)


dataset = TextDataset(
    tokens,
    context_length=256
)


print(
    "Dataset ready"
)


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)



# =========================
# 2. 创建模型
# =========================

config = load_config(
    "configs/tiny.yaml"
)


config.vocab_size = tokenizer.vocab_size


model = GPT(config).to(device)
params = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Parameters: {params/1e6:.2f}M"
)



# =========================
# 3. 优化器
# =========================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)


loss_fn = torch.nn.CrossEntropyLoss()



# =========================
# 4. Training
# =========================

for epoch in range(20):

    total_loss = 0
    steps = 0


    for x, y in loader:

        x = x.to(device)
        y = y.to(device)


        logits = model(x)


        loss = loss_fn(
            logits.view(
                -1,
                config.vocab_size
            ),
            y.view(-1)
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        total_loss += loss.item()

        steps += 1



    print(
        f"epoch {epoch+1}, loss={total_loss / steps:.4f}"
    )

torch.save(
    {
        "model": model.state_dict(),
        "config": config.__dict__,
        "tokenizer": "tokenizer/tiny.model",
        "epoch": epoch + 1,
    },
    f"checkpoints_v03/checkpoint_epoch_{epoch+1}.pt"
)



# =========================
# 5. 保存模型
# =========================

torch.save(
    {
        "model": model.state_dict(),

        "config": config.__dict__,

        "tokenizer": "tokenizer/tiny.model",
    },

    "mini_gpt.pt"
)


print(
    "model saved"
)