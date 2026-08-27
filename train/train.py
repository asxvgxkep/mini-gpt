import os
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
    text = f.read()


tokenizer = SentencePieceTokenizer(
    "tokenizer/tiny.model"
)


print(
    "Encoding text..."
)


tokens = []

chunk_size = 10_000_000

for i in range(0, len(text), chunk_size):

    chunk = text[i:i+chunk_size]

    tokens.extend(
        tokenizer.encode(chunk)
    )

    print(
        f"encoded {i}/{len(text)}"
    )


print(
    "Token count:",
    len(tokens)
)


split = int(
    len(tokens) * 0.9
)


train_tokens = tokens[:split]

val_tokens = tokens[split:]


train_dataset = TextDataset(
    train_tokens,
    context_length=256
)


val_dataset = TextDataset(
    val_tokens,
    context_length=256
)


print(
    "Train tokens:",
    len(train_tokens)
)

print(
    "Val tokens:",
    len(val_tokens)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)



# =========================
# 2. 创建模型
# =========================

config = load_config(
    "configs/v4.yaml"
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

def evaluate(model, loader):

    model.eval()

    total_loss = 0
    steps = 0

    with torch.no_grad():

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

            total_loss += loss.item()

            steps += 1


    model.train()

    return total_loss / steps

# =========================
# 4. Training
# =========================

os.makedirs("logs", exist_ok=True)

os.makedirs("checkpoints_full", exist_ok=True)

os.makedirs("checkpoints_full", exist_ok=True)

best_val_loss = float("inf")


for epoch in range(10):

    total_loss = 0
    steps = 0


    for x, y in train_loader:

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



    avg_loss = total_loss / steps

    val_loss = evaluate(
        model,
        val_loader
    )

    print(
        f"epoch {epoch+1}, train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model": model.state_dict(),
                "config": config.__dict__,
                "tokenizer": "tokenizer/tiny.model",
                "epoch": epoch + 1,
                "val_loss": val_loss,
            },
            "mini_gpt_best.pt"
        )

        print(
            "Saved best model"
        )

    with open(
        "logs/train.log",
        "a",
        encoding="utf-8"
    ) as f:
        f.write(
            f"epoch={epoch+1}, train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}\n"
        )

torch.save(
    {
        "model": model.state_dict(),
        "config": config.__dict__,
        "tokenizer": "tokenizer/tiny.model",
        "epoch": epoch + 1,
    },
    f"checkpoints_full/checkpoint_epoch_{epoch+1}.pt"
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

    "mini_gpt_full.pt"
)


print(
    "model saved"
)