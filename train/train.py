import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer
from train.dataset import TextDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path):
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        config_dict = yaml.safe_load(f)

    return GPTConfig(
        **config_dict
    )


def encode_corpus(
    tokenizer,
    text,
    chunk_size=10_000_000
):
    tokens = []

    for i in range(
        0,
        len(text),
        chunk_size
    ):
        chunk = text[
            i:i + chunk_size
        ]

        tokens.extend(
            tokenizer.encode(chunk)
        )

        print(
            f"encoded {i}/{len(text)}"
        )

    return tokens


def evaluate(
    model,
    loader,
    loss_fn,
    vocab_size
):
    model.eval()

    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)

            loss = loss_fn(
                logits.view(
                    -1,
                    vocab_size
                ),
                y.view(-1)
            )

            total_loss += loss.item()
            steps += 1

    model.train()

    return total_loss / steps


def save_checkpoint(
    path,
    model,
    optimizer,
    config,
    tokenizer_path,
    epoch,
    val_loss
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.__dict__,
            "tokenizer": tokenizer_path,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Total target number of epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Learning rate. Defaults to 3e-4 "
            "for a fresh run. When resuming, "
            "the checkpoint optimizer LR is "
            "preserved unless this is provided."
        )
    )

    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint path to resume from"
    )

    args = parser.parse_args()

    print(
        "Training device:",
        DEVICE
    )

    data_path = "data/train.txt"
    config_path = "configs/v4.yaml"
    tokenizer_path = "tokenizer/tiny.model"

    os.makedirs(
        "logs",
        exist_ok=True
    )

    os.makedirs(
        "checkpoints_full",
        exist_ok=True
    )

    with open(
        data_path,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    tokenizer = SentencePieceTokenizer(
        tokenizer_path
    )

    print(
        "Encoding text..."
    )

    tokens = encode_corpus(
        tokenizer,
        text
    )

    print(
        "Token count:",
        len(tokens)
    )

    split = int(
        len(tokens) * 0.9
    )

    train_tokens = tokens[
        :split
    ]

    val_tokens = tokens[
        split:
    ]

    config = load_config(
        config_path
    )

    config.vocab_size = (
        tokenizer.vocab_size
    )

    train_dataset = TextDataset(
        train_tokens,
        context_length=config.context_length
    )

    val_dataset = TextDataset(
        val_tokens,
        context_length=config.context_length
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

    model = GPT(
        config
    ).to(DEVICE)

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Parameters: {params / 1e6:.2f}M"
    )

    initial_lr = (
        args.lr
        if args.lr is not None
        else 3e-4
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr
    )

    loss_fn = (
        torch.nn.CrossEntropyLoss()
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=DEVICE
        )

        model.load_state_dict(
            checkpoint["model"]
        )

        start_epoch = checkpoint.get(
            "epoch",
            0
        )

        best_val_loss = checkpoint.get(
            "val_loss",
            float("inf")
        )

        if "optimizer" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

            print(
                "Restored optimizer state"
            )
        else:
            print(
                "Checkpoint has no optimizer "
                "state; using a new optimizer"
            )

        if args.lr is not None:
            for group in (
                optimizer.param_groups
            ):
                group["lr"] = args.lr

        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )

    if args.epochs <= start_epoch:
        raise ValueError(
            "--epochs must be greater than "
            "the checkpoint epoch"
        )

    log_mode = (
        "a"
        if args.resume is not None
        else "w"
    )

    last_epoch = start_epoch
    last_val_loss = best_val_loss

    for epoch in range(
        start_epoch,
        args.epochs
    ):
        model.train()

        total_loss = 0.0
        steps = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

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

        avg_loss = (
            total_loss / steps
        )

        val_loss = evaluate(
            model,
            val_loader,
            loss_fn,
            config.vocab_size
        )

        current_epoch = (
            epoch + 1
        )

        print(
            f"epoch={current_epoch}, "
            f"train_loss={avg_loss:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        latest_path = (
            "checkpoints_full/latest.pt"
        )

        save_checkpoint(
            latest_path,
            model,
            optimizer,
            config,
            tokenizer_path,
            current_epoch,
            val_loss
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            save_checkpoint(
                "mini_gpt_best.pt",
                model,
                optimizer,
                config,
                tokenizer_path,
                current_epoch,
                val_loss
            )

            print(
                "Saved best model"
            )

        with open(
            "logs/train.log",
            log_mode,
            encoding="utf-8"
        ) as f:
            f.write(
                f"epoch={current_epoch}, "
                f"train_loss={avg_loss:.4f}, "
                f"val_loss={val_loss:.4f}\n"
            )

        log_mode = "a"

        last_epoch = current_epoch
        last_val_loss = val_loss

    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "tokenizer": tokenizer_path,
            "epoch": last_epoch,
            "val_loss": last_val_loss,
        },
        "mini_gpt_full.pt"
    )

    print(
        "model saved"
    )


if __name__ == "__main__":
    main()