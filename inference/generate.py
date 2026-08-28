import argparse

import torch

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


def load_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu")
    )

    config = GPTConfig(
        **checkpoint["config"]
    )

    model = GPT(config)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

    tokenizer = SentencePieceTokenizer(
        checkpoint["tokenizer"]
    )

    return model, tokenizer, config


def generate(
    model,
    tokens,
    context_length,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50
):
    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than 0"
        )

    tokens = torch.tensor(
        tokens,
        dtype=torch.long
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        input_tokens = tokens[
            :,
            -context_length:
        ]

        logits = model(
            input_tokens
        )

        last_logits = logits[
            :,
            -1,
            :
        ]

        logits = (
            last_logits
            / temperature
        )

        k = min(
            top_k,
            logits.size(-1)
        )

        values, indices = torch.topk(
            logits,
            k,
            dim=-1
        )

        probs = torch.softmax(
            values,
            dim=-1
        )

        sampled = torch.multinomial(
            probs,
            num_samples=1
        )

        next_token = torch.gather(
            indices,
            -1,
            sampled
        )

        tokens = torch.cat(
            [
                tokens,
                next_token
            ],
            dim=1
        )

    return tokens[0].tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prompt",
        nargs="?",
        default="Once upon a time"
    )

    parser.add_argument(
        "--checkpoint",
        default="mini_gpt_best.pt"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50
    )

    args = parser.parse_args()

    model, tokenizer, config = load_model(
        args.checkpoint
    )

    tokens = tokenizer.encode(
        args.prompt
    )

    output = generate(
        model,
        tokens,
        context_length=config.context_length,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print(
        tokenizer.decode(output)
    )


if __name__ == "__main__":
    main()