import torch
import sys

from model.config import GPTConfig
from model.gpt import GPT

from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


checkpoint = torch.load(
    "mini_gpt.pt",
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


def generate(
    model,
    tokens,
    max_new_tokens=100
):

    tokens = torch.tensor(
        tokens
    ).unsqueeze(0)


    for _ in range(max_new_tokens):

        logits = model(tokens)


        last_logits = logits[:, -1, :]


        temperature = 0.8

        logits = last_logits / temperature


        top_k = 50

        values, indices = torch.topk(
            logits,
            top_k,
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
        ).squeeze(0)


        tokens = torch.cat(
            [
                tokens,
                next_token.unsqueeze(0)
            ],
            dim=1
        )


    return tokens[0].tolist()




prompt = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "Once upon a time"
)


tokens = tokenizer.encode(
    prompt
)


output = generate(
    model,
    tokens
)


print(
    tokenizer.decode(output)
)
