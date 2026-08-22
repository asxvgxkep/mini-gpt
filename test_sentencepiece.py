from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


tokenizer = SentencePieceTokenizer(
    "tokenizer/mini.model"
)


text = "hello mini gpt"


tokens = tokenizer.encode(text)


print(tokens)

print(
    tokenizer.decode(tokens)
)

print(
    tokenizer.vocab_size
)