from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


tokenizer = SentencePieceTokenizer(
    "tokenizer/tiny.model"
)


text = "Once upon a time there was a little girl"


tokens = tokenizer.encode(text)


print(tokens)

print(
    tokenizer.decode(tokens)
)

print(
    tokenizer.vocab_size
)