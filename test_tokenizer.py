from tokenizer.simple_tokenizer import SimpleTokenizer


text = "hello mini gpt"


tokenizer = SimpleTokenizer(text)


tokens = tokenizer.encode(
    "hello"
)


print(tokens)

print(
    tokenizer.decode(tokens)
)


print(
    tokenizer.vocab_size
)