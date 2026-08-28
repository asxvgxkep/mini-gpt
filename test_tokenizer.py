from tokenizer.simple_tokenizer import SimpleTokenizer


def test_simple_tokenizer_round_trip():
    text = "hello mini gpt"

    tokenizer = SimpleTokenizer(
        text
    )

    tokens = tokenizer.encode(
        "hello"
    )

    decoded = tokenizer.decode(
        tokens
    )

    assert decoded == "hello"


def test_simple_tokenizer_vocab_size():
    text = "hello"

    tokenizer = SimpleTokenizer(
        text
    )

    assert tokenizer.vocab_size == len(
        set(text)
    )


def test_simple_tokenizer_ids_are_integers():
    tokenizer = SimpleTokenizer(
        "hello mini gpt"
    )

    tokens = tokenizer.encode(
        "mini"
    )

    assert all(
        isinstance(token, int)
        for token in tokens
    )
