from pathlib import Path

from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


TOKENIZER_PATH = Path(__file__).parent / "tokenizer" / "tiny.model"


def test_tiny_tokenizer_vocab_size():
    tokenizer = SentencePieceTokenizer(
        str(TOKENIZER_PATH)
    )

    assert tokenizer.vocab_size == 8000


def test_tiny_tokenizer_round_trip():
    tokenizer = SentencePieceTokenizer(
        str(TOKENIZER_PATH)
    )

    text = (
        "Once upon a time there was "
        "a little girl named Lily."
    )

    tokens = tokenizer.encode(text)

    decoded = tokenizer.decode(tokens)

    assert decoded == text


def test_tiny_tokenizer_returns_token_ids():
    tokenizer = SentencePieceTokenizer(
        str(TOKENIZER_PATH)
    )

    tokens = tokenizer.encode(
        "Once upon a time"
    )

    assert tokens
    assert all(
        isinstance(token, int)
        for token in tokens
    )
