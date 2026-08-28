from pathlib import Path

from tokenizer.sentencepiece_tokenizer import SentencePieceTokenizer


TOKENIZER_PATH = Path(__file__).parent / "tokenizer" / "mini.model"


def test_sentencepiece_tokenizer_round_trip():
    tokenizer = SentencePieceTokenizer(str(TOKENIZER_PATH))
    text = "hello mini gpt"

    tokens = tokenizer.encode(text)

    assert tokenizer.decode(tokens) == text


def test_sentencepiece_tokenizer_vocab_size():
    tokenizer = SentencePieceTokenizer(str(TOKENIZER_PATH))

    assert tokenizer.vocab_size == 100


def test_sentencepiece_tokenizer_returns_token_ids():
    tokenizer = SentencePieceTokenizer(str(TOKENIZER_PATH))

    tokens = tokenizer.encode("hello mini gpt")

    assert tokens
    assert all(isinstance(token, int) for token in tokens)
