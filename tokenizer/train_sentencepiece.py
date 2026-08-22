import sentencepiece as spm


spm.SentencePieceTrainer.train(
    input="data/train.txt",
    model_prefix="tokenizer/tiny",
    vocab_size=8000,
    character_coverage=1.0,
    model_type="bpe"
)