import sentencepiece as spm


class SentencePieceTokenizer:

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(
            model_file=model_path
        )


    def encode(self, text):

        return self.sp.encode(
            text,
            out_type=int
        )


    def decode(self, tokens):

        return self.sp.decode(
            tokens
        )


    @property
    def vocab_size(self):

        return self.sp.get_piece_size()