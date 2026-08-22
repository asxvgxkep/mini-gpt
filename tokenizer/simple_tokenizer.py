class SimpleTokenizer:

    def __init__(self, text):

        chars = sorted(
            list(set(text))
        )

        self.stoi = {
            ch:i
            for i,ch in enumerate(chars)
        }


        self.itos = {
            i:ch
            for ch,i in self.stoi.items()
        }


    def encode(self,text):

        return [
            self.stoi[c]
            for c in text
        ]


    def decode(self,tokens):

        return "".join(
            [
                self.itos[i]
                for i in tokens
            ]
        )


    @property
    def vocab_size(self):

        return len(self.stoi)