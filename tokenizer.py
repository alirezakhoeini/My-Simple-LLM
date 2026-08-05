import regex as re

class SimpleTokenizer:

    def __init__(self,str):
        self.tokens = self.tokenize(str)
        self.vocab_dict = self.build_vocab(self.tokens)
        self.reverse_vocab = self.reverse_vocab(self.vocab_dict)

    def tokenize(self, text):
        # Tokenization logic (e.g., splitting by whitespace and punctuation)
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [token.split()[0] for token in tokens if token.strip()]
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        tokens = [token if token in self.vocab_dict else '<|UNK|>' for token in tokens]
        ids = [self.vocab_dict[token] for token in tokens]
        return ids

    def build_vocab(self, tokens):
        # Build vocabulary from tokens
        ordered_list = sorted(set(tokens))
        ordered_list.extend(['<|endoftext|>','<|BOS|>',
                             '<|PAD|>', '<|EOS|>','<|UNK|>'])
        vocab_dict = {token: idx for idx, token in enumerate(ordered_list)}
        return vocab_dict

    def reverse_vocab (self, vocab_dict):
        # Create reverse vocabulary mapping
        reverse_vocab = {idx: token for token, idx in vocab_dict.items()}
        return reverse_vocab

    def decode(self, ids):
        # Decode token IDs back to text
        tokens = [self.reverse_vocab[id] for id in ids]
        text = ' '.join(tokens)
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # Remove spaces around punctuation
        text = re.sub(r"'\s+", "'", text)
        return text

    def get_vocab_size(self):
        return len(self.vocab)

    def get_tokens(self):
        return self.tokens

    def get_vocab(self):
        return self.vocab