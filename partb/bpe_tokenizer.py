class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        self.char_to_int = {}
        self.int_to_char = {}

        # Ordered
        self.merges = []

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        self.vocab = []

        # tuple of chars to int
        self.frequency = {}

        self.UNK_token = "<|UNK|>"

    def train(self, corpus):
        raise NotImplementedError("Training method not implemented yet.")
        # ----- Pair frequency counting ------

        # -------- repeat for n iterations
        # ----- select best pair(break ties) -------
    
        # ----- apply the merge ------

        # ----- record the operation ------


        # ------ add special tokens (later as they must not be split) ------- assign reserved token IDs
    def encode(self, text):
        # raise NotImplementedError("Encoding method not implemented yet.")
        # -------- text processing --------
        # Treat space character as a distinct token
        words = text.split()

        segmented = []

        for word in words:
            chars = []
            for i, c in enumerate(word):
                token = (SPACE + c) if i==0 else c
                if token in self.char_to_int:
                    chars.append(token)
                else:
                    chars.append(self.UNK_token)

            segmented.append(chars) # segmented is list of lists

        # --------- Apply bpe merges ---------
        # Iteratively apply merge operations in exact same order

        # -------- Token ID conversion ---------
        # Convert tokens to corresponding intiger IDs
        # While handling unknown tokens using UNK

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")
        # -------- Token ID to string conversion ----------
        # Concatenate words

        # --------- Post Processing ---------
        # Handle any special tokens encountered
        # Correctly handle whitespace where space characters are tokenized

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")
        # Save Tokenizer state

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
        # Load tokeniser state
    
    def get_vocab_size(self):
        raise NotImplementedError("Get vocab size method not implemented yet.")
        # return vocab size
    
    def get_unk_id(self):
        raise NotImplementedError("Get unk id method not implemented yet.")
        # return unk id

# Metrics:
# decode(encode(text)) == text
# Less compression ratio (Token/character length)
# Less OOV rate