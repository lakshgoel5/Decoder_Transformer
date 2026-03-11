from collections import defaultdict

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

        SPECIALS = ["<|PAD|>", "<|UNK|>", "<|EOS|>"]

        self.reserved_count = 10 # First 10 int's saved for reserved tokens

        for token in self.special_tokens:
            if token not in SPECIALS:
                SPECIALS.append(token)
        # Special has all special tokens

        # ------ add special tokens (later as they must not be split) ------- assign reserved token IDs
        for token in SPECIALS:
            self.add_token(token, special = true) # Build vocab

    def add_token(token, special = False):
        pass

    def get_best_pair(word_freqs):
        pass

    def apply_merge(pair, word_freqs):
        pass

    def apply_merge_order(pair, segmented):
        pass

    def train(self, corpus):
        # raise NotImplementedError("Training method not implemented yet.")

        # ----- Pair frequency counting ------
        word_freq = defaultdict(int)
        for word in corpus:
            for token in word.split():
                word_freq[token] += 1
        # example
        # the -> 5
        # a -> 7

        unique = set()
        for word, freq in word_freq.items():
            chars = []
            for i, c in enumerate(word):
                token = (SPACE + c) if i==0 else c
                chars.append(token) # List of chars
            # tuple(chars) -> tuple of list elements
            self.frequency[tuple(chars)] = freq
            unique.update(chars)
        # example
        # (_t,h,e) -> 5
        # (_a) -> 7

        for c in unique:
            self.add_token(c) #Build vocabulary of characters

        # working copy
        word_freqs = dict(self.frequency)

        N = 10
        # -------- repeat for n iterations
        for _ in range(N):
            # ----- select best pair(break ties) -------
            pair = self.get_best_pair(word_freqs)

            # ----- apply the merge ------
            word_freq = self.apply_merge(pair, word_freqs)

            # ----- record the operation ------
            self.merges.append(pair)

        if len(self.vocab) > self.vocab_size:
            raise ValueError("Vocabulary size exceeded.")

        #DECODE
        assert(len(self.vocab) == len(self.char_to_int) == len(self.int_to_char))

        print(f"[TRAIN] Vocab size: {len(self.vocab)}")
        print(f"[TRAIN] vocab_size: {self.vocab_size}")
        print(f"[TRAIN] Number of merges: {len(self.merges)}")

    def encode(self, text):
        # raise NotImplementedError("Encoding method not implemented yet.")
        # -------- text processing --------
        # Treat space character as a distinct token
        words = text.split()

        segmented = []

        unique = set()
        for word in words:
            chars = []
            for i, c in enumerate(word):
                token = (SPACE + c) if i==0 else c
                if token in self.char_to_int:
                    chars.append(token)
                else:
                    # Unknown character -> replace with UNK
                    chars.append(self.UNK_token)
            
            unique.update(chars)

            segmented.append(chars) # segmented is list of lists

        print(f"[ENCODE] Length of unique chars in text: {len(unique)}")
        print(f"[ENCODE] Segmented text: {segmented}")

        # --------- Apply bpe merges ---------
        # Iteratively apply merge operations in exact same order
        for pair in self.merges:
            segmented = self.apply_merge_order(pair, segmented)

        print(f"[ENCODE] Length of segmented: {len(segmented)}")
        print(f"[ENCODE] Segmented text: {segmented}")

        # -------- Token ID conversion ---------
        # Convert tokens to corresponding intiger IDs
        # While handling unknown tokens using UNK
        token_ids = []
        for word in segmented:
            for token in word:
                if token in self.char_to_int:
                    token_ids.append(self.char_to_int[token])
                else:
                    token_ids.append(self.char_to_int[self.UNK_token])
        
        print(f"[ENCODE] Token IDs: {token_ids}")
        return token_ids

    def decode(self, token_ids):
        # raise NotImplementedError("Decoding method not implemented yet.")
        # -------- Token ID to string conversion ----------
        # Concatenate words
        tokens = []
        for token_id in token_ids:
            tokens.append(self.int_to_char[token_id])
        
        # --------- Post Processing ---------
        # Handle any special tokens encountered #DEBUG
        # Correctly handle whitespace where space characters are tokenized
        text = "".join(tokens)
        text = text.replace(SPACE, " ")

        print(f"[DECODE] Decoded text: {text}")
        return text

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")
        # Save Tokenizer state

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
        # Load tokeniser state
    
    def get_vocab_size(self):
        # raise NotImplementedError("Get vocab size method not implemented yet.")
        # return vocab size
        return len(self.vocab)
    
    def get_unk_id(self):
        # raise NotImplementedError("Get unk id method not implemented yet.")
        # return unk id
        return self.char_to_int(self.UNK_token)

# Metrics:
# decode(encode(text)) == text
# Less compression ratio (Token/character length)
# Less OOV rate