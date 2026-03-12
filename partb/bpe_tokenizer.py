import os
import json
from collections import defaultdict

SPACE = "\u0120" # From terminal -> Ġ
DEBUG = False

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        self.char_to_int = {}
        self.int_to_char = {}

        # Ordered
        self.merges = []

        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []

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
            self.add_token(token) # Build vocab

        self.iters = 100

    def add_token(self, token):
        if token not in self.char_to_int:
            idx = len(self.char_to_int)
            self.char_to_int[token] = idx
            self.int_to_char[idx] = token
            self.vocab.append(token)

    def get_best_pair(self, word_freqs):
        pairs = defaultdict(int)

        for word_tuple, freq in word_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq

        if not pairs:
            return None

        # Break ties alphabetically
        return max(pairs, key=lambda p: (pairs[p], p))

    def apply_merge(self, pair, word_freqs):
        merged_pair = "".join(pair)

        new_word_freqs = {}

        for word_tuple, freq in word_freqs.items():
            new_word_list = []
            i = 0
            while i < len(word_tuple):
                if (i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i + 1] == pair[1]):
                    new_word_list.append(merged_pair)
                    i += 2
                else:
                    new_word_list.append(word_tuple[i])
                    i += 1
            new_word_freqs[tuple(new_word_list)] = freq

        return new_word_freqs

    def apply_merge_order(self, pair, segmented):
        merged_pair = "".join(pair)

        new_segmented = []

        for word in segmented:
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]):
                    new_word.append(merged_pair)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_segmented.append(new_word)

        return new_segmented

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

        N = self.vocab_size - len(self.char_to_int)
        # -------- repeat for n iterations
        for _ in range(max(0,N)):
            # ----- select best pair(break ties) -------
            pair = self.get_best_pair(word_freqs)

            if pair is None:  # no more pairs to merge
                break

            # ----- apply the merge ------
            word_freqs = self.apply_merge(pair, word_freqs)

            # ----- record the operation ------
            self.merges.append(pair)
            self.add_token("".join(pair))

        if len(self.vocab) > self.vocab_size:
            print(len(self.vocab), self.vocab_size)
            raise ValueError("Vocabulary size exceeded.")

        #DECODE
        assert(len(self.vocab) == len(self.char_to_int) == len(self.int_to_char))

        if DEBUG:
            print(f"[TRAIN] Vocab size: {len(self.vocab)}")
            print(f"[TRAIN] vocab_size: {self.vocab_size}")
            print(f"[TRAIN] Number of merges: {len(self.merges)}")

    def encode(self, text):
        if isinstance(text, list):
            # If text is a list, handle as a sequence of strings
            all_token_ids = []
            for item in text:
                all_token_ids.extend(self.encode(item))
            return all_token_ids
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

        if DEBUG:
            print(f"[ENCODE] Length of unique chars in text: {len(unique)}")
            print(f"[ENCODE] Unque chars: {unique}")
            print(f"[ENCODE] Segmented text: {segmented}")

        # --------- Apply bpe merges ---------
        # Iteratively apply merge operations in exact same order
        for pair in self.merges:
            segmented = self.apply_merge_order(pair, segmented)

        if DEBUG:
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
        text = text.replace(SPACE, " ").strip()

        return text

    def save(self, filepath):
        # raise NotImplementedError("Save method not implemented yet.")
        # Save Tokenizer state
        os.makedirs(filepath, exist_ok=True)
        save_path = os.path.join(filepath, "tokenizer.json")
        state = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "char_to_int": self.char_to_int,
            "merges": self.merges,
            "vocab": self.vocab,
            "UNK_token": self.UNK_token
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=4)

    def load(self, filepath):
        # raise NotImplementedError("Load method not implemented yet.")
        # Load tokeniser state
        load_path = os.path.join(filepath, "tokenizer.json")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No tokenizer file found at {load_path}")
            
        with open(load_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        self.vocab_size = state["vocab_size"]
        self.special_tokens = state["special_tokens"]
        self.char_to_int = state["char_to_int"]
        self.int_to_char = {int(v): k for k, v in self.char_to_int.items()}
        self.vocab = state["vocab"]
        self.merges = [tuple(m) for m in state["merges"]]
        self.UNK_token = state.get("UNK_token", "<|UNK|>")

    
    def get_vocab_size(self):
        # raise NotImplementedError("Get vocab size method not implemented yet.")
        # return vocab size
        return len(self.vocab)
    
    def get_unk_id(self):
        # raise NotImplementedError("Get unk id method not implemented yet.")
        # return unk id
        return self.char_to_int[self.UNK_token]

# Metrics:
# decode(encode(text)) == text
# Less compression ratio (Token/character length)
# Less OOV rate