import os
import json
from collections import defaultdict, Counter

SPACE = "\u0120" # From terminal -> Ġ
DEBUG = False

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        self.char_to_int = {}
        self.int_to_char = {}

        self.min_freq = 2

        # Ordered
        self.merges = []
        self.merge_order = {}

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

    def get_best_pair(self, pair_counts):
        return max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))

    def apply_merge(self, pair, pair_counts, pair_to_words, word_list, word_counts):
        merged_pair = "".join(pair)

        indices_to_update = list(pair_to_words[pair])
        del pair_counts[pair] # That pair count removed
        del pair_to_words[pair] # words associated with that pair removed

        for i in indices_to_update:
            word_tuple = word_list[i]
            # (h, a, pp, y)
            word_frequency = word_counts[i]
            # 5

            # Remove damaged pairs as well
            for j in range(len(word_tuple) - 1):
                p = (word_tuple[j], word_tuple[j+1])
                if p in pair_to_words and i in pair_to_words[p]:
                    pair_to_words[p].remove(i)
                pair_counts[p] -= word_frequency
                if pair_counts[p] <= 0 and p in pair_counts:
                    del pair_counts[p]

            new_word_tuple = []
            j = 0
            while j < len(word_tuple):
                if j < len(word_tuple) - 1 and (word_tuple[j], word_tuple[j+1]) == pair:
                    new_word_tuple.append(merged_pair)
                    j+=2
                else:
                    new_word_tuple.append(word_tuple[j])
                    j+=1
            
            new_word_tuple = tuple(new_word_tuple)
            word_list[i] = new_word_tuple
            # No change in word_frequency

            for j in range(len(new_word_tuple) - 1):
                p = (new_word_tuple[j], new_word_tuple[j+1])
                pair_counts[p] += word_frequency
                pair_to_words[p].add(i)

    def apply_merge_order(self, chars):
        # find merge index
        while len(chars) > 1:
            min_merge_idx = float('inf')
            best_pair_index = -1

            for j in range(len(chars) - 1):
                pair = (chars[j], chars[j+1])

                if pair in self.merge_order:
                    idx = self.merge_order[pair]
                    if idx < min_merge_idx:
                        # update min_merge_idx
                        min_merge_idx = idx
                        best_pair_index = j

            if best_pair_index == -1:
                break

            pair = (chars[best_pair_index], chars[best_pair_index+1])
            new_token = "".join(pair)
            new_chars_list = chars[:best_pair_index] + [new_token] + chars[best_pair_index+2:]
            chars = new_chars_list

        return chars


    def train(self, corpus):
        # raise NotImplementedError("Training method not implemented yet.")

        # ----- Pair frequency counting ------
        word_freq = defaultdict(int)
        for sentence in corpus:
            words = sentence.split(' ')
            word_freq[tuple(words[0])] += 1
            for w in words[1:]:
                word_freq[tuple(SPACE + w)] += 1
        # example
        # the -> 5
        # a -> 7

        # unique = set()
        # for word, freq in word_freq.items():
        #     chars = []
        #     if word == "":
        #         chars = [SPACE]
        #     else:
        #         for i, c in enumerate(word):
        #             token = (SPACE + c) if i==0 else c
        #             chars.append(token) # List of chars
        #     # tuple(chars) -> tuple of list elements
        #     self.frequency[tuple(chars)] = freq
        #     unique.update(chars)
        # example
        # (_t,h,e) -> 5
        # (_a) -> 7

        # for c in unique:
        #     self.add_token(c) #Build vocabulary of characters

        char_freqs = Counter()
        for word, freq in word_freq.items():
            for char in word:
                char_freqs[char] += freq

        max_base_chars = self.vocab_size - len(self.char_to_int)
        for char, _ in char_freqs.most_common(max_base_chars):
            self.add_token(char)

        # working copy
        # word_freqs = dict(self.frequency)

        word_list = list(word_freq.keys())
        word_counts = list(word_freq.values())

        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)

        for i, word in enumerate(word_list):
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                pair_counts[pair] += word_counts[i]
                pair_to_words[pair].add(i)


        N = self.vocab_size - len(self.char_to_int)
        # -------- repeat for n iterations
        for i in range(max(0,N)):
            # ----- select best pair(break ties) -------
            pair = self.get_best_pair(pair_counts)

            if pair is None:  # no more pairs to merge
                break

            if pair_counts[pair] < self.min_freq:
                break

            # ----- apply the merge ------
            self.apply_merge(pair, pair_counts, pair_to_words, word_list, word_counts)

            # ----- record the operation ------
            self.merges.append(pair)
            self.merge_order[pair] = i
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
        if not text:
            return []
            
        words = text.split(" ")

        # segmented = []
        token_ids = []

        for word_idx, word in enumerate(words):
            chars = []
            if word == "":
                # Preserve the consecutive space!
                chars.append(SPACE)
            else:
                if word_idx > 0:
                    chars.append(SPACE)
                for c in word:
                    if c in self.char_to_int:
                        chars.append(c)
                    else:
                        chars.append(self.UNK_token)
            

            # Why save it, encode it here itself
            # segmented.append(chars) # segmented is list of lists
            # --------- Apply bpe merges ---------
            # Iteratively apply merge operations in exact same order
            chars = self.apply_merge_order(chars)

            # -------- Token ID conversion ---------
            # Convert tokens to corresponding intiger IDs
            # While handling unknown tokens using UNK
            for token in chars:
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
        self.merge_order = {m: i for i, m in enumerate(self.merges)}
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