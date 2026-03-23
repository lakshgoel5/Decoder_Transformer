import os
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import unicodedata
import re

SPACE = "\u0120" # From terminal -> Ġ
DEBUG = False

# Knowledge of Hindi vibhakti
# Vowels in Hindi are called svar (स्वर). There are 13 vowels in total. They appear in two forms: their full, independent form when they start a word or stand alone, and their diacritic (mātrā) form when they are attached to a consonant to change its vowel sound.
# The first vowel, अ (a), is special. It has no mātrā because it's the inherent vowel sound automatically included with every consonant. All other vowels have a corresponding diacritic mark.

# Problem 1 — Matras (vowel diacritics) get split from their consonant
# BPE might split `का` into `क` + `ा` — which is meaningless. `ा` alone has no pronunciation.

# Problem 2 — Nukta characters
# क़, ख़, ग़, ज़, ड़, ढ़, फ़  
# These are consonant + nukta (़) 
# BPE might separate the nukta from its consonant

# r stands for raw string, so that \u is treated as a unicode character and not an escape sequence
DEVANAGARI_CHAR = r'\u0900-\u097F'
VEDIC_EXT = r'\u1CD0–\u1CFF'
DEVANAGARI_EXT = r'\uA8E0–\uA8FF'
MATRAS = set('\u093C\u093E\u093F\u0940\u0941\u0942\u0943\u0944'
             '\u0945\u0946\u0947\u0948\u0949\u094A\u094B\u094C'
             '\u094E\u094F\u0902\u0903\u0901')
HALANT = '\u094D'

MULTI_WORD = ["के लिए", "की तरह", "के बाद", "के पास"]

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        # raise NotImplementedError("BPETokenizer initialization not implemented yet.")
        self.char_to_int = {}
        self.int_to_char = {}

        self.min_freq = 3

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

        self.HINDI_PROTECTED = [
            # Vibhakti: particles added to nouns or pronouns to indicate their role in a sentence: 7 types
            "ने", "को", "से", "का", "के", "की", "में", "पर", "तक",
            "के लिए", "की तरह", "के बाद", "के पास",
            
            # Hindi verbs are formed by adding suffixes to the root (Dhatu). The verb changes based on tense, gender, and number.
            "ता", "ती", "ते", "ना", "नी", "ने",
            "गा", "गी", "गे", # future tense
            "था", "थी", "थे", # past tense  
            "है", "हैं", "हो", "हूँ", # present tense
            
            # Pratyaya are words added to the end of words to create new meanings.
            "वाला", "वाली", "वाले", # agent/adjective marker
            "ों", "यों", # plural oblique
            
            # High frequency function words
            "और", "या", "कि", "जो", "तो", "भी", "ही", 
            # "न", "नहीं",
            # "यह", "वह", "वे", "हम", "आप", "मैं", "तुम",
        ]

        self.DEVANAGARI_DIGITS = ['०','१','२','३','४','५','६','७','८','९']

        self.protected_tokens = set(self.HINDI_PROTECTED) # O(1) lookup

        self.PUNCTUATIONS = [
            # Punctuation
            "।", "।।", ",", "!", "?", "-", "—", "(", ")", "\"", "'", ":", ";", "||", "| |", "|"
        ]

        for token in self.special_tokens:
            if token not in SPECIALS:
                SPECIALS.append(token)
        # Special has all special tokens

        # ------ add special tokens (later as they must not be split) ------- assign reserved token IDs
        for token in SPECIALS:
            self.add_token(token) # Build vocab

        for token in self.HINDI_PROTECTED:
            self.add_token(token)

        for token in self.PUNCTUATIONS:
            self.add_token(token)

        for digit in self.DEVANAGARI_DIGITS:
            self.add_token(digit)

        self.iters = 100

    def get_pair_weight(self, a: str, b: str) -> float:
        a_core = a.lstrip(SPACE)  # strip space prefix for checking
        b_core = b.lstrip(SPACE)

        # Never merge anything into a vibhakti from the right
        if b_core in self.protected_tokens and a_core != "":
            return 0.5

        # Never merge a vibhakti with what follows on the right
        if a_core in self.protected_tokens:
            return 0.5

        # --- BOOST: nukta bonds with its consonant ---
        # ड + ़ → ड़
        # NOTE: Nukta (\u093C) is present in MATRAS. This must be checked before MATRAS.
        if b_core == '\u093C':
            return 30.0

        # BOOST: matra must bond tightly with preceding consonant
        if b_core and all(c in MATRAS for c in b_core):
            return 5.0

        # --- BOOST: halant conjuncts stay together ---
        # क् + ष → क्ष
        if a_core.endswith('\u094D'):
            return 8.0

        # if a_core in self.PUNCTUATIONS or b_core in self.PUNCTUATIONS:
        #     return 1

        # --- MILD DISCOURAGE: cross-word merges (SPACE boundary) ---
        # BPE might merge end-of-word with start-of-next
        # if SPACE in b and b != SPACE:
        #     return 0.3

        return 1.0  # default

    def normalize_hindi(self, text):
        for punc in self.PUNCTUATIONS:
            text = text.replace(punc, f" {punc} ")

        return text

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
                weight = self.get_pair_weight(word_tuple[j], word_tuple[j+1])
                pair_counts[p] -= word_frequency * weight
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
                weight = self.get_pair_weight(new_word_tuple[j], new_word_tuple[j+1])
                pair_counts[p] += word_frequency * weight
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
  
    # Solving problems defiend above
    def word_to_unit(self, word):
        units = []
        i = 0

        while i < len(word):
            # Protected tokens (longest match first)
            matched = False
            for token in sorted(self.protected_tokens, key=len, reverse=True):
                if word[i:].startswith(token):
                    units.append(token)
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue

            # SPACE
            if word[i] == SPACE:
                units.append(SPACE)
                i += 1
                continue

            # Punctuation (longest match first)
            punc_matched = False
            for punc in sorted(self.PUNCTUATIONS, key=len, reverse=True):
                if word[i:].startswith(punc):
                    units.append(punc)
                    i += len(punc)
                    punc_matched = True
                    break
            if punc_matched:
                continue

            # Devanagari unit — consonant + matras/halant
            unit = word[i]
            i += 1
            while i < len(word):
                c = word[i]
                if c in MATRAS:
                    unit += c
                    i += 1
                elif c == HALANT:
                    if i + 1 < len(word):# normal case — glue next consonant
                        unit += c + word[i + 1]
                        i += 2
                    else: # trailing halant — glue it anyway
                        unit += c
                        i += 1
                else:
                    break
            units.append(unit)

        return units
            


    def train(self, corpus):
        # raise NotImplementedError("Training method not implemented yet.")

        # corpus = [unicodedata.normalize('NFC', sentence) for sentence in corpus]

        # ----- Pair frequency counting ------
        word_freq = defaultdict(int)
        for sentence in corpus:
            words = sentence.split(' ')
            for w_idx, w in enumerate(words):

                if not w: # Skip training spaces
                    continue

                prefix = '' if w_idx == 0 else SPACE
                # unit = self.word_to_unit(prefix + w)
                unit = prefix + w
                if unit:
                    word_freq[tuple(unit)] += 1
        # example
        # (_t,h,e) -> 5
        # (_a) -> 7

        char_freqs = Counter()
        for word, freq in word_freq.items():
            for char in word:
                char_freqs[char] += freq

        max_base_chars = self.vocab_size - len(self.char_to_int)
        for char, _ in char_freqs.most_common(max_base_chars):
            self.add_token(char)

        word_list = list(word_freq.keys())
        word_counts = list(word_freq.values())

        pair_counts = defaultdict(int)
        pair_to_words = defaultdict(set)

        for i, word in enumerate(word_list):
            for j in range(len(word) - 1):
                pair = (word[j], word[j+1])
                weight = self.get_pair_weight(word[j], word[j+1])
                pair_counts[pair] += word_counts[i] * weight
                pair_to_words[pair].add(i)


        N = self.vocab_size - len(self.char_to_int)
        # -------- repeat for n iterations
        for i in tqdm(range(max(0, N)), desc="BPE training"):
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
        
        # text = unicodedata.normalize('NFC', text)
            
        words = text.split(" ")
        token_ids = []

        for word_idx, word in enumerate(words):
            chars = []
            if word == "":
                # Preserve the consecutive space!
                chars.append(SPACE)
            else:
                prefix = "" if word_idx == 0 else SPACE
                # unit = self.word_to_unit(prefix + word)
                unit = prefix + word
                for token in unit:
                    if token in self.char_to_int:
                        chars.append(token)
                    else:
                        for c in token:
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
                token_ids.append(self.char_to_int.get(token, self.char_to_int[self.UNK_token]))

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
            "UNK_token": self.UNK_token,
            "PUNCTUATIONS": self.PUNCTUATIONS,
            "HINDI_PROTECTED": self.HINDI_PROTECTED
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
        self.PUNCTUATIONS = state.get("PUNCTUATIONS", [])
        self.HINDI_PROTECTED = state.get("HINDI_PROTECTED", [])
        self.protected_tokens = set(self.HINDI_PROTECTED)

    
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