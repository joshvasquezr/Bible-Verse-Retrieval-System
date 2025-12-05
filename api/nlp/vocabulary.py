"""
vocabulary builder w/ frequency filtering
"""
from collections import Counter
from typing import List, Dict


class Vocabulary:
    """build vocab from corpus w/ frequency filtering"""

    def __init__(self, min_count: int = 5, max_vocab_size: int = None):
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        self.vocab_size: int = 0

    def build_vocab(self, tokenized_corpus: List[List[str]]):
        """build vocab from tokenized sentences

        filters rare words (min_count threshold) and optionally limits size
        """
        # count frequencies
        for sentence in tokenized_corpus:
            self.word_counts.update(sentence)

        # filter by min count
        filtered_words = [
            word for word, count in self.word_counts.items()
            if count >= self.min_count
        ]

        # sort by frequency
        filtered_words.sort(key=lambda w: self.word_counts[w], reverse=True)

        # limit vocab size if needed
        if self.max_vocab_size:
            filtered_words = filtered_words[:self.max_vocab_size]

        # create mappings
        for idx, word in enumerate(filtered_words):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self.vocab_size = len(self.word2idx)

        print(f"vocab: {self.vocab_size} words")
        print(f"total tokens: {sum(self.word_counts.values())}")
        print(f"unique words: {len(self.word_counts)}")

    def get_idx(self, word: str) -> int:
        """get index for word (-1 if not in vocab)"""
        return self.word2idx.get(word, -1)

    def get_word(self, idx: int) -> str:
        """get word for index"""
        return self.idx2word.get(idx, "<UNK>")
