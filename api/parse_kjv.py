"""
parse kjv bible and generate verse embeddings
cs686: natural language processing - university of san francisco

author: joshua vasquez
date: december 2024
"""
import re
import pickle
import numpy as np
from typing import List, Dict
from nlp import KJVTokenizer, Vocabulary, Word2Vec


class KJVParser:
    """parse kjv bible text into structured verses"""

    def __init__(self):
        # kjv format: "BookName Chapter:Verse\tText"
        # example: "Genesis 1:1\tIn the beginning..."
        self.verse_pattern = re.compile(r'^([0-9]{0,1}\s?[A-Za-z]+)\s+(\d+):(\d+)\t(.+)$')

    def parse(self, kjv_path: str) -> List[Dict]:
        """parse kjv text file into verse records

        returns list of dicts with: book, chapter, verse_num, text, reference
        """
        verses = []

        with open(kjv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('King James'):
                    continue

                match = self.verse_pattern.match(line)
                if match:
                    book = match.group(1).strip()
                    chapter = int(match.group(2))
                    verse_num = int(match.group(3))
                    text = match.group(4).strip()

                    reference = f"{book} {chapter}:{verse_num}"

                    verses.append({
                        'book': book,
                        'chapter': chapter,
                        'verse_num': verse_num,
                        'text': text,
                        'reference': reference,
                    })

        return verses


class VerseEmbedder:
    """generate embeddings for verses using word2vec"""

    def __init__(self, model: Word2Vec, vocab: Vocabulary, tokenizer: KJVTokenizer):
        self.model = model
        self.vocab = vocab
        self.tokenizer = tokenizer

    def embed_verse(self, text: str) -> np.ndarray:
        """generate embedding for verse by averaging word vectors"""
        # tokenize verse
        tokens = self.tokenizer.tokenize(text)

        # convert to indices
        indices = [self.vocab.get_idx(token) for token in tokens]
        indices = [idx for idx in indices if idx != -1]

        # get sentence vector (mean of word vectors)
        return self.model.get_sentence_vector(indices)

    def embed_verses(self, verses: List[Dict]) -> List[Dict]:
        """add embeddings to all verses"""
        for i, verse in enumerate(verses):
            verse['embedding'] = self.embed_verse(verse['text'])

            if (i + 1) % 1000 == 0:
                print(f"embedded {i+1}/{len(verses)} verses")

        return verses


def generate_verse_embeddings(
    kjv_path: str = "../data/kjv.txt",
    model_path: str = "../word2vec_kjv.npz",
    vocab_path: str = "../word2vec_kjv_vocab.pkl",
) -> List[Dict]:
    """complete pipeline: parse kjv and generate embeddings"""
    print("="*60)
    print("kjv verse embedding generation")
    print("="*60)

    # parse kjv
    print("\n[1/3] parsing kjv...")
    parser = KJVParser()
    verses = parser.parse(kjv_path)
    print(f"parsed {len(verses)} verses")

    # load word2vec model
    print("\n[2/3] loading word2vec model...")
    model = Word2Vec(vocab_size=1, embedding_dim=100)
    model.load(model_path)

    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"loaded model: {vocab.vocab_size} words, {model.embedding_dim} dims")

    # generate embeddings
    print("\n[3/3] generating embeddings...")
    tokenizer = KJVTokenizer()
    embedder = VerseEmbedder(model, vocab, tokenizer)
    verses = embedder.embed_verses(verses)

    print("\n" + "="*60)
    print("embedding generation complete!")
    print("="*60)

    return verses


if __name__ == "__main__":
    # generate embeddings
    verses = generate_verse_embeddings(
        kjv_path="../data/kjv.txt",
        model_path="../word2vec_kjv.npz",
        vocab_path="../word2vec_kjv_vocab.pkl"
    )

    # print sample
    print("\n" + "="*60)
    print("sample verses with embeddings")
    print("="*60)

    for i in range(min(3, len(verses))):
        v = verses[i]
        print(f"\n{v['reference']}:")
        print(f"  text: {v['text'][:80]}...")
        print(f"  embedding shape: {v['embedding'].shape}")
        print(f"  embedding norm: {np.linalg.norm(v['embedding']):.4f}")
