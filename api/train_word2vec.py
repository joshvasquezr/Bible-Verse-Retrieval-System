"""
training script for word2vec on kjv bible
cs686: natural language processing - university of san francisco

author: joshua vasquez
date: december 2024
"""
import pickle
from typing import Tuple
from nlp import KJVTokenizer, Vocabulary, Word2Vec


def load_kjv_corpus(filepath: str = "../data/kjv.txt") -> str:
    """load kjv bible text"""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def train_word2vec_on_kjv(
    kjv_path: str = "../data/kjv.txt",
    embedding_dim: int = 100,
    window_size: int = 5,
    min_count: int = 5,
    epochs: int = 5,
    save_path: str = "word2vec_kjv.npz",
) -> Tuple[Word2Vec, Vocabulary, KJVTokenizer]:
    """complete training pipeline for kjv word2vec"""
    print("="*60)
    print("word2vec training - kjv bible")
    print("="*60)

    # load corpus
    print("\n[1/5] loading corpus...")
    corpus_text = load_kjv_corpus(kjv_path)
    print(f"loaded {len(corpus_text)} chars")

    # tokenize
    print("\n[2/5] tokenizing...")
    tokenizer = KJVTokenizer()
    tokenized_corpus = tokenizer.tokenize_sentences(corpus_text)
    print(f"tokenized: {len(tokenized_corpus)} sentences")
    total_tokens = sum(len(sent) for sent in tokenized_corpus)
    print(f"total tokens: {total_tokens}")

    # build vocab
    print("\n[3/5] building vocab...")
    vocab = Vocabulary(min_count=min_count)
    vocab.build_vocab(tokenized_corpus)

    # train model
    print("\n[4/5] training model...")
    model = Word2Vec(
        vocab_size=vocab.vocab_size,
        embedding_dim=embedding_dim,
        window_size=window_size,
    )
    model.train(tokenized_corpus, vocab, epochs=epochs)

    # save
    print("\n[5/5] saving...")
    model.save(save_path)

    vocab_path = save_path.replace('.npz', '_vocab.pkl')
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"vocab saved to {vocab_path}")

    print("\n" + "="*60)
    print("training complete!")
    print("="*60)

    return model, vocab, tokenizer


if __name__ == "__main__":
    # train model
    model, vocab, tokenizer = train_word2vec_on_kjv(
        kjv_path="../data/kjv.txt",
        embedding_dim=100,
        window_size=5,
        min_count=5,
        epochs=3,
        save_path="word2vec_kjv.npz"
    )

    # test similarities
    print("\n" + "="*60)
    print("testing word similarities")
    print("="*60)

    test_words = ["god", "lord", "love", "faith", "peace", "sin"]
    for word in test_words:
        idx = vocab.get_idx(word)
        if idx != -1:
            print(f"\nmost similar to '{word}':")
            similar = model.most_similar(idx, vocab, top_k=5)
            for w, score in similar:
                print(f"  {w}: {score:.4f}")
