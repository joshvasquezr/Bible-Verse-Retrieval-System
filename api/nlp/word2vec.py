"""
skip-gram word2vec w/ negative sampling
implemented from scratch using only numpy

based on mikolov et al. (2013) papers
"""
import numpy as np
from collections import Counter
from typing import List, Tuple


class Word2Vec:
    """skip-gram word2vec w/ negative sampling

    architecture:
        input (one-hot) -> W1 (embeddings) -> W2 (context vectors) -> output

    objective:
        maximize: log σ(v_context · v_center) for true context words
        minimize: log σ(v_neg · v_center) for k negative samples

    where σ(x) = 1/(1 + exp(-x))

    negative sampling uses smoothed unigram distribution:
        P(w) ∝ count(w)^0.75
    this gives rare words higher sampling prob than raw frequency

    weight matrices:
        W1: vocab_size × embed_dim (these are the final word embeddings)
        W2: embed_dim × vocab_size (context prediction vectors)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        window_size: int = 5,
        num_negative_samples: int = 5,
        learning_rate: float = 0.025,
        min_learning_rate: float = 0.0001,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negative_samples = num_negative_samples
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.min_learning_rate = min_learning_rate

        # initialize weight matrices w/ small random values
        # W1: word embeddings (each row is a word vector)
        self.W1 = (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim

        # W2: context vectors (each column is a context vector)
        self.W2 = (np.random.rand(embedding_dim, vocab_size) - 0.5) / vocab_size

        # will be set during training
        self.negative_sampling_probs = None

    def set_negative_sampling_distribution(self, word_counts: Counter):
        """create smoothed unigram distribution for negative sampling

        uses P(w) = f(w)^0.75 / sum(f^0.75)
        the 0.75 exponent is from the original word2vec paper
        """
        probs = np.zeros(self.vocab_size)

        for word, idx in word_counts.items():
            if idx < self.vocab_size:
                probs[idx] = word_counts[word]

        # apply 0.75 power smoothing
        probs = np.power(probs, 0.75)
        probs = probs / np.sum(probs)

        self.negative_sampling_probs = probs

    def get_negative_samples(self, target_idx: int, num_samples: int) -> np.ndarray:
        """sample k negative words (excluding target)"""
        if self.negative_sampling_probs is None:
            samples = np.random.randint(0, self.vocab_size, size=num_samples)
        else:
            samples = np.random.choice(
                self.vocab_size,
                size=num_samples,
                p=self.negative_sampling_probs
            )

        # remove target if sampled
        samples = samples[samples != target_idx]

        # resample if needed
        while len(samples) < num_samples:
            extra = np.random.choice(self.vocab_size, size=1)
            if extra[0] != target_idx:
                samples = np.append(samples, extra)

        return samples[:num_samples]

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """sigmoid w/ clipping for numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def train_pair(self, center_idx: int, context_idx: int) -> float:
        """train on one (center, context) pair using negative sampling

        algorithm:
        1. get center word embedding: h = W1[center]
        2. positive sample: maximize σ(h · W2[:, context])
        3. negative samples: minimize σ(h · W2[:, neg_i]) for k negatives
        4. update W1[center] and W2 columns via sgd
        """
        # get center embedding
        h = self.W1[center_idx]

        # positive sample
        u_o = self.W2[:, context_idx]
        score_pos = np.dot(h, u_o)
        pred_pos = self.sigmoid(score_pos)

        # loss: -log(σ(score))
        loss = -np.log(pred_pos + 1e-10)

        # gradients for positive
        grad_pos = (pred_pos - 1) * u_o
        grad_w2_pos = (pred_pos - 1) * h

        # negative samples
        neg_indices = self.get_negative_samples(context_idx, self.num_negative_samples)
        grad_neg = np.zeros(self.embedding_dim)

        for neg_idx in neg_indices:
            u_neg = self.W2[:, neg_idx]
            score_neg = np.dot(h, u_neg)
            pred_neg = self.sigmoid(score_neg)

            # loss: -log(1 - σ(score))
            loss += -np.log(1 - pred_neg + 1e-10)

            # gradients for negative
            grad_neg += pred_neg * u_neg
            self.W2[:, neg_idx] -= self.learning_rate * pred_neg * h

        # update weights
        self.W1[center_idx] -= self.learning_rate * (grad_pos + grad_neg)
        self.W2[:, context_idx] -= self.learning_rate * grad_w2_pos

        return loss

    def train(self, tokenized_corpus: List[List[str]], vocabulary, epochs: int = 5, verbose: bool = True):
        """train word2vec on corpus

        generates (center, context) pairs using sliding window
        then trains w/ negative sampling

        example w/ window=2:
            "the lord is my shepherd"
            -> (lord, the), (lord, is), (is, lord), (is, my), etc.
        """
        # setup negative sampling
        vocab_counts = Counter()
        for word, idx in vocabulary.word2idx.items():
            vocab_counts[idx] = vocabulary.word_counts[word]
        self.set_negative_sampling_distribution(vocab_counts)

        # generate training pairs
        training_pairs = []
        for sentence in tokenized_corpus:
            indices = [vocabulary.get_idx(word) for word in sentence]
            indices = [idx for idx in indices if idx != -1]

            # sliding window
            for i, center_idx in enumerate(indices):
                # dynamic window size (helps w/ quality)
                window = np.random.randint(1, self.window_size + 1)

                start = max(0, i - window)
                end = min(len(indices), i + window + 1)

                for j in range(start, end):
                    if i != j:
                        training_pairs.append((center_idx, indices[j]))

        total_pairs = len(training_pairs)
        print(f"training pairs: {total_pairs}")

        # training loop
        for epoch in range(epochs):
            np.random.shuffle(training_pairs)
            total_loss = 0
            batch_size = 10000

            for i, (center, context) in enumerate(training_pairs):
                loss = self.train_pair(center, context)
                total_loss += loss

                # linear lr decay
                progress = (epoch * total_pairs + i) / (epochs * total_pairs)
                self.learning_rate = max(
                    self.min_learning_rate,
                    self.initial_learning_rate * (1 - progress)
                )

                # progress updates
                if verbose and (i + 1) % batch_size == 0:
                    avg_loss = total_loss / (i + 1)
                    print(f"epoch {epoch+1}/{epochs} - pair {i+1}/{total_pairs} - "
                          f"loss: {avg_loss:.4f} - lr: {self.learning_rate:.6f}")

            avg_loss = total_loss / total_pairs
            if verbose:
                print(f"epoch {epoch+1}/{epochs} done - avg loss: {avg_loss:.4f}")

    def get_word_vector(self, word_idx: int) -> np.ndarray:
        """get embedding vector for word"""
        return self.W1[word_idx]

    def get_sentence_vector(self, word_indices: List[int]) -> np.ndarray:
        """get sentence embedding by averaging word vectors"""
        if not word_indices:
            return np.zeros(self.embedding_dim)

        vectors = [self.W1[idx] for idx in word_indices if idx < self.vocab_size]

        if not vectors:
            return np.zeros(self.embedding_dim)

        return np.mean(vectors, axis=0)

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """compute cosine similarity between vectors

        cos(θ) = (A · B) / (||A|| × ||B||)

        range: [-1, 1]
            1.0 = same direction (very similar)
            0.0 = orthogonal (unrelated)
           -1.0 = opposite (dissimilar)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def most_similar(self, word_idx: int, vocabulary, top_k: int = 10) -> List[Tuple[str, float]]:
        """find k most similar words using cosine similarity"""
        if word_idx >= self.vocab_size or word_idx < 0:
            return []

        target_vec = self.get_word_vector(word_idx)

        similarities = []
        for idx in range(self.vocab_size):
            if idx != word_idx:
                vec = self.get_word_vector(idx)
                sim = self.cosine_similarity(target_vec, vec)
                word = vocabulary.get_word(idx)
                similarities.append((word, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def save(self, filepath: str):
        """save model weights"""
        np.savez(filepath, W1=self.W1, W2=self.W2)
        print(f"saved to {filepath}")

    def load(self, filepath: str):
        """load model weights"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.W2 = data['W2']
        self.vocab_size, self.embedding_dim = self.W1.shape
        print(f"loaded from {filepath}")
