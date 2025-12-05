"""
custom tokenizer for king james version bible text
handles archaic english (1611) with special patterns
"""
import re
from typing import List


class KJVTokenizer:
    """tokenizer for kjv bible text

    handles early modern english with archaic forms like:
    - pronouns: thee, thou, thy
    - contractions: 'tis, 'twas, o'er
    """

    def __init__(self):
        # kjv-specific contractions to expand
        self.contractions = {
            "'tis": "it is",
            "'twas": "it was",
            "o'er": "over",
            "e'er": "ever",
            "ne'er": "never",
            "thro'": "through",
        }

        # remove possessives for cleaner embeddings
        self.archaic_patterns = [
            (r"\b(god|lord|christ|jesus)'s\b", r"\1"),
        ]

    def tokenize(self, text: str) -> List[str]:
        """tokenize text into words

        steps:
        1. lowercase
        2. expand contractions
        3. remove verse refs (e.g. john:3:16)
        4. extract words w/ regex
        5. filter short tokens
        """
        if not text:
            return []

        text = text.lower()

        # expand contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # apply patterns
        for pattern, replacement in self.archaic_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # remove verse references
        text = re.sub(r'\b[A-Za-z]+:\d+:\d+\b', '', text)
        text = re.sub(r'\b[A-Za-z]+:\d+\b', '', text)

        # extract words (keeps apostrophes in contractions)
        tokens = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", text)

        # filter short tokens
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def tokenize_sentences(self, text: str) -> List[List[str]]:
        """split on sentence boundaries and tokenize"""
        sentences = re.split(r'[.!?]+\s+', text)
        return [self.tokenize(sent) for sent in sentences if sent.strip()]
