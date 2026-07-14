"""
src/preprocessing.py

Text preprocessing utilities for the News Intelligence System.
Cleans raw article text (HTML, symbols, whitespace) and provides
tokenization / stopword removal for NLP modules that need clean tokens.
"""

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Load English stopwords once at import time (avoids reloading on every call)
STOPWORDS = set(stopwords.words("english"))


def clean_text(raw_text: str) -> str:
    """
    Clean raw article text for downstream NLP processing.

    Steps:
    - Remove HTML tags
    - Decode common HTML entities (&amp;, &quot;, etc.)
    - Remove URLs
    - Collapse excessive whitespace
    - Strip leading/trailing whitespace

    Args:
        raw_text: The original, possibly messy article text.

    Returns:
        A cleaned string, safe to pass to summarization, sentiment,
        NER, or keyword extraction modules.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # Remove HTML tags like <div>, <p>, <br/>
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode a few common HTML entities without pulling in a heavy library
    html_entities = {
        "&amp;": "&", "&quot;": '"', "&apos;": "'",
        "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Collapse multiple whitespace/newlines into a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    """
    Tokenize cleaned text and remove stopwords + punctuation.
    Useful for keyword extraction or any bag-of-words style analysis.

    Args:
        text: Cleaned article text (run through clean_text first).

    Returns:
        A list of lowercase tokens with stopwords and punctuation removed.
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())

    filtered = [
        token for token in tokens
        if token not in STOPWORDS and token not in string.punctuation
    ]

    return filtered


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python3 src/preprocessing.py
    sample = "<p>Apple &amp; Google announced a partnership today! Visit https://example.com for more.</p>"
    cleaned = clean_text(sample)
    print("Cleaned:", cleaned)
    print("Tokens:", tokenize_and_remove_stopwords(cleaned))
