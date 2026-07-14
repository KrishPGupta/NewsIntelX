"""
src/preprocessing.py

Text preprocessing utilities for the News Intelligence System.
Cleans raw article text (HTML, symbols, whitespace) and provides
tokenization / stopword removal for NLP modules that need clean tokens.
"""

import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Ensure required NLTK data is available (auto-downloads on first run,
# e.g. on a fresh deployment server that doesn't have it cached yet).
def _ensure_nltk_data():
    resources = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

_ensure_nltk_data()

STOPWORDS = set(stopwords.words("english"))


def clean_text(raw_text: str) -> str:
    """
    Clean raw article text for downstream NLP processing.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text
    text = re.sub(r"<[^>]+>", " ", text)

    html_entities = {
        "&amp;": "&", "&quot;": '"', "&apos;": "'",
        "&lt;": "<", "&gt;": ">", "&nbsp;": " ",
    }
    for entity, replacement in html_entities.items():
        text = text.replace(entity, replacement)

    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    """
    Tokenize cleaned text and remove stopwords + punctuation.
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
    sample = "<p>Apple &amp; Google announced a partnership today! Visit https://example.com for more.</p>"
    cleaned = clean_text(sample)
    print("Cleaned:", cleaned)
    print("Tokens:", tokenize_and_remove_stopwords(cleaned))
