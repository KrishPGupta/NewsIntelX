"""
src/keyword_extraction.py

Keyword/phrase extraction using KeyBERT, which uses sentence embeddings
to find phrases that are semantically representative of the document —
more accurate than frequency-based methods like plain TF-IDF.
"""

from keybert import KeyBERT

# Initialize the model once at import time (avoids reloading per call).
# First run will download the underlying sentence-transformer model.
_kw_model = KeyBERT()


def extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """
    Extract top keywords/phrases from article text.

    Args:
        text: Cleaned article text (ideally passed through preprocessing.clean_text first).
        top_n: Number of keywords/phrases to return.

    Returns:
        A list of keyword/phrase strings, ranked by relevance (most relevant first).
    """
    if not text or not text.strip():
        return []

    # keyphrase_ngram_range=(1, 2) allows both single words and two-word phrases
    # stop_words="english" removes common filler words from consideration
    results = _kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
    )

    # results is a list of (keyword, score) tuples; we just want the keywords
    return [keyword for keyword, score in results]


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python3 src/keyword_extraction.py
    sample = (
        "Apple announced a major partnership with Microsoft to develop new "
        "artificial intelligence tools for enterprise customers. The deal "
        "focuses on cloud computing infrastructure and machine learning "
        "capabilities, marking a significant shift in the tech industry."
    )
    keywords = extract_keywords(sample)
    print("Keywords:", keywords)
