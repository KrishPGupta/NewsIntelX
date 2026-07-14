"""
src/sentiment.py

Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).
Rule-based and lexicon-driven — fast, free, and doesn't require an API call,
making it ideal for a quick sentiment tag alongside LLM-based summarization.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize analyzer once at import time (avoids re-loading the lexicon per call)
_analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a given piece of text.

    Args:
        text: Cleaned article text (ideally passed through preprocessing.clean_text first).

    Returns:
        A dictionary with:
            - "label": one of "Positive", "Neutral", "Negative"
            - "scores": the raw VADER polarity scores (neg, neu, pos, compound)
    """
    if not text or not text.strip():
        return {"label": "Neutral", "scores": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}}

    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    # Standard VADER thresholds for compound score classification
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"label": label, "scores": scores}


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python3 src/sentiment.py
    samples = [
        "The company reported record profits and investors are thrilled.",
        "The disaster caused widespread devastation and grief across the region.",
        "The meeting was held on Tuesday to discuss quarterly reports.",
    ]
    for s in samples:
        result = analyze_sentiment(s)
        print(f"Text: {s}\n -> {result['label']} (compound={result['scores']['compound']})\n")
