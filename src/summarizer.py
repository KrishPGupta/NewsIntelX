"""
src/summarizer.py

Article summarization using Google's Gemini API (via the google-genai SDK).
Uses prompt engineering to produce concise, consistent summaries with
proper error handling for API failures.
"""

from google import genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME, MAX_SUMMARY_WORDS
from src.utils import call_with_retry

# Create the Gemini client once at import time
_client = genai.Client(api_key=GEMINI_API_KEY)


def summarize_article(text: str, max_words: int = MAX_SUMMARY_WORDS) -> str:
    """
    Generate a concise summary of a news article using Gemini.

    Args:
        text: The article text to summarize (cleaned or raw is fine —
              Gemini handles messy text well).
        max_words: Approximate maximum word count for the summary.

    Returns:
        A concise summary string. Returns an error message string
        (not an exception) if the API call fails, so the Streamlit UI
        can display it gracefully instead of crashing.
    """
    if not text or not text.strip():
        return "No article text provided to summarize."

    prompt = f"""You are a professional news editor. Summarize the following news article
in no more than {max_words} words. Requirements:
- Capture only the most important facts (who, what, when, where, why)
- Write in clear, neutral, third-person journalistic style
- Do not add opinions, commentary, or information not in the article
- Return ONLY the summary text, with no preamble like "Here is a summary"

Article:
\"\"\"
{text}
\"\"\"
"""

    try:
        response = call_with_retry(
            lambda: _client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
        )
        return response.text.strip()

    except Exception as e:
        # Catch-all for network errors, invalid API key, rate limits, etc.
        # We return a readable message instead of raising, so the Streamlit
        # app can show it directly to the user without crashing.
        return f"⚠️ Summarization failed: {str(e)}"


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python3 src/summarizer.py
    sample = (
        "Apple announced on Tuesday a major partnership with Microsoft to "
        "develop new artificial intelligence tools aimed at enterprise "
        "customers. The deal, valued at an estimated 2 billion dollars, "
        "focuses on cloud computing infrastructure and machine learning "
        "capabilities. Analysts say the partnership marks a significant "
        "shift in how the two historic rivals collaborate in the AI era. "
        "Shares of both companies rose following the announcement."
    )
    summary = summarize_article(sample, max_words=30)
    print("Summary:", summary)
