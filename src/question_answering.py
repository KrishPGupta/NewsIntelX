"""
src/question_answering.py

Question answering over article text using Gemini, with prompt engineering
designed to keep answers grounded in the article and avoid hallucination.
"""

from google import genai
from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME
from src.utils import call_with_retry

_client = genai.Client(api_key=GEMINI_API_KEY)


def answer_question(article_text: str, question: str) -> str:
    """
    Answer a user's question based strictly on the given article text.

    Args:
        article_text: The full article text to ground the answer in.
        question: The user's natural-language question.

    Returns:
        A grounded answer string, or a clear "not found" message if the
        article doesn't contain relevant information. Returns a readable
        error message (not an exception) if the API call fails.
    """
    if not article_text or not article_text.strip():
        return "No article text available to answer questions from."
    if not question or not question.strip():
        return "Please enter a question."

    prompt = f"""You are a precise question-answering assistant. Answer the question
using ONLY information found in the article below.

Rules:
- If the answer is explicitly stated or reasonably inferable from the article, answer clearly and concisely.
- If the article does NOT contain enough information to answer, respond exactly with:
  "This information is not available in the article."
- Do not use outside knowledge, assumptions, or make up facts.
- Keep the answer to 2-3 sentences maximum.

Article:
\"\"\"
{article_text}
\"\"\"

Question: {question}

Answer:"""

    try:
        response = call_with_retry(
            lambda: _client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
            )
        )
        return response.text.strip()

    except Exception as e:
        return f"⚠️ Question answering failed: {str(e)}"


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python3 -m src.question_answering
    sample = (
        "Apple announced on Tuesday a major partnership with Microsoft to "
        "develop new artificial intelligence tools aimed at enterprise "
        "customers. The deal, valued at an estimated 2 billion dollars, "
        "focuses on cloud computing infrastructure and machine learning "
        "capabilities. Analysts say the partnership marks a significant "
        "shift in how the two historic rivals collaborate in the AI era."
    )

    questions = [
        "How much is the partnership worth?",
        "What sport does this article discuss?",  # Should trigger "not available"
    ]

    for q in questions:
        answer = answer_question(sample, q)
        print(f"Q: {q}\nA: {answer}\n")
