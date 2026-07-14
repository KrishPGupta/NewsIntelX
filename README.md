# 📰 Intelligent News Summarization & Sentiment Analysis System

An end-to-end NLP pipeline that summarizes news articles, detects sentiment, extracts named entities and keywords, and answers questions about the article — all through an interactive Streamlit dashboard powered by Google's Gemini API.

## Features

- **Text Preprocessing** — HTML cleanup, tokenization, and stopword removal (NLTK)
- **Summarization** — Concise, factual article summaries via Gemini, using prompt engineering to control tone and length
- **Sentiment Analysis** — Fast, rule-based sentiment classification via VADER
- **Named Entity Recognition** — Extracts People, Organizations, and Places using spaCy
- **Keyword Extraction** — Semantically relevant keyphrases via KeyBERT (sentence-embedding based, not just frequency counts)
- **Question Answering** — Ask natural-language questions about the article; answers are grounded strictly in the article text to avoid hallucination
- **Interactive Dashboard** — Built with Streamlit, with graceful error handling throughout

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.13 |
| Web UI | Streamlit |
| LLM (Summarization + Q&A) | Google Gemini API (`google-genai` SDK) |
| Sentiment | VADER (vaderSentiment) |
| NER | spaCy (`en_core_web_sm`) |
| Preprocessing | NLTK |
| Keyword Extraction | KeyBERT |

## Project Structure
NewsIntelX/ ├── app.py # Streamlit entry point (UI layer only) ├── src/ │ ├── preprocessing.py # Text cleaning, tokenization │ ├── summarizer.py # Gemini-based summarization │ ├── sentiment.py # VADER sentiment analysis │ ├── entity_extraction.py # spaCy NER │ ├── keyword_extraction.py # KeyBERT keyword extraction │ ├── question_answering.py # Gemini-based grounded Q&A │ └── utils.py # Shared retry logic for API calls ├── config/ │ └── settings.py # Loads API key & constants from .env ├── data/ │ └── sample_articles/ # A few example articles for quick testing ├── requirements.txt └── README.md

## Setup & Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd NewsIntelX
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 4. Configure your API key
Create a `.env` file in the project root:
GEMINI_API_KEY=your_key_here


### 5. Run the app
```bash
python3 -m streamlit run app.py
```

## Dataset

Sample articles for testing are drawn from the [BBC News Summary dataset](https://www.kaggle.com/datasets/pariza/bbc-news-summary) (Kaggle), which pairs full BBC articles with human-written extractive summaries across 5 categories (business, entertainment, politics, sport, tech). The full dataset is not included in this repo — see the link to download it yourself if needed.

## Known Limitations

- **VADER sentiment is lexicon-based, not context-aware.** It can misjudge tone on nuanced political or conflict-related news (e.g. classifying an assertive geopolitical statement as "Positive" due to confident language, even when the underlying story is tense).
- Gemini model availability changes frequently; the model name in `config/settings.py` may need updating over time as Google deprecates older models.

## Future Improvements

- Support direct URL input (auto-fetch + parse article from a link, not just pasted text)
- Multi-article comparison / batch analysis mode
- Swap VADER for a transformer-based sentiment model for better nuance on complex topics
- Add caching to avoid redundant Gemini calls on repeated Q&A within the same session
- Deploy to Streamlit Community Cloud for a live public demo link
- Add automated tests (`tests/`) for each module

## Author

Built by Krish Gupta as a college NLP/LLM project.
