"""
app.py

Streamlit entry point for the News Intelligence System.
This file handles ONLY the UI layer — all logic lives in src/ modules.
"""

import streamlit as st
from src.preprocessing import clean_text
from src.summarizer import summarize_article
from src.sentiment import analyze_sentiment
from src.entity_extraction import extract_entities
from src.keyword_extraction import extract_keywords
from src.question_answering import answer_question

# --- Page configuration ---
st.set_page_config(
    page_title="News Intelligence System",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Intelligent News Summarization & Sentiment Analysis")
st.caption("Paste a news article below to get a summary, sentiment, entities, keywords, and Q&A.")

# --- Session state setup ---
if "analyzed_text" not in st.session_state:
    st.session_state.analyzed_text = None

MIN_WORDS_REQUIRED = 20  # Below this, NLP models produce unreliable/meaningless output

# --- Input section ---
article_text = st.text_area(
    "Paste article text here:",
    height=250,
    placeholder="Paste the full text of a news article...",
)

analyze_clicked = st.button("🔍 Analyze Article", type="primary")

# --- Run full analysis on click ---
if analyze_clicked:
    if not article_text or not article_text.strip():
        st.warning("Please paste some article text before analyzing.")
        st.session_state.analyzed_text = None
    elif len(article_text.strip().split()) < MIN_WORDS_REQUIRED:
        st.warning(
            f"That text looks too short to analyze meaningfully "
            f"(minimum ~{MIN_WORDS_REQUIRED} words). Please paste a fuller article."
        )
        st.session_state.analyzed_text = None
    else:
        try:
            with st.spinner("Cleaning text..."):
                cleaned = clean_text(article_text)
            if not cleaned:
                st.error("Couldn't extract any readable text from the input. Please check the content and try again.")
                st.session_state.analyzed_text = None
            else:
                st.session_state.analyzed_text = cleaned
        except Exception as e:
            st.error(f"Something went wrong while processing the text: {e}")
            st.session_state.analyzed_text = None

# --- Display results if we have an analyzed article ---
if st.session_state.analyzed_text:
    cleaned = st.session_state.analyzed_text

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Summary")
        try:
            with st.spinner("Generating summary with Gemini..."):
                summary = summarize_article(cleaned)
            st.write(summary)
        except Exception as e:
            st.error(f"Summarization unavailable right now: {e}")

    with col2:
        st.subheader("💬 Sentiment")
        try:
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = analyze_sentiment(cleaned)
            label = sentiment_result["label"]
            compound = sentiment_result["scores"]["compound"]

            if label == "Positive":
                st.success(f"{label} (score: {compound:.2f})")
            elif label == "Negative":
                st.error(f"{label} (score: {compound:.2f})")
            else:
                st.info(f"{label} (score: {compound:.2f})")
        except Exception as e:
            st.error(f"Sentiment analysis unavailable right now: {e}")

    st.divider()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🏷️ Named Entities")
        try:
            with st.spinner("Extracting entities..."):
                entities = extract_entities(cleaned)
            st.markdown("**People:** " + (", ".join(entities["people"]) if entities["people"] else "_None found_"))
            st.markdown("**Organizations:** " + (", ".join(entities["organizations"]) if entities["organizations"] else "_None found_"))
            st.markdown("**Places:** " + (", ".join(entities["places"]) if entities["places"] else "_None found_"))
        except Exception as e:
            st.error(f"Entity extraction unavailable right now: {e}")

    with col4:
        st.subheader("🔑 Keywords")
        try:
            with st.spinner("Extracting keywords..."):
                keywords = extract_keywords(cleaned)
            if keywords:
                tags = " ".join([f"`{kw}`" for kw in keywords])
                st.markdown(tags)
            else:
                st.write("_No keywords found_")
        except Exception as e:
            st.error(f"Keyword extraction unavailable right now: {e}")

    st.divider()

    st.subheader("❓ Ask a Question About This Article")
    question = st.text_input("Your question:", placeholder="e.g. What did the company announce?")
    ask_clicked = st.button("Get Answer")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Thinking..."):
                    answer = answer_question(cleaned, question)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Couldn't get an answer right now: {e}")

else:
    st.info("👆 Paste an article above and click 'Analyze Article' to get started.")
