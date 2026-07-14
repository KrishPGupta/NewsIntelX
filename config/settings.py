"""
config/settings.py

Centralized configuration loader for the News Intelligence System.
Works both locally (reads from a .env file) and when deployed on
Streamlit Community Cloud (reads from st.secrets), so the same
codebase runs in both environments without changes.
"""

import os
from dotenv import load_dotenv

# Load variables from .env for local development
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# If not found locally, try Streamlit's secrets manager (used when deployed)
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        pass

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is missing. "
        "For local development, create a .env file with:\n"
        "GEMINI_API_KEY=your_key_here\n"
        "For Streamlit Cloud, add it under Settings > Secrets."
    )

# Central place for other tunable constants (used later by other modules)
GEMINI_MODEL_NAME = "gemini-3.5-flash"   # Fast + cheap, good for summarization/Q&A
MAX_SUMMARY_WORDS = 150
