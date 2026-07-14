"""
config/settings.py

Centralized configuration loader for the News Intelligence System.
Loads secrets (like API keys) from a local .env file so they never
get hardcoded into source code or committed to version control.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file in the project root into the environment
load_dotenv()

# Gemini API key, required for summarization and Q&A modules
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is missing. "
        "Create a .env file in the project root with:\n"
        "GEMINI_API_KEY=your_key_here"
    )

# Central place for other tunable constants (used later by other modules)
GEMINI_MODEL_NAME = "gemini-3.5-flash"   # Fast + cheap, good for summarization/Q&A
MAX_SUMMARY_WORDS = 150
