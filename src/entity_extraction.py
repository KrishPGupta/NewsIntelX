"""
src/entity_extraction.py

Named Entity Recognition (NER) using spaCy's pretrained English model.
Extracts People, Organizations, and Places (locations/countries/cities)
from article text for display in the dashboard.
"""

import spacy

# Load the small English model once at import time (avoids reloading per call)
_nlp = spacy.load("en_core_web_sm")


def extract_entities(text: str) -> dict:
    """
    Extract named entities from article text, grouped by category.

    Args:
        text: Cleaned article text (ideally passed through preprocessing.clean_text first).

    Returns:
        A dictionary with three keys, each a list of unique entity strings:
            - "people": PERSON entities
            - "organizations": ORG entities
            - "places": GPE (countries/cities/states) and LOC entities
    """
    if not text or not text.strip():
        return {"people": [], "organizations": [], "places": []}

    doc = _nlp(text)

    people = set()
    organizations = set()
    places = set()

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            people.add(ent.text.strip())
        elif ent.label_ == "ORG":
            organizations.add(ent.text.strip())
        elif ent.label_ in ("GPE", "LOC"):
            places.add(ent.text.strip())

    return {
        "people": sorted(people),
        "organizations": sorted(organizations),
        "places": sorted(places),
    }


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python3 src/entity_extraction.py
    sample = (
        "Apple CEO Tim Cook announced a new partnership with Microsoft "
        "during a press conference in San Francisco, California. "
        "The deal was praised by analysts in New York and London."
    )
    result = extract_entities(sample)
    print("People:", result["people"])
    print("Organizations:", result["organizations"])
    print("Places:", result["places"])
