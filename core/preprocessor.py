"""
core/preprocessor.py
--------------------
Shared NLP preprocessing — must stay aligned with training pipeline.
"""

from __future__ import annotations

import os
import re

from nltk.stem import PorterStemmer

_PRESERVE = {
    "breaking", "exclusive", "shocking", "bombshell", "leaked", "leak",
    "secret", "exposed", "banned", "censored", "urgent", "alert",
    "share", "forward", "viral", "sheeple", "globalist", "globalists",
    "cabal", "whistleblower", "insider", "unnamed", "anonymous",
    "deep", "state", "mainstream", "lamestream", "hoax", "scam",
    "microchip", "microchips", "plandemic", "scamdemic", "qanon",
    "conspiracy", "coverup", "cover-up", "suppressed", "silenced",
    "debunked", "factcheck", "truth", "patriot", "patriots",
}

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def _load_stopwords() -> set:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.abspath(os.path.join(base_dir, "..", "nltk_data"))

    if os.path.isdir(bundled):
        import nltk
        nltk.data.path.insert(0, bundled)
        from nltk.corpus import stopwords
        words = set(stopwords.words("english"))
    else:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            words = set(stopwords.words("english"))
        except Exception:
            from core.nlp_lite import get_stopwords
            words = get_stopwords()
            return words

    words -= {
        "no", "not", "never", "nor", "neither", "without", "against",
        "but", "however", "although", "though", "yet",
    }
    return words


_stop_words = _load_stopwords()
_stemmer = PorterStemmer()


def preprocess(text: str) -> str:
    """Clean and normalise news text for TF-IDF."""
    text = str(text).lower().strip()
    text = _URL_RE.sub(" ", text)

    contractions = {
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'re": " are",
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'m": " am",
    }
    for short, full in contractions.items():
        text = text.replace(short, full)

    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    tokens = text.split()

    cleaned = []
    for t in tokens:
        if len(t) <= 1:
            continue
        if t in _PRESERVE:
            cleaned.append(_stemmer.stem(t))
        elif t not in _stop_words:
            cleaned.append(_stemmer.stem(t))

    return " ".join(cleaned)
