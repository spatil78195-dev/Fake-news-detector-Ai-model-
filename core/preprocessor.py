"""
core/preprocessor.py
--------------------
Shared NLP preprocessing pipeline used by the prediction API and retraining
scheduler.  MUST stay in sync with train_model.py's preprocess() function.

Improvements over v1:
  - Negation words (no, not, never …) are kept — they are discriminative
  - Fake-news indicator words (sheeple, globalist, bombshell …) preserved
    before stemming so the heuristic layer can also see them
  - Single-character tokens dropped after stemming
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

_stop_words: set = set(stopwords.words("english"))

_stop_words -= {
    "no", "not", "never", "nor", "neither", "without", "against",
    "but", "however", "although", "though", "yet",
}

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

_stemmer: PorterStemmer = PorterStemmer()

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def preprocess(text: str) -> str:
    """
    Full NLP preprocessing pipeline (v2).

    Steps
    -----
    1. Lowercase
    2. Strip URLs
    3. Remove non-alpha characters
    4. Tokenise on whitespace
    5. Remove stop-words (keeping negations + indicator words)
    6. Porter-stem each token
    7. Drop single-character tokens
    8. Re-join and return
    """
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
