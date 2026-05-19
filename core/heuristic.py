"""
core/heuristic.py
-----------------
Rule-based heuristic analyzer for fake news detection.
Works on RAW text (before NLP preprocessing) to detect
language patterns strongly associated with misinformation.

Returns a fake_bias score (-50 to +50):
  Positive  → more likely fake
  Negative  → more likely real
"""

import re

# ── Fake-news signal patterns (weight = amount added to fake score) ────────────
_FAKE_PATTERNS = [
    # Share / viral / urgency
    (r'share\s+before', 22),
    (r'forward\s+to\s+everyone', 22),
    (r'before\s+(they\s+)?delete', 20),
    (r'before\s+censorship', 20),
    (r'going\s+viral', 8),

    # Wake-up calls
    (r'wake\s+up\s+sheeple', 28),
    (r'\bsheeple\b', 25),
    (r'wake\s+up\b', 10),
    (r'open\s+your\s+eyes', 12),
    (r'do\s+your\s+own\s+research', 16),

    # Suppression / cover-up
    (r'mainstream\s+media\s+won.?t\s+tell', 22),
    (r'lamestream\s+media', 24),
    (r'they\s+don.?t\s+want\s+you\s+to\s+know', 20),
    (r'they\s+are\s+hiding', 20),
    (r"they'?re\s+hiding", 20),
    (r'shadow.?ban', 16),
    (r'fact.?checker', 10),
    (r'censored\s+by', 16),
    (r'banned\s+from', 12),

    # Anonymous / mysterious sources
    (r'cannot\s+be\s+named\s+for\s+their\s+safety', 24),
    (r'unnamed\s+source', 14),
    (r'anonymous\s+insider', 16),
    (r'trusted\s+insider', 16),
    (r'deep\s+state\s+operative', 22),

    # Conspiracy markers
    (r'\bdeep\s+state\b', 20),
    (r'\bglobalist', 20),
    (r'new\s+world\s+order', 24),
    (r'\bcabal\b', 22),
    (r'population\s+control', 20),
    (r'\bdepopulat', 24),
    (r'chem\s*trail', 22),
    (r'mind\s+control', 18),
    (r'flat\s+earth', 26),
    (r'reptilian|shapeshifting\s+reptil|lizard\s+people', 28),
    (r'\bhaarp\b', 22),
    (r'q.?anon|\bqanon\b', 24),
    (r'plandemic|scamdemic', 28),
    (r'great\s+reset', 14),
    (r'agenda\s+2030', 22),
    (r'microchip.{0,20}vaccine|vaccine.{0,20}microchip', 24),
    (r'5g.{0,20}control|control.{0,20}5g', 22),
    (r'big\s+pharma', 16),
    (r'government\s+(cover.?up|conspir)', 18),
    (r'refuse\s+the\s+jab', 20),
    (r'deplatform', 14),
    (r'silenced\s+and\s+deplatform', 18),
    (r'media\s+refuses\s+to\s+report', 20),
    (r'bombshell\s+revelation', 16),
    (r'get\s+censored', 14),
    (r'running\s+out\s+of\s+time', 10),
    (r'nano.?sized\s+microchip', 22),
    (r'track\s+and\s+control\s+the\s+population', 22),
    (r'false\s+flag', 18),
    (r'crisis\s+actor', 20),
    (r'paid\s+shill', 16),
    (r'illuminati', 22),
    (r'crimes\s+against\s+humanity', 14),

    # Emotional manipulation
    (r'\bbombshell\b', 14),
    (r'shocking\s+revelation', 14),
    (r'explosive\s+revelation', 14),
    (r'truth\s+exposed', 16),
    (r'leaked\s+document', 14),
    (r'secret\s+document', 14),

    # Patriot / fight language
    (r'patriots?\s+(must|need)', 14),
    (r'stand\s+up\s+(and\s+)?demand', 12),
    (r'globalists?\s+will\s+have\s+nowhere', 18),
    (r'the\s+truth\s+will\s+come\s+out', 14),
]

# ── Real-news signal patterns (weight = amount subtracted from fake score) ─────
_REAL_PATTERNS = [
    (r'peer.?reviewed', 14),
    (r'study\s+published\s+in', 12),
    (r'according\s+to\s+(data|figures|statistics|the\s+study)', 12),
    (r'spokesperson\s+(said|told|confirmed)', 12),
    (r'told\s+reporters|told\s+journalists', 12),
    (r'press\s+conference', 10),
    (r'official\s+(statement|announcement)', 12),
    (r'committee\s+(hearing|voted|approved)', 12),
    (r'federal\s+(reserve|court|bureau|agency|government)', 10),
    (r'supreme\s+court', 10),
    (r'statistical\s+(analysis|significance)', 14),
    (r'\b(reuters|associated\s+press|ap\s+news)\b', 16),
    (r'world\s+health\s+organization', 12),
    (r'\b(cdc|nih|fda|who)\s+(said|confirmed|announced|reported)', 14),
    (r'independent(ly)?\s+verif', 12),
    (r'peer\s+review', 14),
    (r'congressional\s+(hearing|testimony|report)', 10),
    (r'published\s+in\s+the\s+(journal|lancet|nature|science)', 16),
]


def _structural_score(text: str) -> int:
    """Detect fake news via stylistic / structural features."""
    score = 0
    words = text.split()

    # ALL-CAPS words (≥4 letters)
    caps = [w for w in words if w.isupper() and len(w) >= 4 and w.isalpha()]
    if len(caps) >= 5:
        score += 22
    elif len(caps) >= 3:
        score += 14
    elif len(caps) >= 1:
        score += 6

    # Excessive exclamation marks
    excl = text.count('!')
    if excl >= 5:
        score += 22
    elif excl >= 3:
        score += 14
    elif excl >= 2:
        score += 7

    # BREAKING / EXCLUSIVE / URGENT prefix
    if re.match(r'^\s*(BREAKING|EXCLUSIVE|URGENT|ALERT)\b', text):
        score += 14

    # Multiple question marks
    if text.count('?') >= 3:
        score += 10

    return score


def analyze(raw_text: str) -> dict:
    """
    Analyze raw text for fake / real news signals.

    Returns
    -------
    dict
        fake_bias      : int (-50 to +50). Positive = leans fake.
        fake_signals   : list[str] — top matched fake patterns
        real_signals   : list[str] — top matched real patterns
    """
    tl = raw_text.lower()
    fake_score = 0
    real_score = 0
    fake_signals: list[str] = []
    real_signals: list[str] = []

    for pattern, weight in _FAKE_PATTERNS:
        if re.search(pattern, tl):
            fake_score += weight
            fake_signals.append(pattern[:40])

    for pattern, weight in _REAL_PATTERNS:
        if re.search(pattern, tl):
            real_score += weight
            real_signals.append(pattern[:40])

    fake_score += _structural_score(raw_text)

    # Cap to ±50 so heuristic never completely overrides ML
    fake_bias = min(50, max(-50, fake_score - real_score))

    return {
        'fake_bias':    fake_bias,
        'fake_signals': fake_signals[:6],
        'real_signals': real_signals[:6],
    }
