"""
features/url_checker.py
-----------------------
Extracts news article content from a URL and scores source credibility.

Primary extraction: newspaper3k
Fallback extraction: requests + BeautifulSoup
Credibility scoring: heuristic-based (domain lists, HTTPS, title patterns, length)
"""

import re
import requests
from urllib.parse import urlparse

# ── Optional imports with graceful fallback ──────────────────────────────────
try:
    from newspaper import Article
    NEWSPAPER_OK = True
except ImportError:
    NEWSPAPER_OK = False

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False

# ── Domain credibility lists ─────────────────────────────────────────────────
CREDIBLE = {
    'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk', 'nytimes.com',
    'theguardian.com', 'washingtonpost.com', 'npr.org', 'pbs.org',
    'thehindu.com', 'hindustantimes.com', 'ndtv.com', 'timesofindia.com',
    'aljazeera.com', 'bloomberg.com', 'ft.com', 'economist.com',
    'who.int', 'cdc.gov', 'nih.gov', 'nature.com', 'science.org',
}

UNRELIABLE = {
    'infowars.com', 'naturalnews.com', 'beforeitsnews.com',
    'yournewswire.com', 'worldnewsdailyreport.com', 'empirenews.net',
    'nationalreport.net',
}

SCRAPER_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}

CLICKBAIT_RE = re.compile(
    r'\b(shocking|bombshell|secret|exposed|banned|censored|they hide|wake up)\b'
    r'|share before|mainstream media won',
    re.I
)


# ── Public API ───────────────────────────────────────────────────────────────

def extract_article(url: str) -> dict:
    """
    Extract title + body text from a news URL.
    Returns dict with keys: title, text  OR  error.
    """
    # Validate URL
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        return {'error': 'Invalid URL. Please include http:// or https://'}

    # Primary: newspaper3k
    if NEWSPAPER_OK:
        result = _try_newspaper(url)
        if result:
            return result

    # Fallback: BeautifulSoup
    if BS4_OK:
        result = _try_bs4(url)
        if result:
            return result

    if not NEWSPAPER_OK and not BS4_OK:
        return {'error': 'No extraction library available. Install newspaper3k.'}

    return {'error': 'Could not extract article text. The site may block scrapers.'}


def source_credibility(url: str, title: str, text: str) -> dict:
    """
    Score the credibility of a news source (0–100).
    Returns: { score, label, signals }
    """
    score = 50
    signals = []

    parsed = urlparse(url)
    domain = parsed.netloc.lower().removeprefix('www.')

    # Known domain
    if domain in CREDIBLE:
        score += 30
        signals.append({'text': 'Recognised credible domain', 'positive': True})
    elif domain in UNRELIABLE:
        score -= 40
        signals.append({'text': 'Known unreliable domain', 'positive': False})

    # HTTPS
    if parsed.scheme == 'https':
        score += 5
        signals.append({'text': 'Secure HTTPS connection', 'positive': True})
    else:
        score -= 5
        signals.append({'text': 'Insecure HTTP connection', 'positive': False})

    # Suspicious TLD
    bad_tlds = ('.xyz', '.click', '.win', '.bid', '.loan')
    if any(domain.endswith(t) for t in bad_tlds):
        score -= 20
        signals.append({'text': 'Suspicious domain extension', 'positive': False})

    # Hyphen spam in domain
    if domain.count('-') >= 3:
        score -= 10
        signals.append({'text': 'Excessive hyphens in domain', 'positive': False})

    # Article length
    wc = len(text.split())
    if wc > 400:
        score += 8
        signals.append({'text': f'Substantial article ({wc} words)', 'positive': True})
    elif wc < 100:
        score -= 10
        signals.append({'text': f'Very short article ({wc} words)', 'positive': False})

    # Clickbait title
    if CLICKBAIT_RE.search(title):
        score -= 15
        signals.append({'text': 'Sensationalist/clickbait title', 'positive': False})

    score = max(0, min(100, score))
    return {'score': score, 'label': _cred_label(score), 'signals': signals}


# ── Private helpers ──────────────────────────────────────────────────────────

def _try_newspaper(url: str) -> dict | None:
    try:
        art = Article(url, language='en', request_timeout=15)
        art.download()
        art.parse()
        if art.text and len(art.text) > 100:
            return {'title': art.title or '', 'text': art.text, 'method': 'newspaper3k'}
    except Exception:
        pass
    return None


def _try_bs4(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=SCRAPER_HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')

        # Title
        h1 = soup.find('h1')
        title = h1.get_text(strip=True) if h1 else (
            soup.title.get_text(strip=True) if soup.title else ''
        )

        # Body — prefer <article>, then <main>, then all <p>
        container = (
            soup.find('article') or
            soup.find('div', class_=re.compile(r'article|content|story|post', re.I)) or
            soup.find('main')
        )
        paras = (container or soup).find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 40)

        if len(text) > 100:
            return {'title': title, 'text': text, 'method': 'beautifulsoup'}
    except requests.Timeout:
        return {'error': 'Request timed out.'}
    except Exception:
        pass
    return None


def _cred_label(score: int) -> str:
    if score >= 75:  return 'High Credibility'
    if score >= 50:  return 'Moderate Credibility'
    if score >= 25:  return 'Low Credibility'
    return 'Very Low Credibility'
