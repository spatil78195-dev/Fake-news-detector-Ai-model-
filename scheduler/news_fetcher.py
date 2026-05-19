"""
scheduler/news_fetcher.py
-------------------------
Daily background job that fetches fresh news articles from NewsAPI and
stores them in MongoDB.  Each article is auto-labelled using the current
in-memory model so the dataset stays up-to-date for the next retrain.

Schedule:  Every day at FETCH_HOUR (default 07:00 UTC)

Flow
----
  1.  Call NewsAPI /v2/top-headlines for multiple categories
  2.  Pre-process each article's combined title + description + content
  3.  Auto-label using the current model (0=Fake / 1=Real)
  4.  Upsert into MongoDB news_articles (de-duplicated by URL)
  5.  Write a dated log to logs/fetch_YYYY-MM-DD.log

Graceful degradation
--------------------
  If NewsAPI is unavailable, or the newsapi-python package is not installed,
  the job logs the error and returns without crashing the Flask process.
"""

import logging
import os
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.db import insert_article
from core.model_manager import manager
from core.preprocessor import preprocess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR    = os.path.join(BASE_DIR, "logs")

os.makedirs(LOGS_DIR, exist_ok=True)

# Categories / topics to fetch headlines for
FETCH_CATEGORIES = [
    "general",
    "technology",
    "health",
    "science",
    "politics",
]

# Maximum articles to retrieve per category per run
MAX_PER_CATEGORY = 20


# ---------------------------------------------------------------------------
# Log helper
# ---------------------------------------------------------------------------

def _write_log(lines: list) -> None:
    """Append lines to today's fetch log file."""
    today     = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path  = os.path.join(LOGS_DIR, f"fetch_{today}.log")

    with open(log_path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Article auto-labelling
# ---------------------------------------------------------------------------

def _auto_label(title: str, description: str, content: str) -> dict:
    """
    Use the current in-memory model to label an article.

    Returns
    -------
    dict : {"label": int, "confidence": float}
          label = 0 (Fake) | 1 (Real) | -1 (model unavailable)
    """
    if not manager.is_ready:
        return {"label": -1, "confidence": 0.0}

    raw   = f"{title} {description} {content}"
    clean = preprocess(raw)

    if not clean:
        return {"label": -1, "confidence": 0.0}

    try:
        vec        = manager.vectorize(clean)
        label, confidence, _, _ = manager.predict(vec)
        int_label  = 1 if label == "Real" else 0
        return {"label": int_label, "confidence": confidence}
    except Exception as exc:
        logger.warning("[Fetcher] Auto-label failed: %s", exc)
        return {"label": -1, "confidence": 0.0}


# ---------------------------------------------------------------------------
# Main fetch function (called by APScheduler)
# ---------------------------------------------------------------------------

def fetch_and_store() -> int:
    """
    Fetch latest news from NewsAPI and persist to MongoDB.

    Returns
    -------
    int  Number of new articles successfully stored.
    """
    ts    = datetime.now(timezone.utc).isoformat()
    lines = [f"", f"{'='*60}", f"  News Fetch Run — {ts}", f"{'='*60}"]

    if not NEWSAPI_KEY:
        msg = "[Fetcher] NEWSAPI_KEY is not set — skipping fetch."
        logger.warning(msg)
        lines.append(msg)
        _write_log(lines)
        return 0

    # ------------------------------------------------------------------
    # Import newsapi client (optional dependency)
    # ------------------------------------------------------------------
    try:
        from newsapi import NewsApiClient
    except ImportError:
        msg = (
            "[Fetcher] newsapi-python not installed. "
            "Run:  pip install newsapi-python"
        )
        logger.error(msg)
        lines.append(msg)
        _write_log(lines)
        return 0

    newsapi      = NewsApiClient(api_key=NEWSAPI_KEY)
    stored_count = 0
    total_fetched = 0

    for category in FETCH_CATEGORIES:
        try:
            response = newsapi.get_top_headlines(
                category=category,
                language="en",
                page_size=MAX_PER_CATEGORY,
            )

            articles = response.get("articles", [])
            lines.append(
                f"\n[{category.upper()}]  fetched {len(articles)} articles"
            )

            for art in articles:
                total_fetched += 1

                url         = art.get("url", "") or ""
                title       = art.get("title", "") or ""
                description = art.get("description", "") or ""
                content     = art.get("content", "") or ""
                source_name = (art.get("source") or {}).get("name", "Unknown")
                published   = art.get("publishedAt", "")

                if not url or not title:
                    continue

                # Auto-label using current model
                label_info = _auto_label(title, description, content)

                doc = {
                    "url":          url,
                    "title":        title,
                    "description":  description,
                    "content":      content,
                    "source":       source_name,
                    "category":     category,
                    "published_at": published,
                    "label":        label_info["label"],
                    "confidence":   label_info["confidence"],
                    "fetched_at":   datetime.now(timezone.utc),
                    "origin":       "newsapi",
                }

                if insert_article(doc):
                    stored_count += 1
                    lines.append(
                        f"  [+] {title[:70]}  →  "
                        f"{'REAL' if label_info['label']==1 else 'FAKE'}  "
                        f"({label_info['confidence']:.1f}%)"
                    )

        except Exception as exc:
            err_msg = f"  [ERROR] Category '{category}': {exc}"
            logger.error("[Fetcher] %s", err_msg)
            lines.append(err_msg)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = (
        f"\nSummary: fetched={total_fetched}  "
        f"stored={stored_count}  "
        f"duplicates_skipped={total_fetched - stored_count}"
    )
    lines.append(summary)
    lines.append(f"Run finished at {datetime.now(timezone.utc).isoformat()}")

    _write_log(lines)
    logger.info("[Fetcher] %s", summary.strip())

    return stored_count
