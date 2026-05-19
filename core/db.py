"""
core/db.py
----------
MongoDB connection singleton for the Fake News Detector self-learning system.

Collections
-----------
  news_articles  — fetched articles from NewsAPI (de-duplicated by URL)
  training_runs  — log of every retraining job (accuracy, samples, duration …)
  model_versions — registry of saved model versions and their metadata

Graceful degradation
--------------------
If MongoDB is unavailable at startup, `db` is set to None and every helper
function returns a safe no-op value.  The rest of the application continues
to work; article storage and version tracking are simply skipped.

Usage
-----
  from core.db import get_db, insert_article, record_training_run, ...
  db = get_db()          # MongoDatabase | None
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
_client = None
_db     = None

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/fakenews")
DB_NAME   = MONGO_URI.rsplit("/", 1)[-1].split("?")[0] or "fakenews"


def get_db():
    """
    Return the MongoDatabase instance (lazy singleton).
    Returns None if pymongo is not installed or MongoDB is unreachable.
    """
    global _client, _db

    if _db is not None:
        return _db

    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        # Force a round-trip to verify the connection is alive
        _client.admin.command("ping")
        _db = _client[DB_NAME]

        # Ensure unique index on articles URL to prevent duplicates
        _db.news_articles.create_index("url", unique=True)

        logger.info("[DB] Connected to MongoDB at %s (db=%s)", MONGO_URI, DB_NAME)
        return _db

    except ImportError:
        logger.warning("[DB] pymongo not installed — skipping MongoDB.")
        return None
    except Exception as exc:
        logger.warning("[DB] MongoDB unavailable (%s) — running without persistence.", exc)
        return None


# ---------------------------------------------------------------------------
# Article helpers
# ---------------------------------------------------------------------------

def insert_article(article: Dict[str, Any]) -> bool:
    """
    Upsert a news article into the news_articles collection.

    Parameters
    ----------
    article : dict
        Must contain at least: url, title, content, source, label, confidence,
        fetched_at (datetime).

    Returns
    -------
    bool  True if inserted, False if duplicate or DB unavailable.
    """
    db = get_db()
    if db is None:
        return False

    try:
        from pymongo.errors import DuplicateKeyError
        article.setdefault("fetched_at", datetime.now(timezone.utc))
        db.news_articles.insert_one(article)
        return True
    except Exception:
        # DuplicateKeyError → article already stored; other errors are logged
        return False


def count_articles(label: Optional[int] = None) -> int:
    """
    Count articles in the database, optionally filtered by label (0=Fake, 1=Real).
    """
    db = get_db()
    if db is None:
        return 0

    try:
        query = {"label": label} if label is not None else {}
        return db.news_articles.count_documents(query)
    except Exception:
        return 0


def get_all_articles(limit: int = 0) -> List[Dict]:
    """Return all stored articles as a list of dicts (no _id field)."""
    db = get_db()
    if db is None:
        return []

    try:
        cursor = db.news_articles.find({}, {"_id": 0})
        if limit:
            cursor = cursor.limit(limit)
        return list(cursor)
    except Exception:
        return []


def get_articles_today_count() -> int:
    """Return the number of articles fetched today (UTC date)."""
    db = get_db()
    if db is None:
        return 0

    try:
        today = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return db.news_articles.count_documents({"fetched_at": {"$gte": today}})
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Training run helpers
# ---------------------------------------------------------------------------

def record_training_run(run: Dict[str, Any]) -> Optional[str]:
    """
    Insert a training run document.

    Expected keys: started_at, finished_at, accuracy, samples_total,
                   fake_count, real_count, replaced_model (bool), notes.

    Returns the inserted document id (str) or None.
    """
    db = get_db()
    if db is None:
        return None

    try:
        run.setdefault("started_at", datetime.now(timezone.utc))
        result = db.training_runs.insert_one(run)
        return str(result.inserted_id)
    except Exception as exc:
        logger.error("[DB] Failed to record training run: %s", exc)
        return None


def get_training_runs(limit: int = 10) -> List[Dict]:
    """Return the most recent training runs (newest first)."""
    db = get_db()
    if db is None:
        return []

    try:
        cursor = (
            db.training_runs
            .find({}, {"_id": 0})
            .sort("started_at", -1)
            .limit(limit)
        )
        return list(cursor)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Model version helpers
# ---------------------------------------------------------------------------

def register_model_version(meta: Dict[str, Any]) -> Optional[str]:
    """
    Record a new model version in the model_versions collection.

    Expected keys: version_id, model_path, vectorizer_path, accuracy,
                   trained_at, samples, is_active (bool).
    """
    db = get_db()
    if db is None:
        return None

    try:
        meta.setdefault("trained_at", datetime.now(timezone.utc))
        result = db.model_versions.insert_one(meta)
        return str(result.inserted_id)
    except Exception as exc:
        logger.error("[DB] Failed to register model version: %s", exc)
        return None


def set_active_version(version_id: str) -> None:
    """Mark one version as active, deactivate all others."""
    db = get_db()
    if db is None:
        return

    try:
        db.model_versions.update_many({}, {"$set": {"is_active": False}})
        db.model_versions.update_one(
            {"version_id": version_id},
            {"$set": {"is_active": True}},
        )
    except Exception as exc:
        logger.error("[DB] Failed to set active version: %s", exc)


def get_model_versions(limit: int = 20) -> List[Dict]:
    """Return model version history (newest first)."""
    db = get_db()
    if db is None:
        return []

    try:
        cursor = (
            db.model_versions
            .find({}, {"_id": 0})
            .sort("trained_at", -1)
            .limit(limit)
        )
        return list(cursor)
    except Exception:
        return []
