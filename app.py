"""
app.py
------
Fake News Detector — Production Flask Application
with Automated Self-Learning System

Routes (Prediction):
  GET  /                     → main dashboard (index.html)
  POST /predict              → { text } → prediction JSON
  POST /api/url-check        → URL article prediction
  POST /api/image-check      → image manipulation analysis
  POST /api/voice-check      → audio transcription + prediction
  POST /api/lang-check       → multilingual prediction
  GET  /api/sample           → sample articles for UI demo
  GET  /health               → health check

Routes (Admin):
  GET  /admin                → admin dashboard (admin.html)
  GET  /api/admin/status     → live system status JSON
  GET  /api/admin/versions   → model version history JSON
  GET  /api/admin/logs       → last N lines of latest log
  POST /api/admin/retrain    → trigger manual retrain (password-protected)
  POST /api/admin/fetch-now  → trigger immediate news fetch

Self-Learning:
  APScheduler runs two background jobs:
    • Daily  at FETCH_HOUR    → scheduler.news_fetcher.fetch_and_store()
    • Weekly at RETRAIN_HOUR  → scheduler.retrainer.retrain_model()

  Predictions are NEVER blocked during retraining.
  The model is hot-swapped atomically via ModelManager after each retrain.

Usage:
  python app.py                        # dev
  gunicorn app:app --workers 2 --bind 0.0.0.0:5000   # prod
"""

import logging
import os
import re
from datetime import datetime, timezone

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, jsonify, render_template, request

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Vercel serverless: no background scheduler / retrain
IS_VERCEL = os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))

# ── Feature modules (graceful — missing deps won't crash the app) ─────────────
try:
    from features.url_checker import extract_article, source_credibility
    URL_CHECKER_OK = True
    logger.info("[App] URL checker loaded successfully.")
except Exception as _e:
    URL_CHECKER_OK = False
    logger.warning("[App] URL checker unavailable: %s", _e)

try:
    from features.image_checker import analyse_image
    IMAGE_CHECKER_OK = True
    logger.info("[App] Image checker loaded successfully.")
except Exception as _e:
    IMAGE_CHECKER_OK = False
    logger.warning("[App] Image checker unavailable: %s", _e)

try:
    from features.voice_checker import transcribe_audio
    VOICE_CHECKER_OK = True
    logger.info("[App] Voice checker loaded successfully.")
except Exception as _e:
    VOICE_CHECKER_OK = False
    logger.warning("[App] Voice checker unavailable: %s", _e)

try:
    from features.lang_support import prepare_for_prediction
    LANG_SUPPORT_OK = True
    logger.info("[App] Language support loaded successfully.")
except Exception as _e:
    LANG_SUPPORT_OK = False
    logger.warning("[App] Language support unavailable: %s", _e)

# ── Self-learning core modules ────────────────────────────────────────────────
from core.model_manager import manager           # thread-safe singleton
from core.prediction    import predict_text      # unified ML + heuristic prediction
from core.preprocessor  import preprocess        # shared NLP pipeline

try:
    from core.db import (
        count_articles,
        get_articles_today_count,
        get_model_versions,
        get_training_runs,
    )
    DB_OK = True
except Exception as _e:
    DB_OK = False
    logger.warning("[App] DB module unavailable: %s", _e)
    def count_articles(label=None): return 0
    def get_articles_today_count(): return 0
    def get_model_versions(limit=20): return []
    def get_training_runs(limit=10): return []

# ── Scheduler jobs (disabled on Vercel — no long-running background tasks) ───
NEWS_FETCHER_OK = False
RETRAINER_OK = False


def fetch_and_store() -> int:
    return 0


def retrain_model() -> dict:
    return {
        "success": False,
        "message": "Retraining is not available on serverless hosting. Run locally or use Render.",
    }


if not IS_VERCEL:
    try:
        from scheduler.news_fetcher import fetch_and_store as _fetch_and_store
        fetch_and_store = _fetch_and_store
        NEWS_FETCHER_OK = True
        logger.info("[App] News fetcher loaded.")
    except Exception as _e:
        logger.warning("[App] News fetcher unavailable: %s", _e)

    try:
        from scheduler.retrainer import retrain_model as _retrain_model
        retrain_model = _retrain_model
        RETRAINER_OK = True
        logger.info("[App] Retrainer loaded.")
    except Exception as _e:
        logger.warning("[App] Retrainer unavailable: %s", _e)
else:
    logger.info("[App] Vercel mode — scheduler and retrain disabled.")

# ── Config ────────────────────────────────────────────────────────────────────
ADMIN_PASSWORD        = os.getenv("ADMIN_PASSWORD", "admin123")
FETCH_HOUR            = int(os.getenv("FETCH_HOUR", "7"))
RETRAIN_DAY_OF_WEEK   = os.getenv("RETRAIN_DAY_OF_WEEK", "mon")
RETRAIN_HOUR          = int(os.getenv("RETRAIN_HOUR", "2"))

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "change-me-in-production")

# ---------------------------------------------------------------------------
# APScheduler — Background task scheduler
# ---------------------------------------------------------------------------
_scheduler = None

def _start_scheduler() -> None:
    """
    Initialise and start the APScheduler BackgroundScheduler with two jobs:
      1. Daily news fetch   (every day at FETCH_HOUR UTC)
      2. Weekly model retrain (every RETRAIN_DAY_OF_WEEK at RETRAIN_HOUR UTC)
    """
    global _scheduler
    if IS_VERCEL:
        logger.info("[Scheduler] Skipped — not supported on Vercel serverless.")
        return
    if not NEWS_FETCHER_OK or not RETRAINER_OK:
        logger.warning("[Scheduler] Scheduler jobs skipped — fetcher/retrainer unavailable.")
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron         import CronTrigger

        _scheduler = BackgroundScheduler(timezone="UTC", daemon=True)

        # ── Job 1: Daily news fetch ──────────────────────────────────────────
        _scheduler.add_job(
            func     = fetch_and_store,
            trigger  = CronTrigger(hour=FETCH_HOUR, minute=0),
            id       = "daily_news_fetch",
            name     = "Daily NewsAPI Fetch",
            replace_existing = True,
            misfire_grace_time = 600,       # allow up to 10-min late start
        )

        # ── Job 2: Weekly retrain ────────────────────────────────────────────
        _scheduler.add_job(
            func     = retrain_model,
            trigger  = CronTrigger(
                day_of_week = RETRAIN_DAY_OF_WEEK,
                hour        = RETRAIN_HOUR,
                minute      = 0,
            ),
            id       = "weekly_retrain",
            name     = "Weekly Model Retrain",
            replace_existing = True,
            misfire_grace_time = 3600,      # allow up to 1-hour late start
        )

        _scheduler.start()
        logger.info(
            "[Scheduler] Started — fetch daily at %02d:00 UTC | "
            "retrain every %s at %02d:00 UTC",
            FETCH_HOUR, RETRAIN_DAY_OF_WEEK.upper(), RETRAIN_HOUR,
        )

    except ImportError:
        logger.warning(
            "[Scheduler] APScheduler not installed — background jobs disabled. "
            "Run:  pip install apscheduler"
        )
    except Exception as exc:
        logger.error("[Scheduler] Failed to start: %s", exc)


def _get_next_run_time(job_id: str) -> str | None:
    """Return ISO string of the next scheduled run for a given job ID."""
    if _scheduler is None:
        return None
    try:
        job = _scheduler.get_job(job_id)
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Sample articles for the frontend demo
# ---------------------------------------------------------------------------
SAMPLES = [
    {
        "label": "fake",
        "title": "Government Admits to Secret Mind-Control Programme",
        "text": (
            "Whistleblowers inside the Pentagon have confirmed that a decades-long "
            "mind-control programme using 5G towers has been operating in secret. "
            "Officials deny any knowledge of the project, but leaked documents show "
            "otherwise. The mainstream media refuses to report this bombshell revelation. "
            "Share before this gets censored! The deep state is running out of time."
        ),
    },
    {
        "label": "real",
        "title": "Federal Reserve Raises Interest Rates Amid Inflation Concerns",
        "text": (
            "The Federal Reserve raised its benchmark interest rate by 25 basis points "
            "on Wednesday, continuing its campaign to bring inflation back to the 2% target. "
            "Chair Jerome Powell said the decision reflects the committee's commitment to "
            "restoring price stability while sustaining a strong labour market. "
            "Markets had widely anticipated the move, with futures pricing in a 90% probability "
            "of the hike ahead of the announcement."
        ),
    },
    {
        "label": "fake",
        "title": "Scientists Discover COVID Vaccine Contains Microchip",
        "text": (
            "Independent researchers have published findings showing that COVID-19 vaccines "
            "contain nano-sized microchips designed to track and control the population. "
            "Big Pharma and government agencies are suppressing this information. "
            "Thousands of doctors who speak out are being silenced and deplatformed. "
            "The globalist agenda is becoming clear — wake up and refuse the jab!"
        ),
    },
    {
        "label": "real",
        "title": "NASA Artemis Mission Successfully Enters Lunar Orbit",
        "text": (
            "NASA's Artemis spacecraft successfully entered lunar orbit on Friday after a "
            "six-day journey from Earth, marking a major milestone in the agency's plan to "
            "return astronauts to the Moon. Flight controllers at Johnson Space Center "
            "confirmed the orbital insertion burn completed nominally. The mission will "
            "conduct a series of critical tests before a crewed landing attempt planned "
            "for later in the decade."
        ),
    },
]


# ===========================================================================
# PREDICTION ROUTES  (unchanged public API)
# ===========================================================================

@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST { "text": "<news article>" }
    Returns prediction JSON compatible with the existing frontend.
    Uses ModelManager for thread-safe inference.
    """
    if not manager.is_ready:
        return jsonify({
            "error": "Model not loaded. Please run train_model.py first and restart the server."
        }), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    raw_text = str(data["text"]).strip()
    if len(raw_text) < 20:
        return jsonify({"error": "Please enter at least 20 characters of news text."}), 400

    try:
        pred = predict_text(raw_text)
        return jsonify({
            "result":     pred["result"],
            "confidence": pred["confidence"],
            "fake_prob":  pred["fake_prob"],
            "real_prob":  pred["real_prob"],
            "word_count": len(raw_text.split()),
            "signals":    pred.get("signals", {}),
            "uncertain":  pred.get("uncertain", False),
        })
    except Exception as exc:
        logger.error("[Predict] Unexpected error: %s", exc)
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/api/sample", methods=["GET"])
def sample_articles():
    """Return sample news articles for the frontend demo carousel."""
    return jsonify(SAMPLES)


@app.route("/health", methods=["GET"])
def health():
    """Health check — for uptime monitors and deployment platforms."""
    info = manager.get_info()
    return jsonify({
        "status":       "ok",
        "model_loaded": manager.is_ready,
        "is_training":  info.get("is_training", False),
        "version_id":   info.get("version_id"),
        "accuracy":     info.get("accuracy"),
        "platform":     "vercel" if IS_VERCEL else "standard",
        "features": {
            "url_checker":   URL_CHECKER_OK,
            "image_checker": IMAGE_CHECKER_OK,
            "voice_checker": VOICE_CHECKER_OK,
            "lang_support":  LANG_SUPPORT_OK,
            "db":            DB_OK,
        }
    })


# ===========================================================================
# ADVANCED FEATURE ROUTES  (graceful degradation if deps missing)
# ===========================================================================

@app.route("/api/url-check", methods=["POST"])
def url_check():
    """Extract article from URL, run ML prediction, score source credibility."""
    if not URL_CHECKER_OK:
        return jsonify({"error": "URL checker unavailable. Install: newspaper3k beautifulsoup4 requests"}), 503

    if not manager.is_ready:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Request body must contain a 'url' field."}), 400

    url = str(data["url"]).strip()

    try:
        extracted = extract_article(url)
        if "error" in extracted:
            return jsonify({"error": extracted["error"]}), 422

        title = extracted.get("title", "")
        text  = extracted.get("text", "")

        if len(text.split()) < 20:
            return jsonify({"error": "Extracted text too short for reliable prediction."}), 422

        pred = predict_text(text)
        cred = source_credibility(url, title, text)

        return jsonify({
            "result":      pred["result"],
            "confidence":  pred["confidence"],
            "fake_prob":   pred["fake_prob"],
            "real_prob":   pred["real_prob"],
            "word_count":  len(text.split()),
            "title":       title,
            "excerpt":     text[:400] + ("..." if len(text) > 400 else ""),
            "credibility": cred,
            "method":      extracted.get("method", "unknown"),
        })
    except Exception as exc:
        logger.error("[URL-Check] Error: %s", exc)
        return jsonify({"error": f"URL analysis failed: {exc}"}), 500


@app.route("/api/image-check", methods=["POST"])
def image_check():
    """Analyse image for manipulation using ELA + EXIF + OpenCV."""
    if not IMAGE_CHECKER_OK:
        return jsonify({"error": "Image checker unavailable. Install: Pillow imagehash opencv-python-headless"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    allowed_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed_exts:
        return jsonify({"error": f"Unsupported image type: {ext}"}), 400

    file_bytes = f.read()
    if len(file_bytes) > 10 * 1024 * 1024:
        return jsonify({"error": "Image too large (max 10 MB)."}), 413

    try:
        result = analyse_image(file_bytes, f.filename)
        if "error" in result:
            return jsonify({"error": result["error"]}), 422
        return jsonify(result)
    except Exception as exc:
        logger.error("[Image-Check] Error: %s", exc)
        return jsonify({"error": f"Image analysis failed: {exc}"}), 500


@app.route("/api/voice-check", methods=["POST"])
def voice_check():
    """Transcribe audio then run ML prediction on transcript."""
    if not VOICE_CHECKER_OK:
        return jsonify({"error": "Voice checker unavailable. Install: SpeechRecognition pydub"}), 503

    if not manager.is_ready:
        return jsonify({"error": "Model not loaded."}), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Use field name 'audio'."}), 400

    f = request.files["audio"]
    if f.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    file_bytes = f.read()
    if len(file_bytes) > 25 * 1024 * 1024:
        return jsonify({"error": "Audio file too large (max 25 MB)."}), 413

    try:
        trans_result = transcribe_audio(file_bytes, f.filename)
        if "error" in trans_result:
            return jsonify({"error": trans_result["error"]}), 422

        transcript = trans_result["transcript"]
        word_count = trans_result["word_count"]

        if word_count < 10:
            return jsonify({
                "transcript": transcript,
                "word_count": word_count,
                "error": "Transcript too short for reliable prediction (min 10 words).",
            }), 422

        pred = predict_text(transcript)

        return jsonify({
            "transcript": transcript,
            "word_count": word_count,
            "result":     pred["result"],
            "confidence": pred["confidence"],
            "fake_prob":  pred["fake_prob"],
            "real_prob":  pred["real_prob"],
            "signals":    pred.get("signals", {}),
        })
    except Exception as exc:
        logger.error("[Voice-Check] Error: %s", exc)
        return jsonify({"error": f"Voice analysis failed: {exc}"}), 500


@app.route("/api/lang-check", methods=["POST"])
def lang_check():
    """Detect language, translate to English if needed, then predict."""
    if not LANG_SUPPORT_OK:
        return jsonify({"error": "Language support unavailable. Install: langdetect deep-translator"}), 503

    if not manager.is_ready:
        return jsonify({"error": "Model not loaded."}), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    raw_text = str(data["text"]).strip()
    if len(raw_text) < 20:
        return jsonify({"error": "Please enter at least 20 characters."}), 400

    try:
        lang_result  = prepare_for_prediction(raw_text)
        english_text = lang_result.get("translated_text", raw_text)

        pred = predict_text(english_text)

        return jsonify({
            "result":              pred["result"],
            "confidence":          pred["confidence"],
            "fake_prob":           pred["fake_prob"],
            "real_prob":           pred["real_prob"],
            "word_count":          len(raw_text.split()),
            "detected_lang_code":  lang_result.get("detected_lang_code", "en"),
            "detected_lang_name":  lang_result.get("detected_lang_name", "English"),
            "original_text":       raw_text,
            "translated_text":     english_text,
            "translation_note":    lang_result.get("translation_note", ""),
            "translation_warning": lang_result.get("translation_warning", ""),
        })
    except Exception as exc:
        logger.error("[Lang-Check] Error: %s", exc)
        return jsonify({"error": f"Language check failed: {exc}"}), 500


# ===========================================================================
# ADMIN ROUTES
# ===========================================================================

@app.route("/admin")
def admin_dashboard():
    """Serve the admin control panel."""
    return render_template("admin.html")


@app.route("/api/admin/status", methods=["GET"])
def admin_status():
    """
    Return live system status JSON consumed by the admin dashboard.
    """
    model_info   = manager.get_info()
    total        = count_articles()
    fake_count   = count_articles(label=0)
    real_count   = count_articles(label=1)
    today_count  = get_articles_today_count()

    # Serialise datetime objects in model_info
    if isinstance(model_info.get("trained_at"), datetime):
        model_info["trained_at"] = model_info["trained_at"].isoformat()

    return jsonify({
        "model_info": model_info,
        "article_stats": {
            "total": total,
            "fake":  fake_count,
            "real":  real_count,
            "today": today_count,
        },
        "next_jobs": {
            "fetch":   _get_next_run_time("daily_news_fetch"),
            "retrain": _get_next_run_time("weekly_retrain"),
        },
        "recent_runs": _serialise_docs(get_training_runs(limit=5)),
    })


@app.route("/api/admin/versions", methods=["GET"])
def admin_versions():
    """Return model version history."""
    versions = get_model_versions(limit=20)
    return jsonify({"versions": _serialise_docs(versions)})


@app.route("/api/admin/logs", methods=["GET"])
def admin_logs():
    """Return the last N lines of today's training or fetch log."""
    try:
        lines = int(request.args.get("lines", 100))
    except ValueError:
        lines = 100

    log_output = _read_latest_log(lines)
    return jsonify({"log": log_output})


@app.route("/api/admin/retrain", methods=["POST"])
def admin_retrain():
    """
    Trigger an immediate retrain cycle.
    Requires { "password": "<ADMIN_PASSWORD>" } in the JSON body.
    """
    if not RETRAINER_OK:
        return jsonify({"error": "Retrainer not available."}), 503

    data = request.get_json(silent=True) or {}
    if data.get("password") != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password."}), 403

    if manager.is_training:
        return jsonify({
            "success": False,
            "message": "A retrain is already in progress. Please wait for it to finish.",
        }), 409

    logger.info("[Admin] Manual retrain triggered by admin dashboard.")

    # Run the retrain (blocking — returns result dict)
    try:
        result = retrain_model()
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500

    return jsonify({
        "success":  result.get("success", False),
        "accuracy": result.get("accuracy"),
        "replaced": result.get("replaced", False),
        "message":  result.get("message", ""),
    })


@app.route("/api/admin/fetch-now", methods=["POST"])
def admin_fetch_now():
    """
    Trigger an immediate news fetch cycle.
    """
    if not NEWS_FETCHER_OK:
        return jsonify({"error": "News fetcher not available."}), 503

    data = request.get_json(silent=True) or {}
    pw   = data.get("password", "")
    # Allow fetch without password if called from internal dashboard
    if pw and pw != ADMIN_PASSWORD:
        return jsonify({"error": "Invalid admin password."}), 403

    logger.info("[Admin] Manual news fetch triggered.")
    try:
        stored = fetch_and_store()
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 500

    return jsonify({
        "success": True,
        "stored":  stored,
        "message": f"News fetch complete. {stored} new articles stored.",
    })


# ===========================================================================
# HELPERS
# ===========================================================================

def _read_latest_log(n: int = 100) -> str:
    """
    Read the last `n` lines from the most recent log file
    (training or fetch log, whichever is newest).
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")

    if not os.path.isdir(logs_dir):
        return "(no logs directory found)"

    # Find the newest log file
    try:
        log_files = sorted(
            [f for f in os.listdir(logs_dir) if f.endswith(".log")],
            reverse=True,
        )
        if not log_files:
            return "(no log files yet — run a retrain or news fetch first)"

        log_path = os.path.join(logs_dir, log_files[0])
        with open(log_path, "r", encoding="utf-8") as fh:
            all_lines = fh.readlines()

        tail = all_lines[-n:] if len(all_lines) > n else all_lines
        return "".join(tail)

    except Exception as exc:
        return f"(error reading log: {exc})"


def _serialise_docs(docs: list) -> list:
    """
    Convert a list of MongoDB documents (which may contain datetime objects)
    to plain JSON-serialisable dicts.
    """
    result = []
    for doc in docs:
        clean = {}
        for k, v in doc.items():
            if isinstance(v, datetime):
                clean[k] = v.isoformat()
            else:
                clean[k] = v
        result.append(clean)
    return result


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Start the background scheduler (news fetch + auto-retrain)
    _start_scheduler()

    logger.info("[App] Starting Fake News Detector server…")
    logger.info("[App] Admin dashboard: http://localhost:5000/admin")
    logger.info("[App] Main app:        http://localhost:5000/")

    # Debug mode for local development only
    # In production use:  gunicorn app:app --workers 2 --bind 0.0.0.0:5000
    app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
