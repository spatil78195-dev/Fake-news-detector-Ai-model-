"""
scheduler/retrainer.py
----------------------
Weekly background retrain using the shared core.training pipeline.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from datetime import datetime, timezone

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from core.db import (
    get_all_articles,
    record_training_run,
    register_model_version,
    set_active_version,
)
from core.model_manager import manager
from core.training import FAKE_CSV, TRUE_CSV, load_dataframe, train_pipeline

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MIN_ACCURACY_DELTA = float(os.getenv("MIN_ACCURACY_DELTA", "-0.02"))
MIN_SAMPLES = 100

os.makedirs(LOGS_DIR, exist_ok=True)


def _write_log(lines: list) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = os.path.join(LOGS_DIR, f"training_{today}.log")
    with open(log_path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")
    return log_path


def _load_mongodb_articles() -> pd.DataFrame:
    articles = get_all_articles()
    if not articles:
        return pd.DataFrame(columns=["content", "label"])

    rows = []
    for art in articles:
        label = art.get("label", -1)
        if label not in (0, 1):
            continue
        title = art.get("title", "") or ""
        desc = art.get("description", "") or ""
        content = art.get("content", "") or ""
        rows.append({
            "content": f"{title} {desc} {content}".strip(),
            "label": label,
        })
    return pd.DataFrame(rows)


def retrain_model() -> dict:
    """Execute a full retrain cycle; hot-swap if accuracy check passes."""
    start_time = time.time()
    started_at = datetime.now(timezone.utc)
    log_lines: list = []
    result = {
        "success": False,
        "accuracy": None,
        "replaced": False,
        "message": "",
        "log_path": None,
    }

    def log(msg: str) -> None:
        log_lines.append(msg)
        logger.info("[Retrainer] %s", msg)

    log("")
    log("=" * 65)
    log(f"  Retrain Job Started — {started_at.isoformat()}")
    log("=" * 65)

    manager.set_training(True)

    try:
        log("\n[1/4] Loading training data …")
        mongo_df = _load_mongodb_articles()
        try:
            df = load_dataframe(extra_df=mongo_df if not mongo_df.empty else None)
        except FileNotFoundError as exc:
            msg = str(exc)
            log(f"  [ABORT] {msg}")
            result["message"] = msg
            return result

        total_samples = len(df)
        fake_count = int((df["label"] == 0).sum())
        real_count = int((df["label"] == 1).sum())
        log(f"  Total (balanced): {total_samples} | Fake: {fake_count} | Real: {real_count}")

        if total_samples < MIN_SAMPLES:
            msg = f"Insufficient data ({total_samples} < {MIN_SAMPLES}). Aborting."
            log(f"  [ABORT] {msg}")
            result["message"] = msg
            return result

        log("\n[2/4] Training (shared pipeline) …")
        train_result = train_pipeline(df=df)
        accuracy = train_result["accuracy"]
        fake_threshold = train_result["fake_threshold"]

        log(f"  Model            : {train_result['model_name']}")
        log(f"  Test accuracy    : {accuracy * 100:.2f}%")
        log(f"  Fake F1          : {train_result['fake_f1'] * 100:.2f}%")
        log(f"  Fake threshold   : {fake_threshold:.3f}")

        for line in train_result["classification_report"].splitlines():
            log(f"  {line}")
        log(f"  Confusion matrix: {train_result['confusion_matrix'].tolist()}")

        prev_info = manager.get_info()
        prev_accuracy = prev_info.get("accuracy")

        if prev_accuracy is not None:
            delta = accuracy - prev_accuracy
            log(
                f"\n  Prev accuracy: {prev_accuracy * 100:.2f}% | "
                f"Delta: {delta * 100:+.2f}% | Threshold: {MIN_ACCURACY_DELTA * 100:.1f}%"
            )
            if delta < MIN_ACCURACY_DELTA:
                msg = (
                    f"New model rejected ({accuracy*100:.2f}% vs "
                    f"{prev_accuracy*100:.2f}% previous)."
                )
                log(f"  [REJECTED] {msg}")
                result.update({"success": True, "accuracy": accuracy, "message": msg})
                return result

        version_id = f"v_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "fake_threshold": fake_threshold,
            "accuracy": accuracy,
            "fake_f1": train_result["fake_f1"],
            "model_name": train_result["model_name"],
        }

        model_path, vec_path = manager.save_versioned(
            train_result["model"],
            train_result["vectorizer"],
            version_id,
            meta=meta,
        )
        log(f"\n  Saved: {model_path}")

        version_meta = {
            "version_id": version_id,
            "accuracy": accuracy,
            "fake_f1": train_result["fake_f1"],
            "fake_threshold": fake_threshold,
            "trained_at": started_at,
            "samples": total_samples,
            "fake_count": fake_count,
            "real_count": real_count,
            "model_path": model_path,
            "vectorizer_path": vec_path,
            "is_active": True,
        }

        manager.hot_swap(
            train_result["model"],
            train_result["vectorizer"],
            version_meta,
        )
        log(f"  [HOT-SWAP] Active version: {version_id}")

        register_model_version(version_meta)
        set_active_version(version_id)

        duration = round(time.time() - start_time, 2)
        finish_msg = (
            f"Retrain complete in {duration}s — "
            f"accuracy={accuracy*100:.2f}% fake_f1={train_result['fake_f1']*100:.2f}%"
        )
        log(finish_msg)

        record_training_run({
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc),
            "duration_sec": duration,
            "accuracy": accuracy,
            "samples_total": total_samples,
            "fake_count": fake_count,
            "real_count": real_count,
            "version_id": version_id,
            "replaced_model": True,
            "notes": "Automated weekly retrain (shared pipeline)",
        })

        result.update({
            "success": True,
            "accuracy": accuracy,
            "replaced": True,
            "message": finish_msg,
        })
        return result

    except Exception as exc:
        err_msg = f"[ERROR] {exc}"
        log(err_msg)
        log(traceback.format_exc())
        result["message"] = str(exc)
        record_training_run({
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc),
            "duration_sec": round(time.time() - start_time, 2),
            "accuracy": None,
            "replaced_model": False,
            "notes": f"Failed: {exc}",
        })
        return result

    finally:
        manager.set_training(False)
        result["log_path"] = _write_log(log_lines)
