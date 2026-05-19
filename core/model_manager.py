"""
core/model_manager.py
---------------------
Thread-safe model manager for zero-downtime hot-swapping.
"""

from __future__ import annotations

import logging
import os
import pickle
import shutil
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.pkl")
MODELS_DIR = os.path.join(BASE_DIR, "models")

LABEL_FAKE = 0
LABEL_REAL = 1
DEFAULT_FAKE_THRESHOLD = 0.45

os.makedirs(MODELS_DIR, exist_ok=True)


class ModelManager:
    """Thread-safe wrapper around sklearn model + TF-IDF vectorizer."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model = None
        self._vectorizer = None
        self._is_training = False
        self._fake_class_idx = 0
        self._fake_threshold = DEFAULT_FAKE_THRESHOLD

        self._version_meta: Dict[str, Any] = {
            "version_id": "v0_initial",
            "accuracy": None,
            "trained_at": None,
            "samples": 0,
            "fake_threshold": DEFAULT_FAKE_THRESHOLD,
            "model_path": MODEL_PATH,
            "vectorizer_path": VECTORIZER_PATH,
        }

        self._load_from_disk()

    def _load_meta(self) -> None:
        if not os.path.exists(META_PATH):
            return
        try:
            with open(META_PATH, "rb") as f:
                meta = pickle.load(f)
            self._fake_threshold = float(
                meta.get("fake_threshold", DEFAULT_FAKE_THRESHOLD)
            )
            self._version_meta.update(
                {k: v for k, v in meta.items() if k in self._version_meta or k == "fake_f1"}
            )
        except Exception as exc:
            logger.warning("[ModelManager] Could not load model_meta.pkl: %s", exc)

    def _sync_class_index(self) -> None:
        if self._model is None:
            return
        classes = list(getattr(self._model, "classes_", [LABEL_FAKE, LABEL_REAL]))
        try:
            self._fake_class_idx = classes.index(LABEL_FAKE)
        except ValueError:
            self._fake_class_idx = 0

    def _load_from_disk(self) -> None:
        with self._lock:
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                with open(VECTORIZER_PATH, "rb") as f:
                    self._vectorizer = pickle.load(f)
                self._sync_class_index()
                self._load_meta()
                logger.info(
                    "[ModelManager] Model loaded (fake_threshold=%.3f).",
                    self._fake_threshold,
                )
            except FileNotFoundError:
                logger.warning(
                    "[ModelManager] model.pkl or vectorizer.pkl not found. "
                    "Run train_model.py first."
                )

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._model is not None and self._vectorizer is not None

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def fake_threshold(self) -> float:
        return self._fake_threshold

    def vectorize(self, text: str) -> Any:
        with self._lock:
            if self._vectorizer is None:
                raise RuntimeError("Vectorizer not loaded.")
            return self._vectorizer.transform([text])

    def predict_proba(self, sparse_vector) -> Dict[str, Any]:
        """
        Return label and probabilities using tuned fake threshold.

        Uses model.classes_ for correct probability column mapping.
        """
        with self._lock:
            if self._model is None:
                raise RuntimeError("Model not loaded.")

            proba = self._model.predict_proba(sparse_vector)[0]
            p_fake = float(proba[self._fake_class_idx])
            p_real = 1.0 - p_fake

            if p_fake >= self._fake_threshold:
                prediction = LABEL_FAKE
                label = "Fake"
            else:
                prediction = LABEL_REAL
                label = "Real"

        fake_prob = round(p_fake * 100, 2)
        real_prob = round(p_real * 100, 2)
        confidence = round(max(fake_prob, real_prob), 2)

        return {
            "label": label,
            "prediction": prediction,
            "confidence": confidence,
            "fake_prob": fake_prob,
            "real_prob": real_prob,
            "p_fake_raw": p_fake,
        }

    def predict(self, sparse_vector) -> Tuple[str, float, float, float]:
        """Backward-compatible tuple API."""
        out = self.predict_proba(sparse_vector)
        return out["label"], out["confidence"], out["fake_prob"], out["real_prob"]

    def predict_raw(self, text_vector) -> int:
        with self._lock:
            if self._model is None:
                return -1
            out = self.predict_proba(text_vector)
            return int(out["prediction"])

    def get_info(self) -> Dict[str, Any]:
        with self._lock:
            info = dict(self._version_meta)
            info["model_ready"] = self._model is not None
            info["is_training"] = self._is_training
            info["fake_threshold"] = self._fake_threshold
            return info

    def hot_swap(
        self,
        new_model,
        new_vectorizer,
        version_meta: Dict[str, Any],
    ) -> None:
        logger.info(
            "[ModelManager] Hot-swapping → version %s accuracy=%.4f",
            version_meta.get("version_id", "?"),
            version_meta.get("accuracy", 0.0) or 0.0,
        )

        with self._lock:
            self._model = new_model
            self._vectorizer = new_vectorizer
            self._fake_threshold = float(
                version_meta.get("fake_threshold", DEFAULT_FAKE_THRESHOLD)
            )
            self._version_meta.update(version_meta)
            self._sync_class_index()

        logger.info("[ModelManager] Hot-swap complete.")

    def save_versioned(
        self,
        model,
        vectorizer,
        version_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str]:
        model_filename = f"model_{version_id}.pkl"
        vectorizer_filename = f"vectorizer_{version_id}.pkl"

        versioned_model_path = os.path.join(MODELS_DIR, model_filename)
        versioned_vectorizer_path = os.path.join(MODELS_DIR, vectorizer_filename)

        with open(versioned_model_path, "wb") as f:
            pickle.dump(model, f)
        with open(versioned_vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        shutil.copy2(versioned_model_path, MODEL_PATH)
        shutil.copy2(versioned_vectorizer_path, VECTORIZER_PATH)

        if meta is not None:
            with open(META_PATH, "wb") as f:
                pickle.dump(meta, f)
            meta_copy = os.path.join(MODELS_DIR, f"model_{version_id}_meta.pkl")
            with open(meta_copy, "wb") as f:
                pickle.dump(meta, f)

        logger.info("[ModelManager] Saved versioned model: %s", versioned_model_path)
        return versioned_model_path, versioned_vectorizer_path

    def set_training(self, state: bool) -> None:
        self._is_training = state


manager = ModelManager()
