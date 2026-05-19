"""
core/prediction.py
------------------
Unified prediction logic for all API routes.

Combines ML probabilities, tuned fake-class threshold, confidence handling,
and heuristic pattern detection.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from core.heuristic import analyze as heuristic_analyze
from core.model_manager import manager
from core.preprocessor import preprocess

# Minimum max-class probability before we trust the ML label alone
CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "52.0"))

# When ML is uncertain but heuristics strongly indicate fake
HEURISTIC_OVERRIDE_BIAS = 18


def predict_text(raw_text: str) -> Dict[str, Any]:
    """
    Run full prediction on raw article text.

    Returns
    -------
    dict with keys: result, confidence, fake_prob, real_prob, signals,
                   ml_label, heuristic_bias, uncertain (bool)
    """
    if not manager.is_ready:
        raise RuntimeError("Model not loaded.")

    processed = preprocess(raw_text)
    if not processed:
        raise ValueError("Text could not be processed after cleaning.")

    vec = manager.vectorize(processed)
    ml = manager.predict_proba(vec)

    h_res = _safe_heuristic(raw_text)
    bias = h_res.get("fake_bias", 0)

    fake_p, real_p = _blend_probs(ml["fake_prob"], ml["real_prob"], bias)
    label = _final_label(
        fake_p=fake_p,
        real_p=real_p,
        ml_label=ml["label"],
        ml_confidence=ml["confidence"],
        bias=bias,
    )
    confidence = round(max(fake_p, real_p), 2)
    uncertain = confidence < CONFIDENCE_THRESHOLD

    return {
        "result": label,
        "confidence": confidence,
        "fake_prob": fake_p,
        "real_prob": real_p,
        "signals": {
            "fake_signals": h_res.get("fake_signals", []),
            "real_signals": h_res.get("real_signals", []),
        },
        "ml_label": ml["label"],
        "heuristic_bias": bias,
        "uncertain": uncertain,
    }


def _safe_heuristic(raw_text: str) -> dict:
    try:
        return heuristic_analyze(raw_text)
    except Exception:
        return {"fake_bias": 0, "fake_signals": [], "real_signals": []}


def _blend_probs(
    fake_p: float, real_p: float, bias: int
) -> Tuple[float, float]:
    """Apply heuristic bias and re-normalise to percentages."""
    new_fake = max(0.0, min(100.0, fake_p + bias))
    new_real = max(0.0, min(100.0, real_p - bias))
    total = new_fake + new_real
    if total <= 0:
        return 50.0, 50.0
    return (
        round((new_fake / total) * 100, 2),
        round((new_real / total) * 100, 2),
    )


def _final_label(
    fake_p: float,
    real_p: float,
    ml_label: str,
    ml_confidence: float,
    bias: int,
) -> str:
    """
    Decide Fake vs Real using blended probabilities and safeguards.

    Rules (in order):
      1. Strong heuristic fake bias + borderline ML → Fake
      2. Blended probabilities: Fake if fake_p > real_p (strict >)
      3. Tie-break toward Fake when bias is positive (reduces false Real)
    """
    if bias >= HEURISTIC_OVERRIDE_BIAS and ml_confidence < CONFIDENCE_THRESHOLD + 8:
        return "Fake"

    if fake_p > real_p:
        return "Fake"
    if real_p > fake_p:
        return "Real"

    # Exact tie — prefer Fake when heuristics lean fake, else ML label
    if bias > 0:
        return "Fake"
    return ml_label
