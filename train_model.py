"""
train_model.py
--------------
Train and save the Fake News Detector model.

Run:
  python generate_dataset.py   # if CSVs missing
  python train_model.py
"""

import os
import sys

from core.training import BASE_DIR, save_artifacts, train_pipeline

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")


def train() -> float:
    print("=" * 55)
    print("  Fake News Detector — Training")
    print("=" * 55)

    result = train_pipeline()

    meta = {
        "fake_threshold": result["fake_threshold"],
        "accuracy": result["accuracy"],
        "fake_f1": result["fake_f1"],
        "model_name": result["model_name"],
        "cv_f1_macro": result["cv_f1_macro"],
        "samples": result["samples"],
        "fake_count": result["fake_count"],
        "real_count": result["real_count"],
    }

    save_artifacts(
        result["model"],
        result["vectorizer"],
        MODEL_PATH,
        VECTORIZER_PATH,
        meta=meta,
    )

    print(f"\n  Samples       : {result['samples']}")
    print(f"  Fake / Real   : {result['fake_count']} / {result['real_count']}")
    print(f"  Model         : {result['model_name']}")
    print(f"  CV F1 (macro) : {result['cv_f1_macro'] * 100:.2f}%")
    print(f"  Test accuracy : {result['accuracy'] * 100:.2f}%")
    print(f"  Fake F1       : {result['fake_f1'] * 100:.2f}%")
    print(f"  Fake threshold: {result['fake_threshold']:.3f}")
    print("\nClassification Report:")
    print(result["classification_report"])
    print("Confusion Matrix (rows=true, cols=pred):")
    print(result["confusion_matrix"])
    print(f"\n[OK] Model saved      -> {MODEL_PATH}")
    print(f"[OK] Vectorizer saved -> {VECTORIZER_PATH}")
    print(f"[OK] Metadata saved   -> {MODEL_PATH.replace('.pkl', '_meta.pkl')}")
    print("\nTraining complete!  Run  python app.py  to start the server.")
    return result["accuracy"]


if __name__ == "__main__":
    try:
        train()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
