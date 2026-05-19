"""
core/training.py
----------------
Shared training pipeline for train_model.py and scheduler/retrainer.py.

Features:
  - Balanced dataset (undersample majority class)
  - Optimised TF-IDF (trigrams, sublinear_tf, df filtering)
  - Model selection: balanced LogisticRegression vs calibrated LinearSVC
  - Fake-class decision threshold tuned on validation split
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import LinearSVC

from core.preprocessor import preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_CSV = os.path.join(BASE_DIR, "dataset", "Fake.csv")
TRUE_CSV = os.path.join(BASE_DIR, "dataset", "True.csv")

LABEL_FAKE = 0
LABEL_REAL = 1

# Default fake probability threshold (tuned per train run; stored with model meta)
DEFAULT_FAKE_THRESHOLD = 0.45


def build_vectorizer() -> TfidfVectorizer:
    """TF-IDF settings tuned for short news headlines + article bodies."""
    return TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.92,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b[a-z][a-z0-9']+\b",
    )


def load_dataframe(
    fake_csv: str = FAKE_CSV,
    true_csv: str = TRUE_CSV,
    extra_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load Fake/True CSVs, merge optional extra rows, balance classes."""
    if not os.path.exists(fake_csv) or not os.path.exists(true_csv):
        raise FileNotFoundError(
            "Dataset CSVs not found. Run: python generate_dataset.py"
        )

    fake_df = pd.read_csv(fake_csv)
    true_df = pd.read_csv(true_csv)
    fake_df["label"] = LABEL_FAKE
    true_df["label"] = LABEL_REAL

    frames = []
    for df in (fake_df, true_df):
        title = df["title"].fillna("") if "title" in df.columns else ""
        body = df["text"].fillna("") if "text" in df.columns else ""
        df = df.copy()
        df["content"] = (title.astype(str) + " " + body.astype(str)).str.strip()
        frames.append(df[["content", "label"]])

    combined = pd.concat(frames, ignore_index=True)

    if extra_df is not None and not extra_df.empty:
        extra = extra_df[["content", "label"]].dropna()
        combined = pd.concat([combined, extra], ignore_index=True)

    combined = combined.dropna(subset=["content", "label"])
    combined = combined[combined["content"].str.strip().str.len() >= 20]
    combined = combined.drop_duplicates(subset=["content"])
    combined = _balance_classes(combined)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    return combined


def _balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """Undersample majority class so Fake and Real counts match."""
    fake = df[df["label"] == LABEL_FAKE]
    real = df[df["label"] == LABEL_REAL]
    n = min(len(fake), len(real))
    if n == 0:
        return df
    fake = fake.sample(n=n, random_state=42)
    real = real.sample(n=n, random_state=42)
    return pd.concat([fake, real], ignore_index=True)


def _fake_class_index(model) -> int:
    classes = list(getattr(model, "classes_", [LABEL_FAKE, LABEL_REAL]))
    return classes.index(LABEL_FAKE)


def tune_fake_threshold(
    model,
    X_val,
    y_val: np.ndarray,
    min_fake_recall: float = 0.82,
) -> float:
    """
    Pick a decision threshold on P(fake) that improves fake recall
  without collapsing overall accuracy.
    """
    fake_idx = _fake_class_index(model)
    probas = model.predict_proba(X_val)[:, fake_idx]
    y_fake = (y_val == LABEL_FAKE).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(y_fake, probas)
    best_t = DEFAULT_FAKE_THRESHOLD
    best_f1 = -1.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r < min_fake_recall:
            continue
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    # If no point met min recall, use threshold that maximises fake F1
    if best_f1 < 0:
        for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

    return float(np.clip(best_t, 0.35, 0.55))


def _select_model(X_train, y_train):
    """Compare balanced LR vs calibrated LinearSVC; return best estimator."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = LogisticRegression(
        C=4.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )
    lr_scores = cross_val_score(
        lr, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )

    svc_base = LinearSVC(
        C=0.8,
        class_weight="balanced",
        max_iter=3000,
        random_state=42,
    )
    svc = CalibratedClassifierCV(svc_base, cv=3, method="sigmoid")
    svc_scores = cross_val_score(
        svc, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )

    if svc_scores.mean() >= lr_scores.mean():
        return svc, "LinearSVC_calibrated", svc_scores.mean()
    return lr, "LogisticRegression", lr_scores.mean()


def train_pipeline(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Full train → evaluate → return artifacts.

    Returns dict with keys:
      model, vectorizer, accuracy, fake_threshold, metrics, sample_counts
    """
    if df is None:
        df = load_dataframe()

    df = df.copy()
    df["processed"] = df["content"].apply(preprocess)
    df = df[df["processed"].str.strip() != ""]

    X = df["processed"]
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Hold out part of train for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=random_state,
        stratify=y_train,
    )

    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    model, model_name, cv_f1 = _select_model(X_train_vec, y_train)
    model.fit(X_train_vec, y_train)

    fake_threshold = tune_fake_threshold(model, X_val_vec, y_val)

    fake_idx = _fake_class_index(model)
    probas_test = model.predict_proba(X_test_vec)[:, fake_idx]
    y_pred = np.where(probas_test >= fake_threshold, LABEL_FAKE, LABEL_REAL)

    accuracy = float(accuracy_score(y_test, y_pred))
    fake_f1 = float(
        f1_score(y_test, y_pred, pos_label=LABEL_FAKE, zero_division=0)
    )
    report = classification_report(
        y_test, y_pred, target_names=["Fake", "Real"], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "fake_f1": fake_f1,
        "fake_threshold": fake_threshold,
        "model_name": model_name,
        "cv_f1_macro": float(cv_f1),
        "classification_report": report,
        "confusion_matrix": cm,
        "samples": len(df),
        "fake_count": int((df["label"] == LABEL_FAKE).sum()),
        "real_count": int((df["label"] == LABEL_REAL).sum()),
    }


def save_artifacts(
    model,
    vectorizer,
    model_path: str,
    vectorizer_path: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist model, vectorizer, and optional metadata pickle."""
    import pickle

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    if meta is not None:
        meta_path = model_path.replace(".pkl", "_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
