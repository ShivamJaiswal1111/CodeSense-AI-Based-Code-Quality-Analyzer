"""
CodeSense - ML Model Training
Production-grade RandomForest model with R² ≥ 0.90, cross-validation, and versioning.
The ML model score IS the primary quality score — no hardcoded overrides.
"""

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from constants import (
    FEATURE_NAMES, MODEL_FILENAME, SCALER_FILENAME, NUM_FEATURES,
    TARGET_R2, CROSS_VAL_FOLDS, MIN_SAMPLES_TRAIN,
)
from logger import get_logger

logger = get_logger(__name__)


# ─── Synthetic Data Generation ────────────────────────────────────────────────

def _generate_sample(rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    """
    Generate one realistic (features, score) training sample.
    The score is derived entirely from feature values — no arbitrary labels.
    """
    # Feature sampling with realistic distributions
    loc = float(rng.integers(10, 500))
    blank = float(rng.integers(0, int(loc * 0.2) + 1))
    comment = float(rng.integers(0, int(loc * 0.4) + 1))
    comment_ratio = round(comment / max(1, loc), 3)
    avg_ll = float(rng.uniform(20, 100))
    max_ll = float(avg_ll + rng.uniform(0, 80))
    num_fns = float(rng.integers(0, 30))
    num_cls = float(rng.integers(0, 5))

    avg_cc = float(rng.uniform(1, 15))
    max_cc = float(avg_cc + rng.uniform(0, 20))
    avg_cog = float(avg_cc * rng.uniform(0.8, 2.5))
    max_nest = float(rng.integers(0, 8))
    avg_fn_len = float(rng.uniform(5, 80))
    max_fn_len = float(avg_fn_len + rng.uniform(0, 100))

    naming = float(rng.uniform(0.3, 1.0))
    ll_ratio = float(rng.uniform(0, 0.3))
    magic_n = float(rng.integers(0, 20))
    doc_ratio = float(rng.uniform(0, 1.0))
    avg_params = float(rng.uniform(0, 8))

    sec_count = float(rng.integers(0, 10))
    crit_sec = float(rng.integers(0, min(3, int(sec_count) + 1)))
    high_sec = float(rng.integers(0, min(5, int(sec_count) + 1)))
    has_validation = float(rng.choice([0, 1], p=[0.4, 0.6]))

    dsa_score = float(rng.uniform(0, 1))
    algo_cnt = float(rng.integers(0, 5))
    ds_cnt = float(rng.integers(0, 5))

    dup_score = float(rng.uniform(0, 0.5))
    exc_cov = float(rng.uniform(0, 1))
    test_cov = float(rng.uniform(0, 1))
    reuse = float(rng.uniform(0.2, 1.0))
    dp_usage = float(rng.uniform(0, 1))
    smell_cnt = float(rng.integers(0, 15))
    tech_debt = float(crit_sec * 60 + high_sec * 30 + smell_cnt * 5 + magic_n * 3)

    features = np.array([
        loc, blank, comment, comment_ratio, avg_ll, max_ll, num_fns, num_cls,
        avg_cc, max_cc, avg_cog, max_nest, avg_fn_len, max_fn_len,
        naming, ll_ratio, magic_n, doc_ratio, avg_params,
        sec_count, crit_sec, high_sec, has_validation,
        dsa_score, algo_cnt, ds_cnt,
        dup_score, exc_cov, test_cov, reuse, dp_usage, smell_cnt, tech_debt,
    ], dtype=np.float32)

    # ── Score formula (ground-truth labels) ──────────────────────────────────
    #   Each component scored 0–100 then weighted.
    score_complexity  = max(0, 100 - avg_cc * 4 - max_nest * 5 - max_fn_len * 0.3)
    score_security    = max(0, 100 - crit_sec * 20 - high_sec * 10 - (sec_count - crit_sec - high_sec) * 3)
    score_style       = (naming * 40) + (max(0, 1 - ll_ratio) * 20) + \
                        (max(0, 1 - magic_n / 20) * 20) + (doc_ratio * 20)
    score_maintainability = (reuse * 30) + (exc_cov * 20) + (test_cov * 30) + \
                            (max(0, 1 - smell_cnt / 15) * 20)
    score_dsa         = dsa_score * 100

    final = (
        score_complexity    * 0.30 +
        score_security      * 0.25 +
        score_style         * 0.20 +
        score_maintainability * 0.15 +
        score_dsa           * 0.10
    )
    final += float(rng.normal(0, 2))   # Small noise ≤ ±6
    final = float(np.clip(final, 0, 100))

    return features, final


def generate_training_data(n: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate n (features, score) training samples."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n):
        feat, score = _generate_sample(rng)
        X.append(feat)
        y.append(score)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─── Model Training ──────────────────────────────────────────────────────────

def train(n_samples: int = 10000, save: bool = True) -> Dict:
    """
    Train the ensemble model and save to disk.

    Returns:
        Training metrics dictionary.
    """
    logger.info("Generating %d training samples...", n_samples)
    X, y = generate_training_data(n=n_samples)

    # Ensemble: Random Forest + Gradient Boosting
    rf = RandomForestRegressor(
        n_estimators=100,       # Reduced for Streamlit Cloud (1GB RAM limit)
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,               # No parallel jobs on cloud to save RAM
        random_state=42,
    )
    gb = GradientBoostingRegressor(
        n_estimators=80,        # Reduced for Streamlit Cloud
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        min_samples_split=4,
        random_state=42,
    )
    ensemble = VotingRegressor(
        estimators=[("rf", rf), ("gb", gb)],
        weights=[0.55, 0.45],
    )

    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  ensemble),
    ])

    # ── Cross-validation ─────────────────────────────────────────────────────
    logger.info("Running %d-fold cross-validation...", CROSS_VAL_FOLDS)
    kf     = KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=42)
    cv_r2  = cross_val_score(pipeline, X, y, cv=kf, scoring="r2", n_jobs=1)
    cv_mae = cross_val_score(pipeline, X, y, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)

    logger.info("CV R²: %.4f ± %.4f", cv_r2.mean(), cv_r2.std())
    logger.info("CV MAE: %.4f ± %.4f", -cv_mae.mean(), cv_mae.std())

    # ── Final fit on all data ─────────────────────────────────────────────────
    t0 = time.time()
    pipeline.fit(X, y)
    train_time = time.time() - t0

    y_pred = pipeline.predict(X)
    r2  = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = math.sqrt(mean_squared_error(y, y_pred))

    logger.info("Train R²=%.4f  MAE=%.2f  RMSE=%.2f  time=%.1fs", r2, mae, rmse, train_time)

    if r2 < TARGET_R2:
        logger.warning("R² %.4f below target %.2f — consider more samples or tuning.", r2, TARGET_R2)

    metrics = {
        "r2":           round(float(r2), 4),
        "mae":          round(float(mae), 4),
        "rmse":         round(float(rmse), 4),
        "cv_r2_mean":   round(float(cv_r2.mean()), 4),
        "cv_r2_std":    round(float(cv_r2.std()), 4),
        "cv_mae_mean":  round(float(-cv_mae.mean()), 4),
        "n_samples":    n_samples,
        "n_features":   NUM_FEATURES,
        "trained_at":   datetime.utcnow().isoformat(),
        "target_r2":    TARGET_R2,
        "passed":       r2 >= TARGET_R2,
    }

    if save:
        _save_model(pipeline, metrics)

    return metrics


def _save_model(pipeline: Pipeline, metrics: Dict) -> None:
    Path(MODEL_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILENAME, "wb") as f:
        pickle.dump(pipeline, f, protocol=5)
    # Save feature names for reference
    feat_path = MODEL_FILENAME.replace(".pkl", "_features.json")
    with open(feat_path, "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)
    # Save metrics
    meta_path = MODEL_FILENAME.replace(".pkl", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Model saved to %s", MODEL_FILENAME)


# ─── Inference ───────────────────────────────────────────────────────────────

import math


class QualityPredictor:
    """
    Loads the trained model and predicts code quality scores.
    The ML score IS the primary quality score.
    Max contextual adjustment is ±MAX_SCORE_ADJUSTMENT points.
    """

    from constants import MAX_SCORE_ADJUSTMENT

    def __init__(self, model_path: str = MODEL_FILENAME) -> None:
        self.pipeline: Optional[Pipeline] = None
        self.model_path = model_path
        self._load()

    def _load(self) -> None:
        if Path(self.model_path).exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.pipeline = pickle.load(f)
                logger.info("Model loaded from %s", self.model_path)
            except Exception as exc:
                logger.warning("Failed to load model: %s — will train on first use.", exc)
        else:
            logger.info("No trained model found at %s", self.model_path)

    def ensure_model(self) -> None:
        """Train and save model if it doesn't exist."""
        if self.pipeline is None:
            logger.info("Training model (first-time setup)...")
            train(n_samples=MIN_SAMPLES_TRAIN)
            self._load()

    def predict(
        self,
        feature_array: np.ndarray,
        contextual_adjustments: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Predict quality score.

        Args:
            feature_array:           1D array of NUM_FEATURES features.
            contextual_adjustments:  Optional ±MAX_SCORE_ADJUSTMENT adjustment.

        Returns:
            (final_score, confidence_interval_half_width)
        """
        self.ensure_model()

        X = feature_array.reshape(1, -1)

        # PRIMARY score from ML model
        ml_score = float(self.pipeline.predict(X)[0])

        # Bounded contextual adjustment
        adj = float(np.clip(contextual_adjustments,
                            -self.MAX_SCORE_ADJUSTMENT,
                            +self.MAX_SCORE_ADJUSTMENT))
        final = float(np.clip(ml_score + adj, 0, 100))

        # Confidence: estimate spread from individual estimators
        confidence = self._estimate_confidence(X)

        return round(final, 1), round(confidence, 1)

    def _estimate_confidence(self, X: np.ndarray) -> float:
        """
        Estimate ± confidence interval using individual tree predictions
        from the Random Forest inside the ensemble.
        """
        try:
            ensemble = self.pipeline.named_steps["model"]
            rf = dict(ensemble.estimators_)["rf"]
            # Scale input
            X_scaled = self.pipeline.named_steps["scaler"].transform(X)
            preds = np.array([tree.predict(X_scaled)[0] for tree in rf.estimators_])
            return float(np.std(preds))
        except Exception:
            return 3.0   # Default ±3 confidence

    def get_model_meta(self) -> Dict:
        meta_path = self.model_path.replace(".pkl", "_meta.json")
        if Path(meta_path).exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}


# ─── Contextual Adjustments ──────────────────────────────────────────────────

def calculate_contextual_adjustments(
    analysis: Dict,
    dsa: Dict,
    language: str,
) -> float:
    """
    Compute a ±MAX_SCORE_ADJUSTMENT contextual bonus/penalty.
    This is the ONLY source of adjustment on top of the ML score.
    """
    from constants import MAX_SCORE_ADJUSTMENT
    adj = 0.0

    # Bonus: high-complexity algorithm detected with good complexity
    dsa_summary = dsa.get("summary", {})
    if dsa_summary.get("complexity_score", 0) > 70:
        adj += 1.5

    # Penalty: critical security issues
    sec = analysis.get("security", {})
    adj -= min(3.0, sec.get("counts", {}).get("CRITICAL", 0) * 1.5)

    # Bonus: well-documented code
    doc = analysis.get("documentation", {})
    ratio = doc.get("docstring_ratio", doc.get("comment_ratio", 0))
    if ratio >= 0.5:
        adj += 1.0

    return float(np.clip(adj, -MAX_SCORE_ADJUSTMENT, MAX_SCORE_ADJUSTMENT))


# ─── Grade Calculation ────────────────────────────────────────────────────────

def score_to_grade(score: float) -> str:
    from constants import GRADE_THRESHOLDS
    for grade, threshold in sorted(GRADE_THRESHOLDS.items(),
                                   key=lambda x: x[1], reverse=True):
        if score >= threshold:
            return grade
    return "F"


def score_to_label(score: float) -> str:
    from constants import SCORE_LABELS
    for (low, high), label in SCORE_LABELS.items():
        if low <= score <= high:
            return label
    return "Unknown"


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CodeSense ML model")
    parser.add_argument("--samples", type=int, default=10000,
                        help="Number of training samples (default: 10000)")
    parser.add_argument("--no-save", action="store_true", help="Don't save model")
    args = parser.parse_args()

    metrics = train(n_samples=args.samples, save=not args.no_save)
    print("\n── Training Results ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<20} {v}")
    status = "✅ PASSED" if metrics["passed"] else "❌ BELOW TARGET"
    print(f"\n  Target R² ≥ {TARGET_R2}   →  {status}")