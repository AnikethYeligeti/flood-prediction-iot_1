"""
Flood Prediction Model
Uses a Random Forest Regressor trained on preprocessed IoT sensor features
to predict flood risk score (0–10). Also includes a threshold-based alert classifier.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional, Tuple


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "flood_model.joblib")


# ─────────────────────────────────────────────
# ALERT THRESHOLDS
# ─────────────────────────────────────────────

ALERT_THRESHOLDS = {
    "NORMAL":   (0.0, 3.0),
    "WATCH":    (3.0, 5.0),
    "WARNING":  (5.0, 7.0),
    "DANGER":   (7.0, 8.5),
    "CRITICAL": (8.5, 10.1),
}


def score_to_alert(score: float) -> str:
    for level, (lo, hi) in ALERT_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "CRITICAL"


def score_to_color(score: float) -> str:
    alert = score_to_alert(score)
    return {
        "NORMAL":   "#22c55e",
        "WATCH":    "#84cc16",
        "WARNING":  "#f59e0b",
        "DANGER":   "#f97316",
        "CRITICAL": "#ef4444",
    }.get(alert, "#ef4444")


# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray,
                test_size: float = 0.2,
                random_state: int = 42) -> dict:
    """
    Train a Random Forest on (X, y) arrays from the sliding window preprocessor.
    X shape: (samples, window_size, features) → flattened to (samples, window_size*features)
    """
    if X is None or y is None or len(X) == 0:
        return {"error": "Insufficient data for training"}

    # Flatten 3D windows → 2D
    X_flat = X.reshape(len(X), -1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=test_size, random_state=random_state, shuffle=False  # time-ordered
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "rmse":  round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "mae":   round(float(mean_absolute_error(y_test, y_pred)), 4),
        "r2":    round(float(r2_score(y_test, y_pred)), 4),
        "train_samples": len(X_train),
        "test_samples":  len(X_test),
    }

    # Persist model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return {"model": model, "metrics": metrics, "y_pred": y_pred.tolist(), "y_test": y_test.tolist()}


def load_model() -> Optional[RandomForestRegressor]:
    """Load persisted model if it exists."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

def predict_flood_risk(model, X_window: np.ndarray) -> dict:
    """
    Given a single sliding window (window_size, n_features), predict flood risk.
    Returns score, alert level, and confidence band.
    """
    X_flat = X_window.reshape(1, -1)
    score = float(np.clip(model.predict(X_flat)[0], 0, 10))

    # Estimate uncertainty using individual tree predictions
    tree_preds = np.array([tree.predict(X_flat)[0] for tree in model.estimators_])
    std = float(np.std(tree_preds))

    return {
        "flood_risk_score": round(score, 2),
        "alert_level": score_to_alert(score),
        "color": score_to_color(score),
        "confidence_lower": round(max(0, score - 2 * std), 2),
        "confidence_upper": round(min(10, score + 2 * std), 2),
        "uncertainty_std": round(std, 3),
    }


# ─────────────────────────────────────────────
# DUMMY PREDICTION (no model trained yet)
# ─────────────────────────────────────────────

def heuristic_flood_risk(rain_mm_hr: float, water_level_m: float,
                          rain_accum_6h_mm: float = 0,
                          danger_level_m: float = 4.5) -> dict:
    """
    Fallback heuristic prediction when ML model is not yet trained.
    Used for live demo before a full training cycle.
    """
    wl_ratio  = min(water_level_m / danger_level_m, 1.33)  # cap at 1.33× danger level
    rain_norm = min(rain_mm_hr / 64.5, 1.0)               # 64.5 mm/hr = very heavy rain
    accum_norm = min(rain_accum_6h_mm / 150.0, 1.0)

    score = (wl_ratio * 5.0 + rain_norm * 2.5 + accum_norm * 2.5)
    score = round(float(np.clip(score, 0, 10)), 2)

    return {
        "flood_risk_score": score,
        "alert_level": score_to_alert(score),
        "color": score_to_color(score),
        "method": "heuristic",
        "confidence_lower": max(0, score - 1.5),
        "confidence_upper": min(10, score + 1.5),
    }
