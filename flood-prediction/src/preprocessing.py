"""
Data Preprocessing Pipeline
Handles all preprocessing steps for IoT sensor data:
- Missing value imputation
- Outlier detection & removal
- Normalization / Scaling
- Feature engineering
- Sliding window time-series construction
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────
# 1. RAW DATA FLATTENING
# ─────────────────────────────────────────────

def flatten_batch(batch: dict) -> dict:
    """
    Flatten a single IoT batch reading into a tabular row.
    Aggregates multi-sensor readings into summary statistics.
    """
    rain_readings = batch.get("rainfall_readings", [])
    wl_readings = batch.get("water_level_readings", [])

    # Rainfall: collect valid (non-faulty) values
    rain_values = [
        r["rainfall_mm_hr"] for r in rain_readings
        if r.get("status") == "OK" and r.get("rainfall_mm_hr") is not None
    ]
    wl_values = [
        r["water_level_m"] for r in wl_readings
        if r.get("status") == "OK" and r.get("water_level_m") is not None
    ]

    row = {
        "timestamp": batch.get("batch_timestamp"),
        # Rainfall features
        "rain_mean_mm_hr": np.mean(rain_values) if rain_values else np.nan,
        "rain_max_mm_hr": np.max(rain_values) if rain_values else np.nan,
        "rain_min_mm_hr": np.min(rain_values) if rain_values else np.nan,
        "rain_std_mm_hr": np.std(rain_values) if len(rain_values) > 1 else 0.0,
        "rain_sensor_count": len(rain_values),
        # Water level features
        "wl_mean_m": np.mean(wl_values) if wl_values else np.nan,
        "wl_max_m": np.max(wl_values) if wl_values else np.nan,
        "wl_min_m": np.min(wl_values) if wl_values else np.nan,
        "wl_std_m": np.std(wl_values) if len(wl_values) > 1 else 0.0,
        "wl_sensor_count": len(wl_values),
        # Alert flags
        "any_critical": int(any(r.get("alert") == "CRITICAL" for r in wl_readings)),
        "any_danger":   int(any(r.get("alert") in ["DANGER", "CRITICAL"] for r in wl_readings)),
        "fault_count":  sum(1 for r in rain_readings + wl_readings if r.get("status") == "FAULT"),
    }
    return row


def batches_to_dataframe(batches: list) -> pd.DataFrame:
    """Convert a list of IoT batch readings into a pandas DataFrame."""
    rows = [flatten_batch(b) for b in batches]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# 2. MISSING VALUE HANDLING
# ─────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
    - Short gaps (≤ 3 consecutive) → linear interpolation
    - Longer gaps → forward-fill then median imputation
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Linear interpolation for short gaps
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=3, limit_direction="both")

    # Forward fill for slightly longer gaps
    df[numeric_cols] = df[numeric_cols].ffill(limit=6)

    # Final fallback: median imputation
    imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df


# ─────────────────────────────────────────────
# 3. OUTLIER DETECTION (IQR + Z-Score)
# ─────────────────────────────────────────────

def detect_and_remove_outliers(df: pd.DataFrame,
                                cols: Optional[List[str]] = None,
                                z_thresh: float = 3.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detects outliers using modified Z-score (robust to skew).
    Returns cleaned DataFrame and a DataFrame of removed outlier rows.
    """
    df = df.copy()
    if cols is None:
        cols = ["rain_mean_mm_hr", "rain_max_mm_hr", "wl_mean_m", "wl_max_m"]

    outlier_mask = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            continue
        modified_z = 0.6745 * (df[col] - median) / mad
        outlier_mask |= modified_z.abs() > z_thresh

    outliers = df[outlier_mask].copy()
    cleaned = df[~outlier_mask].copy()
    return cleaned, outliers


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives time-domain and domain-specific features:
    - Rolling rainfall accumulation (1h, 3h, 6h, 12h, 24h)
    - Water level rate of change (velocity)
    - Hour of day / day of week (cyclical encoding)
    - Rainfall intensity class
    """
    df = df.copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Approximate rows per hour (10-min intervals → 6 per hour)
    rph = 6

    # Rolling rainfall accumulation
    for hours in [1, 3, 6, 12, 24]:
        window = hours * rph
        df[f"rain_accum_{hours}h_mm"] = (
            df["rain_mean_mm_hr"].rolling(window, min_periods=1).mean() * hours
        )

    # Water level velocity (rate of change, m per 10 min)
    df["wl_velocity_m_per_10min"] = df["wl_mean_m"].diff().fillna(0)
    df["wl_acceleration"] = df["wl_velocity_m_per_10min"].diff().fillna(0)

    # Rolling max water level (6h window)
    df["wl_rolling_max_6h"] = df["wl_max_m"].rolling(rph * 6, min_periods=1).max()

    # Time features (cyclical encoding to preserve periodicity)
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)

    # Rainfall intensity classification (IMD scale)
    bins   = [-1, 0, 2.5, 7.5, 35.5, 64.5, 115.5, np.inf]
    labels = [0, 1, 2, 3, 4, 5, 6]  # No rain → Extremely heavy
    df["rain_intensity_class"] = pd.cut(df["rain_mean_mm_hr"], bins=bins, labels=labels).astype(int)

    # Composite flood risk score (heuristic, 0–10)
    wl_norm  = df["wl_mean_m"].clip(0, 8) / 8
    rain_norm = df["rain_accum_6h_mm"].clip(0, 200) / 200
    df["flood_risk_score"] = ((wl_norm * 0.6 + rain_norm * 0.4) * 10).round(2)

    return df


# ─────────────────────────────────────────────
# 5. NORMALIZATION
# ─────────────────────────────────────────────

FEATURE_COLS = [
    "rain_mean_mm_hr", "rain_max_mm_hr", "rain_std_mm_hr",
    "wl_mean_m", "wl_max_m", "wl_velocity_m_per_10min", "wl_acceleration",
    "rain_accum_1h_mm", "rain_accum_3h_mm", "rain_accum_6h_mm",
    "rain_accum_12h_mm", "rain_accum_24h_mm",
    "wl_rolling_max_6h", "rain_intensity_class",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "flood_risk_score",
]


def normalize_features(df: pd.DataFrame,
                        scaler: Optional[MinMaxScaler] = None
                        ) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Apply Min-Max scaling to feature columns. Returns scaled df and fitted scaler."""
    df = df.copy()
    cols = [c for c in FEATURE_COLS if c in df.columns]

    if scaler is None:
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
    else:
        df[cols] = scaler.transform(df[cols])

    return df, scaler


# ─────────────────────────────────────────────
# 6. SLIDING WINDOW (TIME-SERIES)
# ─────────────────────────────────────────────

def create_sliding_windows(df: pd.DataFrame,
                            window_size: int = 18,
                            target_col: str = "flood_risk_score"
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates (X, y) arrays for supervised time-series learning.
    window_size=18 → 18 × 10min = 3 hours of look-back.
    """
    feature_cols = [c for c in FEATURE_COLS if c in df.columns and c != target_col]
    X, y = [], []

    values = df[feature_cols].values
    targets = df[target_col].values

    for i in range(window_size, len(df)):
        X.append(values[i - window_size:i])
        y.append(targets[i])

    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# 7. FULL PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing_pipeline(batches: list,
                                scaler: Optional[MinMaxScaler] = None,
                                window_size: int = 18
                                ) -> dict:
    """
    End-to-end preprocessing pipeline.
    Returns a dict with all intermediate and final outputs for transparency.
    """
    # Step 1: Flatten
    raw_df = batches_to_dataframe(batches)

    # Step 2: Missing value handling
    imputed_df = handle_missing_values(raw_df)

    # Step 3: Outlier removal
    cleaned_df, outliers_df = detect_and_remove_outliers(imputed_df)

    # Step 4: Feature engineering
    featured_df = engineer_features(cleaned_df)

    # Step 5: Normalization
    normalized_df, scaler = normalize_features(featured_df, scaler)

    # Step 6: Sliding windows (if enough rows)
    X, y = None, None
    if len(normalized_df) >= window_size + 1:
        X, y = create_sliding_windows(normalized_df, window_size)

    return {
        "raw_df": raw_df,
        "imputed_df": imputed_df,
        "cleaned_df": cleaned_df,
        "outliers_df": outliers_df,
        "featured_df": featured_df,
        "normalized_df": normalized_df,
        "scaler": scaler,
        "X": X,
        "y": y,
        "stats": {
            "raw_rows": len(raw_df),
            "after_imputation": len(imputed_df),
            "after_outlier_removal": len(cleaned_df),
            "outliers_removed": len(outliers_df),
            "features_engineered": len([c for c in FEATURE_COLS if c in featured_df.columns]),
            "windows_created": len(X) if X is not None else 0,
        },
    }
