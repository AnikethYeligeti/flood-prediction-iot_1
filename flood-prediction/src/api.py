"""
Flask REST API
Exposes endpoints for the IoT Flood Prediction system.
"""

import json
import numpy as np
from flask import Flask, jsonify, request, render_template
from datetime import datetime

from src.iot_sensors import IoTSensorNetwork
from src.preprocessing import run_preprocessing_pipeline, FEATURE_COLS
from src.model import train_model, load_model, predict_flood_risk, heuristic_flood_risk


# Shared state (in production, use Redis or a DB)
_sensor_network = IoTSensorNetwork()
_model = None
_scaler = None
_last_result = None


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    # ── Serve Dashboard ──────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    # ── Live Sensor Readings ──────────────────────────────────────────────────

    @app.route("/api/sensors/live", methods=["GET"])
    def live_sensors():
        """Return a single batch of live sensor readings."""
        batch = _sensor_network.read_all()
        return jsonify(batch)

    # ── Run Full Preprocessing Pipeline ──────────────────────────────────────

    @app.route("/api/preprocess", methods=["GET"])
    def preprocess():
        """
        Generate 48h of historical sensor data and run the full preprocessing pipeline.
        Returns stats and the latest normalized row for prediction.
        """
        global _scaler, _last_result

        hours = int(request.args.get("hours", 48))
        batches = _sensor_network.generate_historical_data(hours=hours)

        result = run_preprocessing_pipeline(batches, scaler=_scaler)
        _scaler = result["scaler"]
        _last_result = result

        # Serialize only what's needed (DataFrames → JSON)
        stats = result["stats"]
        featured = result["featured_df"]

        # Latest preprocessed row summary
        latest = {}
        if not featured.empty:
            row = featured.iloc[-1]
            latest = {
                "timestamp": str(row.get("timestamp", "")),
                "rain_mean_mm_hr": round(float(row.get("rain_mean_mm_hr", 0)), 2),
                "rain_accum_6h_mm": round(float(row.get("rain_accum_6h_mm", 0)), 2),
                "wl_mean_m": round(float(row.get("wl_mean_m", 0)), 3),
                "wl_max_m": round(float(row.get("wl_max_m", 0)), 3),
                "flood_risk_score": round(float(row.get("flood_risk_score", 0)), 2),
                "rain_intensity_class": int(row.get("rain_intensity_class", 0)),
                "wl_velocity_m_per_10min": round(float(row.get("wl_velocity_m_per_10min", 0)), 4),
            }

        # Time-series chart data (last 72 rows = 12 hours)
        chart_rows = featured.tail(72)
        chart_data = {
            "timestamps": chart_rows["timestamp"].astype(str).tolist(),
            "rain_mean":  _safe_list(chart_rows, "rain_mean_mm_hr"),
            "wl_mean":    _safe_list(chart_rows, "wl_mean_m"),
            "risk_score": _safe_list(chart_rows, "flood_risk_score"),
            "rain_accum_6h": _safe_list(chart_rows, "rain_accum_6h_mm"),
        }

        return jsonify({"stats": stats, "latest": latest, "chart_data": chart_data})

    # ── Train Model ───────────────────────────────────────────────────────────

    @app.route("/api/train", methods=["POST"])
    def train():
        """Train the ML model using the most recently preprocessed data."""
        global _model, _last_result

        if _last_result is None:
            return jsonify({"error": "Run /api/preprocess first"}), 400

        X = _last_result.get("X")
        y = _last_result.get("y")

        if X is None or len(X) < 20:
            return jsonify({"error": "Not enough windowed samples. Try preprocessing more hours."}), 400

        result = train_model(X, y)
        if "error" in result:
            return jsonify(result), 400

        _model = result["model"]
        return jsonify({
            "message": "Model trained successfully",
            "metrics": result["metrics"],
            "sample_predictions": list(zip(
                [round(v, 2) for v in result["y_test"][:10]],
                [round(v, 2) for v in result["y_pred"][:10]]
            )),
        })

    # ── Predict ───────────────────────────────────────────────────────────────

    @app.route("/api/predict", methods=["GET"])
    def predict():
        """Predict current flood risk using latest sensor data."""
        global _model, _last_result, _scaler

        # Always fetch fresh live sensor data for prediction
        batch = _sensor_network.read_all()
        rain_readings = batch.get("rainfall_readings", [])
        wl_readings   = batch.get("water_level_readings", [])

        rain_vals = [r["rainfall_mm_hr"] for r in rain_readings if r.get("status") == "OK" and r.get("rainfall_mm_hr") is not None]
        wl_vals   = [r["water_level_m"] for r in wl_readings if r.get("status") == "OK" and r.get("water_level_m") is not None]

        rain_mean = float(np.mean(rain_vals)) if rain_vals else 0.0
        wl_mean   = float(np.mean(wl_vals))   if wl_vals   else 0.0

        # Try ML model first, fall back to heuristic
        if _model is not None and _last_result is not None:
            X = _last_result.get("X")
            if X is not None and len(X) > 0:
                prediction = predict_flood_risk(_model, X[-1])
                prediction["method"] = "ml_model"
                return jsonify(prediction)

        # Heuristic fallback
        prediction = heuristic_flood_risk(
            rain_mm_hr=rain_mean,
            water_level_m=wl_mean,
            rain_accum_6h_mm=rain_mean * 6,
        )
        return jsonify(prediction)

    # ── Outlier Report ────────────────────────────────────────────────────────

    @app.route("/api/outliers", methods=["GET"])
    def outliers():
        if _last_result is None:
            return jsonify({"error": "Run /api/preprocess first"}), 400
        df = _last_result["outliers_df"]
        cols = ["timestamp", "rain_mean_mm_hr", "wl_mean_m"]
        cols = [c for c in cols if c in df.columns]
        records = df[cols].head(20).to_dict(orient="records")
        for r in records:
            r["timestamp"] = str(r.get("timestamp", ""))
        return jsonify({"count": len(df), "samples": records})

    # ── Health Check ──────────────────────────────────────────────────────────

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": _model is not None,
            "preprocessed": _last_result is not None,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })

    return app


def _safe_list(df, col):
    if col not in df.columns:
        return []
    return [round(float(v), 3) if not np.isnan(v) else None for v in df[col].tolist()]
