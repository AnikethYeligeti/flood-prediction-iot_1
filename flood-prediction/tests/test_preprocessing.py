"""
Unit Tests for IoT Flood Prediction Preprocessing Pipeline
Run with: pytest tests/test_preprocessing.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.iot_sensors import IoTSensorNetwork, RainfallSensor, WaterLevelSensor
from src.preprocessing import (
    flatten_batch, batches_to_dataframe, handle_missing_values,
    detect_and_remove_outliers, engineer_features, normalize_features,
    create_sliding_windows, run_preprocessing_pipeline
)
from src.model import heuristic_flood_risk, score_to_alert, ALERT_THRESHOLDS


# ─────────────────────────────────────────────
# SENSOR TESTS
# ─────────────────────────────────────────────

class TestSensors:
    def test_rainfall_sensor_reads_value(self):
        s = RainfallSensor("RS-TEST", "Test", 17.38, 78.47)
        reading = s.read(datetime(2024, 6, 1, 12, 0, 0))
        assert reading["sensor_id"] == "RS-TEST"
        assert reading["sensor_type"] == "rainfall"
        assert reading["status"] in ("OK", "FAULT")
        if reading["status"] == "OK":
            assert reading["rainfall_mm_hr"] is not None
            assert reading["rainfall_mm_hr"] >= 0

    def test_water_level_sensor_alert_normal(self):
        s = WaterLevelSensor("WL-TEST", "Test", 17.38, 78.47, danger_level_m=4.5)
        # Force a low reading
        s._current_level = 1.0
        reading = s.read()
        if reading["status"] == "OK":
            assert reading["alert"] in ("NORMAL", "WARNING", "DANGER", "CRITICAL")

    def test_sensor_network_returns_all_sensors(self):
        net = IoTSensorNetwork()
        batch = net.read_all()
        assert "rainfall_readings" in batch
        assert "water_level_readings" in batch
        assert len(batch["rainfall_readings"]) == 4
        assert len(batch["water_level_readings"]) == 3

    def test_historical_data_length(self):
        net = IoTSensorNetwork()
        data = net.generate_historical_data(hours=2)
        # 2h × 6 readings per hour = 12 batches
        assert len(data) == 12


# ─────────────────────────────────────────────
# PREPROCESSING TESTS
# ─────────────────────────────────────────────

class TestPreprocessing:
    @pytest.fixture
    def sample_batches(self):
        net = IoTSensorNetwork()
        return net.generate_historical_data(hours=10)

    def test_flatten_batch_keys(self, sample_batches):
        row = flatten_batch(sample_batches[0])
        for key in ["rain_mean_mm_hr", "wl_mean_m", "fault_count", "timestamp"]:
            assert key in row

    def test_batches_to_dataframe_shape(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        assert len(df) == len(sample_batches)
        assert "timestamp" in df.columns

    def test_missing_value_handling_no_nan(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        # Inject some NaNs
        df.loc[df.index[:5], "rain_mean_mm_hr"] = np.nan
        cleaned = handle_missing_values(df)
        assert cleaned["rain_mean_mm_hr"].isna().sum() == 0

    def test_outlier_removal_reduces_rows(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        df = handle_missing_values(df)
        # Inject obvious outliers
        df.loc[df.index[10], "rain_mean_mm_hr"] = 9999.0
        cleaned, outliers = detect_and_remove_outliers(df)
        assert len(cleaned) <= len(df)
        assert len(outliers) >= 1

    def test_feature_engineering_adds_columns(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        df = handle_missing_values(df)
        featured = engineer_features(df)
        for col in ["rain_accum_6h_mm", "wl_velocity_m_per_10min", "flood_risk_score", "hour_sin"]:
            assert col in featured.columns, f"Missing: {col}"

    def test_flood_risk_score_range(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        df = handle_missing_values(df)
        featured = engineer_features(df)
        assert featured["flood_risk_score"].between(0, 10).all()

    def test_normalization_range(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        df = handle_missing_values(df)
        df, _ = detect_and_remove_outliers(df)
        featured = engineer_features(df)
        normalized, scaler = normalize_features(featured)
        for col in ["rain_mean_mm_hr", "wl_mean_m"]:
            if col in normalized.columns:
                assert normalized[col].between(-0.01, 1.01).all(), f"{col} out of [0,1]"

    def test_sliding_windows_shape(self, sample_batches):
        df = batches_to_dataframe(sample_batches)
        df = handle_missing_values(df)
        df, _ = detect_and_remove_outliers(df)
        featured = engineer_features(df)
        normalized, _ = normalize_features(featured)
        window_size = 6
        X, y = create_sliding_windows(normalized, window_size=window_size)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == window_size

    def test_full_pipeline_runs(self, sample_batches):
        result = run_preprocessing_pipeline(sample_batches, window_size=6)
        assert "stats" in result
        assert result["stats"]["raw_rows"] == len(sample_batches)
        assert result["X"] is not None


# ─────────────────────────────────────────────
# MODEL TESTS
# ─────────────────────────────────────────────

class TestModel:
    def test_heuristic_normal_conditions(self):
        result = heuristic_flood_risk(rain_mm_hr=1.0, water_level_m=1.0, rain_accum_6h_mm=5.0)
        assert result["flood_risk_score"] < 5.0
        assert result["alert_level"] in ("NORMAL", "WATCH")

    def test_heuristic_extreme_conditions(self):
        result = heuristic_flood_risk(rain_mm_hr=100.0, water_level_m=8.0, rain_accum_6h_mm=500.0)
        assert result["flood_risk_score"] >= 7.0

    def test_score_to_alert_thresholds(self):
        assert score_to_alert(0.0) == "NORMAL"
        assert score_to_alert(4.0) == "WATCH"
        assert score_to_alert(6.0) == "WARNING"
        assert score_to_alert(7.5) == "DANGER"
        assert score_to_alert(9.0) == "CRITICAL"

    def test_heuristic_score_bounds(self):
        for _ in range(20):
            rain = np.random.uniform(0, 200)
            wl   = np.random.uniform(0, 10)
            result = heuristic_flood_risk(rain_mm_hr=rain, water_level_m=wl)
            assert 0 <= result["flood_risk_score"] <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
