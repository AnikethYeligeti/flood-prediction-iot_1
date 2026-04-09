# 🌊 FloodSense IoT — Rainfall & Water-Level Data Preprocessing for Flood Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A full-stack IoT data preprocessing and flood prediction system built with Python, Flask, and scikit-learn. It simulates a real-world network of IoT rainfall gauges and water-level sensors, runs a complete machine-learning preprocessing pipeline, trains a Random Forest model, and serves a real-time dashboard.

---

## 📸 Dashboard Preview

> Live sensor readings → preprocessing pipeline → ML prediction → visual risk meter

![Dashboard](static/preview.png)

---

## 🏗️ System Architecture

```
IoT Sensor Network
│
├── RainfallSensor (RS-001 … RS-004)   → tipping-bucket rain gauge simulation
└── WaterLevelSensor (WL-001 … WL-003) → ultrasonic sensor simulation
        │
        ▼
Data Preprocessing Pipeline
│
├── 1. IoT Ingest       → flatten multi-sensor batch → pandas DataFrame
├── 2. Missing Values   → linear interpolation → ffill → median imputation
├── 3. Outlier Removal  → modified Z-score (robust IQR)
├── 4. Feature Eng.     → rolling accumulations, velocity, cyclical time, risk score
├── 5. Normalization    → Min-Max scaling (0–1)
└── 6. Sliding Window   → (samples, 18 timesteps, N features) for time-series ML
        │
        ▼
ML Model (Random Forest Regressor)
│
└── Predicts flood_risk_score (0–10)
        │
        ▼
Flask REST API  →  Interactive Dashboard (Chart.js)
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🛰️ IoT Simulation | 7 sensors (4 rainfall + 3 water-level) with realistic noise, spikes, and 2% fault rate |
| 🩹 Missing Value Imputation | Linear interpolation → forward-fill → median fallback |
| 🚨 Outlier Detection | Modified Z-score (robust to skewed distributions) |
| ⚙️ Feature Engineering | 19+ features: rolling accumulations (1h–24h), water-level velocity, cyclical time encoding, IMD intensity class |
| 📊 Normalization | Min-Max scaling with scaler persistence |
| 🪟 Sliding Window | 18-step (3h) look-back window for time-series learning |
| 🤖 ML Model | Random Forest Regressor (100 trees, time-ordered train/test split) |
| 🔮 Heuristic Fallback | Domain-formula prediction when ML model is not yet trained |
| 📡 REST API | `/api/sensors/live`, `/api/preprocess`, `/api/train`, `/api/predict`, `/api/outliers` |
| 📈 Dashboard | Real-time Chart.js time-series, animated risk meter, sensor tile grid |
| ✅ Unit Tests | 18 pytest tests covering sensors, preprocessing, and model |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10 or higher
- pip

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/flood-prediction-iot.git
cd flood-prediction-iot
```

### 2. Create a virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask server
```bash
python app.py
```

### 5. Open the dashboard
Navigate to **http://localhost:5000** in your browser.

---

## 🖥️ Using the Dashboard

Follow these steps in order:

1. **Click "Live Sensors"** — Loads the current IoT sensor network readings (7 sensors).
2. **Click "▶ Run Preprocessing (48h)"** — Generates 48 hours of historical sensor data and runs the full 6-step preprocessing pipeline. Charts update automatically.
3. **Click "🤖 Train ML Model"** — Trains a Random Forest Regressor on the windowed data. Model metrics (R², RMSE, MAE) appear in the panel below.
4. **Click "🔮 Predict Risk"** — Runs flood risk prediction and updates the animated risk meter and alert badge.

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Dashboard HTML |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/sensors/live` | Live batch reading from all 7 sensors |
| `GET` | `/api/preprocess?hours=48` | Run full preprocessing pipeline |
| `POST`| `/api/train` | Train ML model on preprocessed data |
| `GET` | `/api/predict` | Predict current flood risk score |
| `GET` | `/api/outliers` | List detected outlier records |

### Example: Live Sensor Reading
```bash
curl http://localhost:5000/api/sensors/live
```
```json
{
  "batch_timestamp": "2024-06-15T08:30:00Z",
  "rainfall_readings": [
    {
      "sensor_id": "RS-001",
      "sensor_type": "rainfall",
      "location": "Upstream Forest",
      "rainfall_mm_hr": 12.4,
      "battery_v": 3.9,
      "signal_dbm": -65,
      "status": "OK"
    }
  ],
  "water_level_readings": [...]
}
```

### Example: Flood Risk Prediction
```bash
curl http://localhost:5000/api/predict
```
```json
{
  "flood_risk_score": 6.72,
  "alert_level": "WARNING",
  "color": "#f59e0b",
  "confidence_lower": 5.2,
  "confidence_upper": 8.1,
  "method": "ml_model"
}
```

---

## 📁 Project Structure

```
flood-prediction-iot/
│
├── app.py                    # Flask application entry point
├── requirements.txt          # Python dependencies
├── README.md
├── LICENSE
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── iot_sensors.py        # IoT sensor simulation (rainfall + water-level)
│   ├── preprocessing.py      # Full preprocessing pipeline
│   ├── model.py              # ML model (Random Forest + heuristic)
│   └── api.py                # Flask REST API routes
│
├── templates/
│   └── index.html            # Dashboard (Chart.js + vanilla JS)
│
├── static/                   # Static assets (CSS, images)
│
├── models/                   # Saved ML model artifacts (.joblib)
│
├── tests/
│   └── test_preprocessing.py # 18 pytest unit tests
│
└── notebooks/                # (Optional) Jupyter EDA notebooks
```

---

## 🧪 Running Tests

```bash
pytest tests/test_preprocessing.py -v
```

Expected output:
```
tests/test_preprocessing.py::TestSensors::test_rainfall_sensor_reads_value PASSED
tests/test_preprocessing.py::TestSensors::test_water_level_sensor_alert_normal PASSED
...
18 passed in 3.2s
```

---

## 🌧️ Rainfall Intensity Classes (IMD Scale)

| Class | Range (mm/hr) | Description |
|-------|--------------|-------------|
| 0 | 0 | No Rain |
| 1 | 0–2.5 | Light Rain |
| 2 | 2.5–7.5 | Moderate Rain |
| 3 | 7.5–35.5 | Rather Heavy Rain |
| 4 | 35.5–64.5 | Heavy Rain |
| 5 | 64.5–115.5 | Very Heavy Rain |
| 6 | >115.5 | Extremely Heavy Rain |

---

## 🚨 Flood Risk Alert Levels

| Level | Score | Action |
|-------|-------|--------|
| 🟢 NORMAL | 0–3 | No action required |
| 🟡 WATCH | 3–5 | Monitor sensors closely |
| 🟠 WARNING | 5–7 | Prepare response teams |
| 🔴 DANGER | 7–8.5 | Evacuate low-lying areas |
| 🆘 CRITICAL | 8.5–10 | Emergency response immediate |

---

## 🔧 Configuration

Edit `src/iot_sensors.py` to:
- Add/remove sensors from the network
- Change geographic coordinates
- Adjust fault rates and noise parameters
- Replace simulation with real MQTT/HTTP sensor calls

---

## 🔮 Future Enhancements

- [ ] MQTT broker integration (real IoT sensors via Eclipse Mosquitto)
- [ ] LSTM / Transformer model for better time-series prediction
- [ ] PostgreSQL/TimescaleDB for sensor data persistence
- [ ] WebSocket live streaming to dashboard
- [ ] Docker containerization
- [ ] Telegram/SMS alert notifications

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- India Meteorological Department (IMD) rainfall intensity classification
- scikit-learn, Flask, Chart.js, pandas communities
