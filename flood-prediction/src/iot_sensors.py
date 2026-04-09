"""
IoT Sensor Simulator
Simulates real-time data streams from IoT rainfall and water-level sensors.
In production, replace this with actual MQTT/HTTP sensor integrations.
"""

import random
import time
import math
from datetime import datetime, timedelta
from typing import Generator


class RainfallSensor:
    """Simulates an IoT rainfall sensor (tipping bucket rain gauge)."""

    def __init__(self, sensor_id: str, location: str, base_lat: float, base_lon: float):
        self.sensor_id = sensor_id
        self.location = location
        self.latitude = base_lat + random.uniform(-0.05, 0.05)
        self.longitude = base_lon + random.uniform(-0.05, 0.05)
        self._time_offset = random.uniform(0, 2 * math.pi)  # phase offset for simulation

    def read(self, timestamp: datetime = None) -> dict:
        """Read current rainfall intensity in mm/hr."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        t = timestamp.timestamp()
        # Simulate realistic rainfall pattern with sine wave + noise + spikes
        base = max(0, 5 * math.sin(t / 3600 + self._time_offset))
        noise = random.gauss(0, 1)
        spike = random.choices([0, random.uniform(20, 80)], weights=[0.97, 0.03])[0]
        rainfall_mm_hr = max(0, round(base + noise + spike, 2))

        # Simulate occasional sensor faults
        battery = round(random.uniform(3.2, 4.2), 2)
        signal_strength = random.randint(-90, -40)
        is_faulty = random.random() < 0.02  # 2% fault rate

        return {
            "sensor_id": self.sensor_id,
            "sensor_type": "rainfall",
            "location": self.location,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "timestamp": timestamp.isoformat() + "Z",
            "rainfall_mm_hr": None if is_faulty else rainfall_mm_hr,
            "battery_v": battery,
            "signal_dbm": signal_strength,
            "status": "FAULT" if is_faulty else "OK",
        }


class WaterLevelSensor:
    """Simulates an IoT ultrasonic water-level sensor installed in a river/reservoir."""

    def __init__(self, sensor_id: str, location: str, base_lat: float, base_lon: float,
                 danger_level_m: float = 4.5, critical_level_m: float = 6.0):
        self.sensor_id = sensor_id
        self.location = location
        self.latitude = base_lat + random.uniform(-0.05, 0.05)
        self.longitude = base_lon + random.uniform(-0.05, 0.05)
        self.danger_level_m = danger_level_m
        self.critical_level_m = critical_level_m
        self._current_level = random.uniform(1.0, 3.0)
        self._time_offset = random.uniform(0, 2 * math.pi)

    def read(self, timestamp: datetime = None) -> dict:
        """Read current water level in meters."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        t = timestamp.timestamp()
        # Simulate slowly changing water level with tidal/seasonal pattern
        trend = 1.5 * math.sin(t / 7200 + self._time_offset)
        noise = random.gauss(0, 0.05)
        self._current_level = max(0.1, min(8.0, self._current_level + trend * 0.01 + noise))
        level = round(self._current_level, 3)

        # Determine alert level
        if level >= self.critical_level_m:
            alert = "CRITICAL"
        elif level >= self.danger_level_m:
            alert = "DANGER"
        elif level >= self.danger_level_m * 0.75:
            alert = "WARNING"
        else:
            alert = "NORMAL"

        is_faulty = random.random() < 0.02

        return {
            "sensor_id": self.sensor_id,
            "sensor_type": "water_level",
            "location": self.location,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "timestamp": (timestamp if timestamp else datetime.utcnow()).isoformat() + "Z",
            "water_level_m": None if is_faulty else level,
            "danger_level_m": self.danger_level_m,
            "critical_level_m": self.critical_level_m,
            "alert": "FAULT" if is_faulty else alert,
            "battery_v": round(random.uniform(3.2, 4.2), 2),
            "signal_dbm": random.randint(-90, -40),
            "status": "FAULT" if is_faulty else "OK",
        }


class IoTSensorNetwork:
    """Manages a network of IoT sensors and provides batch readings."""

    def __init__(self):
        # Define sensor network (mimics real deployment in a river basin)
        self.rainfall_sensors = [
            RainfallSensor("RS-001", "Upstream Forest", 17.38, 78.47),
            RainfallSensor("RS-002", "Hilltop Station", 17.42, 78.52),
            RainfallSensor("RS-003", "Urban Catchment", 17.35, 78.44),
            RainfallSensor("RS-004", "Agricultural Zone", 17.40, 78.55),
        ]
        self.water_level_sensors = [
            WaterLevelSensor("WL-001", "River Gauge A", 17.36, 78.46, danger_level_m=4.5, critical_level_m=6.0),
            WaterLevelSensor("WL-002", "River Gauge B", 17.38, 78.49, danger_level_m=5.0, critical_level_m=7.0),
            WaterLevelSensor("WL-003", "Reservoir Outlet", 17.33, 78.43, danger_level_m=6.0, critical_level_m=8.0),
        ]

    def read_all(self, timestamp: datetime = None) -> dict:
        """Collect readings from all sensors simultaneously."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        return {
            "batch_timestamp": timestamp.isoformat() + "Z",
            "rainfall_readings": [s.read(timestamp) for s in self.rainfall_sensors],
            "water_level_readings": [s.read(timestamp) for s in self.water_level_sensors],
        }

    def generate_historical_data(self, hours: int = 48) -> list:
        """Generate historical sensor data for the past N hours (for demo/training)."""
        records = []
        now = datetime.utcnow()
        for i in range(hours * 6):  # Every 10 minutes
            ts = now - timedelta(minutes=i * 10)
            batch = self.read_all(ts)
            records.append(batch)
        return list(reversed(records))
