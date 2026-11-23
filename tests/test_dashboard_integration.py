import hashlib
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from simulation import (
    map_date_range_to_historical,
    to_historical_month,
    to_simulated_dataframe,
    to_simulated_date,
    to_simulated_month,
    to_simulated_timestamp,
    transform_forecast_date,
)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "energy.db"

ORIGINAL_DB_SHA256 = "24bbc28e0c931509c8055e8da57d8fb183badfc252085897a09feb1a1f367ce5"
ORIGINAL_MODELS_SHA256 = {
    "anomaly_validation.csv": "475b188f8a9838e087e1acd9ca437e734fc1d693c610acba07c02e241aced1be",
    "forecast_latest.csv": "988b57ec83f1a8a21cf7c2d19a2c7023f3da5e14e6c446b5af419aed0d15d83c",
    "forecast_walkforward.csv": "1ec4e668dfe0aca55ae130f609e405954cd3e6ffa8cd465e91db4f0bb33420c2",
    "household_profiles.csv": "e83b210f1f32707a5bf252b8bad6472f1e01ce6ff266268a145377da8c565951",
}


def test_database_hash_is_preserved():
    """Verify that data/energy.db was not modified or overwritten."""
    with open(DB_PATH, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()
    assert current_hash == ORIGINAL_DB_SHA256, "energy.db has been modified!"


def test_model_files_are_preserved():
    """Verify that all existing trained model files and weights were preserved."""
    models_dir = ROOT / "models"
    for filename, expected_hash in ORIGINAL_MODELS_SHA256.items():
        file_path = models_dir / filename
        assert file_path.exists(), f"Missing model file: {filename}"
        with open(file_path, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"Model file {filename} was modified!"


def test_simulated_date_filter_mapping_to_sql():
    """Test that selecting 2026 dates maps to historical dates and queries database successfully."""
    sim_start, sim_end = date(2026, 9, 1), date(2026, 9, 7)
    hist_start, hist_end = map_date_range_to_historical(sim_start, sim_end)

    assert hist_start == date(2013, 9, 1)
    assert hist_end == date(2013, 9, 7)

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT COUNT(*) FROM energy_readings WHERE date >= ? AND date <= ?",
            [hist_start.isoformat(), hist_end.isoformat()],
        ).fetchone()[0]
        assert rows > 0
    finally:
        con.close()


def test_household_filtering_with_simulated_dates():
    """Test filtering by specific household within simulated dates."""
    household = "MAC000002"
    sim_start, sim_end = date(2026, 5, 1), date(2026, 5, 3)
    hist_start, hist_end = map_date_range_to_historical(sim_start, sim_end)

    con = sqlite3.connect(DB_PATH)
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT COUNT(*) FROM energy_readings WHERE date >= ? AND date <= ? AND household_id = ?",
            [hist_start.isoformat(), hist_end.isoformat(), household],
        ).fetchone()[0]
        # 3 full days of 48 half-hourly readings = 144 readings
        assert rows == 144
    finally:
        con.close()


def test_household_days_transformation():
    """Test that household_days query results transform correctly to 2026 without altering values."""
    con = sqlite3.connect(DB_PATH)
    try:
        raw_df = pd.read_sql_query(
            """SELECT household_id, date, day_of_week, SUM(energy_kwh) AS energy_kwh
               FROM energy_readings WHERE date >= '2013-09-13' AND date <= '2013-09-15' AND household_id = 'MAC000002'
               GROUP BY household_id, date HAVING COUNT(*) = 48""",
            con,
        )
    finally:
        con.close()

    assert len(raw_df) == 3
    orig_energy = raw_df["energy_kwh"].copy()

    transformed = to_simulated_dataframe(raw_df, date_cols=["date"])

    # 1. Energy consumption values must be identical
    pd.testing.assert_series_equal(transformed["energy_kwh"], orig_energy)

    # 2. Dates must be shifted to 2026
    assert list(transformed["date"]) == ["2026-09-13", "2026-09-14", "2026-09-15"]

    # 3. 2026-09-13 is Sunday, 2026-09-14 is Monday, 2026-09-15 is Tuesday
    assert list(transformed["day_of_week"]) == ["Sunday", "Monday", "Tuesday"]


def test_anomaly_files_simulation_in_memory():
    """Verify that loading anomalies and transforming them shifts dates to 2026 while preserving scores."""
    anoms_path = ROOT / "data" / "processed" / "anomalies_only.csv"
    if anoms_path.exists():
        df = pd.read_csv(anoms_path, nrows=50)
        orig_energy = df["energy_kwh"].copy()
        orig_limits = df["upper_limit"].copy()

        sim_df = to_simulated_dataframe(df, timestamp_cols=["timestamp", "date"])

        # Preserved values
        pd.testing.assert_series_equal(sim_df["energy_kwh"], orig_energy)
        pd.testing.assert_series_equal(sim_df["upper_limit"], orig_limits)

        # Shifted timestamps
        first_orig_year = pd.to_datetime(df["timestamp"].iloc[0]).year
        first_sim_year = pd.to_datetime(sim_df["timestamp"].iloc[0]).year
        assert first_sim_year == first_orig_year + 13


def test_monthly_report_mapping():
    """Verify month mapping between simulated 2026 months and historical months."""
    sim_m = "2026-08"
    hist_m = to_historical_month(sim_m)
    assert hist_m == "2013-08"
    assert to_simulated_month(hist_m) == sim_m


def test_forecast_latest_target_date_simulation_regression():
    """Regression test: verify that models/forecast_latest.csv target date 2014-02-28 maps to 2027-02-28."""
    df = pd.read_csv(ROOT / "models" / "forecast_latest.csv")
    raw_target = df["target_date"].iloc[0]
    assert raw_target == "2014-02-28"
    sim_target = transform_forecast_date(raw_target)
    assert sim_target == "2027-02-28"


def test_live_monitoring_reading_time_metric_uses_2026_simulation():
    """Verify that historical live readings are mapped to simulated 2026 timestamps

    for the Reading time (2026) metric and chart timeline, preserving original readings.
    """
    historical_ts = "2013-09-17 23:30:00"
    simulated_ts = to_simulated_timestamp(historical_ts)
    assert simulated_ts == "2026-09-17 23:30:00"

    sim_dt = pd.to_datetime(simulated_ts)
    formatted = sim_dt.strftime("%d %b %Y %H:%M")
    assert formatted == "17 Sep 2026 23:30"

    # Test with live_readings table format from data/live.db
    live_df = pd.DataFrame({
        "source": ["replay", "replay"],
        "device_id": ["MAC000002", "MAC000002"],
        "timestamp": pd.to_datetime(["2013-09-17 23:00:00", "2013-09-17 23:30:00"]),
        "power_kw": [0.428, 0.526],
        "received_at": ["2026-10-03 21:05:10", "2026-10-03 21:05:11"],
    })
    sim_df = to_simulated_dataframe(live_df, timestamp_cols=["timestamp"])

    # 1. Numerical power readings and device metadata are preserved
    pd.testing.assert_series_equal(sim_df["power_kw"], live_df["power_kw"])
    pd.testing.assert_series_equal(sim_df["device_id"], live_df["device_id"])
    pd.testing.assert_series_equal(sim_df["source"], live_df["source"])

    # 2. Latest reading timestamp mapped to 2026
    latest_reading = sim_df.iloc[-1]
    assert latest_reading["timestamp"].year == 2026
    assert latest_reading["timestamp"].strftime("%d %b %Y %H:%M") == "17 Sep 2026 23:30"

    # 3. Reading time metric formatting
    metric_time = (
        to_simulated_timestamp(latest_reading["timestamp"])
        if latest_reading["timestamp"].year < 2020
        else latest_reading["timestamp"]
    )
    assert metric_time.strftime("%d %b %Y %H:%M") == "17 Sep 2026 23:30"
