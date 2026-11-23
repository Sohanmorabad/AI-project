import os
import shutil
import sqlite3
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from simulation import (  # noqa: E402
    DEFAULT_BASE_YEAR,
    DEFAULT_SIMULATION_YEAR,
    get_year_offset,
    map_date_range_to_historical,
    map_date_range_to_simulated,
    to_historical_date,
    to_historical_month,
    to_historical_timestamp,
    to_simulated_dataframe,
    to_simulated_date,
    to_simulated_month,
    to_simulated_series,
    to_simulated_timestamp,
    transform_forecast_date,
)

DB_PATH = "data/energy.db"
DB_URL = "https://github.com/millyc17-lgtm/smart-energy-monitoring-system/releases/download/data-v1/energy_db.zip"


def ensure_database():
    """On a fresh machine (for example the hosted app), download the database once."""
    if os.path.exists(DB_PATH):
        return
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    zip_path = DB_PATH + ".zip"
    try:
        with st.spinner("First start: downloading the dataset. This can take a minute..."):
            urllib.request.urlretrieve(DB_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z, z.open("energy.db") as src, open(DB_PATH + ".tmp", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(DB_PATH + ".tmp", DB_PATH)
            os.remove(zip_path)
    except Exception as e:
        st.error(f"Could not download the dataset: {e}")
        st.stop()


@st.cache_data
def run_query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


def sidebar_filters(simulation_year: int = DEFAULT_SIMULATION_YEAR):
    """Draw the sidebar controls once and store the choices for every page to use."""
    ensure_database()
    bounds = run_query("SELECT MIN(date) AS first_day, MAX(date) AS last_day FROM energy_readings")
    first_hist_day = pd.Timestamp(bounds["first_day"][0]).date()
    last_hist_day = pd.Timestamp(bounds["last_day"][0]).date()

    sim_min_day = to_simulated_date(first_hist_day, simulation_year=simulation_year)
    sim_max_day = to_simulated_date(last_hist_day, simulation_year=simulation_year)

    household_ids = run_query(
        "SELECT DISTINCT household_id FROM energy_readings ORDER BY household_id"
    )["household_id"].tolist()

    st.sidebar.header("⚡ Energy Filters – 2026 Simulation")
    st.sidebar.info(
        "💡 **2026 Simulation Active**\n\n"
        "Historical smart-meter readings are mapped to a 2026 simulated calendar. "
        "Real consumption values and ML models are preserved."
    )
    household = st.sidebar.selectbox("Household", ["All households"] + household_ids)

    # Default to the complete 2026 simulation calendar year
    default_start = max(date(simulation_year, 1, 1), sim_min_day)
    default_end = min(date(simulation_year, 12, 31), sim_max_day)

    dates = st.sidebar.date_input(
        "Simulated Reporting Period",
        value=(default_start, default_end),
        min_value=sim_min_day,
        max_value=sim_max_day,
        help="Select dates within the simulated period.",
    )
    if len(dates) != 2:
        st.info("Select a start date and an end date.")
        st.stop()
    start, end = dates

    st.sidebar.header("Assumptions")
    price = st.sidebar.number_input("Electricity price (£ per kWh)", min_value=0.0, value=0.25, step=0.01)
    factor = st.sidebar.number_input(
        "Emissions factor (kg CO₂e per kWh)",
        min_value=0.0, value=0.13096, step=0.001, format="%.5f",
        help="UK grid electricity, DESNZ/DEFRA 2026 conversion factors. Change it to test other assumptions.",
    )

    # Convert the selected simulated 2026 dates to their corresponding historical dates before SQL query execution
    hist_start, hist_end = map_date_range_to_historical(start, end, simulation_year=simulation_year)

    # Filters are passed to SQL as ? placeholders, never pasted into the query text
    where = "date >= ? AND date <= ?"
    params = [hist_start.isoformat(), hist_end.isoformat()]
    if household != "All households":
        where += " AND household_id = ?"
        params.append(household)

    st.session_state["filters"] = {
        "household": household,
        "start": start,
        "end": end,
        "hist_start": hist_start,
        "hist_end": hist_end,
        "price": price,
        "factor": factor,
        "where": where,
        "params": tuple(params),
        "simulation_year": simulation_year,
    }


def get_filters():
    return st.session_state["filters"]


def household_days(where, params):
    """Total energy per household per full day (48 readings), aggregated in SQL and mapped to 2026 simulation."""
    d = run_query(
        f"""SELECT household_id, date, day_of_week, SUM(energy_kwh) AS energy_kwh
            FROM energy_readings WHERE {where}
            GROUP BY household_id, date
            HAVING COUNT(*) = 48""",
        params,
    )
    if d.empty:
        d["date"] = pd.to_datetime(d.get("date", []))
        return d
    # Map dates to simulated timeline without mutating original DataFrame
    d = to_simulated_dataframe(d, date_cols=["date"])
    d["date"] = pd.to_datetime(d["date"])
    return d
