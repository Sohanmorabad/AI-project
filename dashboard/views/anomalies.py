import pandas as pd
import streamlit as st

from common import get_filters, to_simulated_dataframe

f = get_filters()
household, start, end = f["household"], f["start"], f["end"]
st.title("Energy Anomaly Analysis – 2026 Simulation")
st.caption(
    "Flags indicate energy consumption that is unusually high for a given household at that time of day. "
    "All anomaly occurrences are presented on the 2026 simulated timeline. "
    "Statistical rules, Isolation Forest ML models, and anomaly scores are strictly preserved."
)


def in_selection(d):
    d = d[(d["timestamp"] >= pd.Timestamp(start)) & (d["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))]
    if household != "All households":
        d = d[d["household_id"] == household]
    return d


@st.cache_data
def load_anomalies():
    df = pd.read_csv("data/processed/anomalies_only.csv", parse_dates=["timestamp"])
    return to_simulated_dataframe(df, timestamp_cols=["timestamp", "date"])


@st.cache_data
def load_ml_anomalies():
    df = pd.read_csv("data/processed/ml_anomalies_only.csv", parse_dates=["timestamp"])
    return to_simulated_dataframe(df, timestamp_cols=["timestamp", "date"])


st.subheader("Rule-Based Anomaly Detector – 2026 Simulation")
anoms = in_selection(load_anomalies())
st.write(f"{len(anoms):,} unusual readings detected in the selected simulated period")
top = anoms.nlargest(10, "energy_kwh")[
    ["household_id", "timestamp", "energy_kwh", "normal_mean", "upper_limit"]
].round({"energy_kwh": 2, "normal_mean": 2, "upper_limit": 2})
st.dataframe(top, width="stretch", hide_index=True)

st.subheader("Isolation Forest – Machine Learning Detector – 2026 Simulation")
ml = in_selection(load_ml_anomalies())
st.write(
    f"{len(ml):,} readings flagged by the trained Isolation Forest model, "
    f"{int(ml['is_anomaly'].sum()):,} of which were also flagged by the rule-based detector"
)
top_ml = ml.nsmallest(10, "ml_score")[
    ["household_id", "timestamp", "energy_kwh", "normal_mean", "ml_score", "is_anomaly"]
].rename(columns={"is_anomaly": "also_flagged_by_rule"})
top_ml = top_ml.round({"energy_kwh": 2, "normal_mean": 2, "ml_score": 3})
st.dataframe(top_ml, width="stretch", hide_index=True)
