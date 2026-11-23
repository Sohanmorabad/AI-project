import os
from datetime import datetime

import pandas as pd
import streamlit as st

from common import get_filters, to_simulated_date, transform_forecast_date

f = get_filters()
household, price = f["household"], f["price"]
st.title("Forecasted Energy Consumption – 2026 Simulation")
st.caption(
    "Day-ahead machine learning forecasts presented on the 2026 simulated timeline. "
    "Trained Random Forest and regression models, feature weights, and numerical forecasts are strictly preserved."
)


@st.cache_data
def load_latest():
    return pd.read_csv("models/forecast_latest.csv")


@st.cache_data
def load_walkforward():
    return pd.read_csv("models/forecast_walkforward.csv")


latest = load_latest()
target_date_raw = latest["target_date"].iloc[0]
target_date = transform_forecast_date(target_date_raw)

if household != "All households":
    latest = latest[latest["household_id"] == household]

if latest.empty:
    st.info("No forecast available for this household (not enough recent history).")
else:
    forecast_kwh = latest["forecast_kwh"].mean()
    a, b = st.columns(2)
    a.metric(f"Forecast for {target_date} (per household)", f"{forecast_kwh:.1f} kWh")
    b.metric("Estimated cost", f"£{forecast_kwh * price:.2f}")
    st.caption(
        f"The forecast target date ({target_date}) is for the day immediately following the final simulated evaluation period "
        "(December 2026 – February 2027). Original model predictions are preserved. Weather inputs use observed weather for that day."
    )

st.subheader("Model Accuracy Across 2026 Simulated Evaluation Windows")
wf = load_walkforward()
base = wf.loc[wf["model"] == "Baseline: same as yesterday", "mean MAE"].iloc[0]
models_only = wf[wf["features"] != "-"]
best = models_only.loc[models_only["mean MAE"].idxmin()]


def _map_period_col(col: str) -> str:
    try:
        dt = datetime.strptime(col, "%b %Y")
        sim_d = to_simulated_date(dt.date())
        return f"{sim_d.strftime('%b %Y')} (Simulated)"
    except Exception:
        return col


wf_display = wf.rename(columns={c: _map_period_col(c) for c in wf.columns})

st.write(
    "Tested with walk-forward validation: trained on prior history and evaluated across five distinct 2-month periods "
    "mapped to the 2026 simulation calendar. "
    f"The best model ({best['model']}, features: {best['features']}) achieves an average error of "
    f"{best['mean MAE']:.2f} kWh per household per day, "
    f"{100 * (1 - best['mean MAE'] / base):.0f}% better than repeating yesterday's usage ({base:.2f} kWh)."
)
st.dataframe(wf_display, width="stretch", hide_index=True)
st.caption("Mean absolute error in kWh per household per day for each simulated test window. Lower is better.")

sim_plot = "outputs/actual_vs_predicted_2026_simulation.png"
plot_img = sim_plot if os.path.exists(sim_plot) else "outputs/actual_vs_predicted_v2.png"
st.image(
    plot_img,
    caption="Actual vs predicted daily energy across final evaluation window (Dec 2026 – Feb 2027, averaged across households – 2026 Simulation)",
)
