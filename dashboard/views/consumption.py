import plotly.express as px
import streamlit as st

from common import get_filters, household_days

f = get_filters()
st.title("Energy Consumption Analysis")
st.caption(
    "Historical smart-meter consumption trends displayed on the 2026 simulated timeline. "
    "Daily, weekly, and monthly calculations preserve all actual meter readings."
)

days = household_days(f["where"], f["params"])
if days.empty:
    st.warning("No full days of data for this selection. Try adjusting the simulated date range.")
    st.stop()

# Weekday vs weekend (average kWh per household per day in simulated 2026 calendar)
is_weekend = days["date"].dt.dayofweek >= 5
a, b = st.columns(2)
a.metric("Weekday average", f"{days.loc[~is_weekend, 'energy_kwh'].mean():.1f} kWh per household per day")
b.metric("Weekend average", f"{days.loc[is_weekend, 'energy_kwh'].mean():.1f} kWh per household per day")

daily = days.groupby("date", as_index=False)["energy_kwh"].mean()
daily["cost"] = daily["energy_kwh"] * f["price"]
fig_cost = px.line(
    daily, x="date", y="cost",
    title="Simulated Daily Cost per Household (£)",
    labels={"date": "Simulated Date (2026)", "cost": "Cost (£)"},
)
st.plotly_chart(fig_cost, width="stretch")

order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
by_weekday = days.groupby("day_of_week", observed=False)["energy_kwh"].mean().reindex(order).reset_index()
fig_week = px.bar(
    by_weekday, x="day_of_week", y="energy_kwh",
    title="Average Daily Energy by Weekday (kWh per household)",
    labels={"day_of_week": "Day of Week", "energy_kwh": "Energy (kWh)"},
)

days["month"] = days["date"].dt.to_period("M").astype(str)
monthly = days.groupby("month", as_index=False)["energy_kwh"].mean()
fig_month = px.line(
    monthly, x="month", y="energy_kwh", markers=True,
    title="Average Daily Energy by Month – 2026 Simulation (kWh per household)",
    labels={"month": "Simulated Month (2026)", "energy_kwh": "Energy (kWh)"},
)

left, right = st.columns(2)
left.plotly_chart(fig_week, width="stretch")
right.plotly_chart(fig_month, width="stretch")
