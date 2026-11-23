import pandas as pd
import streamlit as st

from common import get_filters, run_query, to_historical_month, to_simulated_month

f = get_filters()
household, price, factor = f["household"], f["price"], f["factor"]
st.title("Monthly Energy Report – 2026 Simulation")
st.caption(
    "Monthly energy consumption and cost summary mapped to the 2026 simulation timeline. "
    "All calculations and aggregation logic preserve original meter measurements."
)

raw_months = run_query(
    "SELECT DISTINCT substr(date, 1, 7) AS month FROM energy_readings WHERE date >= '2012-10-01' ORDER BY month"
)["month"].tolist()

# Map historical months to simulated 2026 months
month_map = {to_simulated_month(m): m for m in raw_months}
all_sim_months = sorted(list(month_map.keys()))

# Default to 2026 simulation calendar months
sim_months_2026 = [m for m in all_sim_months if m.startswith("2026")]
available_months = sim_months_2026 if sim_months_2026 else all_sim_months

month = st.selectbox("Simulated Month", available_months, index=max(len(available_months) - 4, 0))


def month_stats(sim_m):
    hist_m = month_map.get(sim_m, to_historical_month(sim_m))
    sql = """
        SELECT COUNT(*) AS household_days, SUM(day_kwh) AS total_kwh,
               AVG(day_kwh) AS avg_day_kwh, MAX(peak_kw) AS peak_kw
        FROM (
            SELECT household_id, date, SUM(energy_kwh) AS day_kwh, MAX(power_kw) AS peak_kw
            FROM energy_readings
            WHERE substr(date, 1, 7) = ?"""
    p = [hist_m]
    if household != "All households":
        sql += " AND household_id = ?"
        p.append(household)
    sql += """
            GROUP BY household_id, date
            HAVING COUNT(*) = 48
        )"""
    return run_query(sql, tuple(p)).iloc[0]


cur = month_stats(month)
prev_month = (pd.Period(month) - 1).strftime("%Y-%m")
prev = month_stats(prev_month) if prev_month in all_sim_months else None

if cur["household_days"] == 0:
    st.info("No full days of data for this simulated month and household.")
else:
    total_kwh = cur["total_kwh"]
    change = None
    if prev is not None and prev["household_days"] > 0:
        change = (cur["avg_day_kwh"] / prev["avg_day_kwh"] - 1) * 100

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total energy", f"{total_kwh:,.0f} kWh")
    r2.metric("Estimated cost", f"£{total_kwh * price:,.0f}")
    r3.metric("Estimated emissions", f"{total_kwh * factor:,.0f} kg CO₂e")
    r4.metric(
        "Average per household per day",
        f"{cur['avg_day_kwh']:.1f} kWh",
        delta=None if change is None else f"{change:+.1f}% vs previous month",
        delta_color="inverse",
    )

    report = pd.DataFrame({
        "Metric": [
            "Simulated Reporting Period", "Household", "Total energy (kWh)", "Estimated cost (GBP)",
            "Estimated emissions (kg CO2e)", "Average per household per day (kWh)",
            "Peak power (kW)", "Change vs previous month (%)",
        ],
        "Value": [
            f"{month} (2026 Simulation)", household, f"{total_kwh:.1f}", f"{total_kwh * price:.2f}",
            f"{total_kwh * factor:.1f}", f"{cur['avg_day_kwh']:.2f}", f"{cur['peak_kw']:.2f}",
            "n/a" if change is None else f"{change:.1f}",
        ],
    })
    st.dataframe(report, width="stretch", hide_index=True)
    st.download_button(
        "Download simulated report (CSV)",
        report.to_csv(index=False),
        file_name=f"simulated_energy_report_{month}.csv",
        mime="text/csv",
    )
    st.caption(
        "Full days only (48 half-hour readings). Uses the price and emissions factor from the sidebar. "
        "The comparison uses per-household averages because the number of reporting households varies."
    )
