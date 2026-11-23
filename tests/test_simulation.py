import calendar
from datetime import date, datetime
import numpy as np
import pandas as pd
import pytest

from simulation import (
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


def test_year_offset():
    assert get_year_offset(2026, 2013) == 13
    assert get_year_offset(2025, 2013) == 12


def test_prompt_example_timestamp():
    # Original timestamp from requirement: '2013-09-13 18:30:00' -> '2026-09-13 18:30:00'
    orig = "2013-09-13 18:30:00"
    sim = to_simulated_timestamp(orig)
    assert sim == "2026-09-13 18:30:00"
    # Round-trip back to historical
    back = to_historical_timestamp(sim)
    assert back == orig


def test_timestamp_preserves_time_of_day():
    dt = datetime(2013, 7, 4, 15, 45, 30, 123456)
    sim = to_simulated_timestamp(dt)
    assert sim.year == 2026
    assert sim.month == 7
    assert sim.day == 4
    assert (sim.hour, sim.minute, sim.second, sim.microsecond) == (15, 45, 30, 123456)


def test_leap_year_handling():
    # 2012 was a leap year (Feb 29 exists)
    # 2012 + 13 = 2025 (not a leap year) -> should safely snap to Feb 28, 2025
    leap_date = date(2012, 2, 29)
    sim_date = to_simulated_date(leap_date)
    assert sim_date == date(2025, 2, 28)

    # String format
    assert to_simulated_timestamp("2012-02-29 12:00:00") == "2025-02-28 12:00:00"

    # Backward mapping: 2024 is a leap year (Feb 29 exists)
    # 2024 - 13 = 2011 (not a leap year) -> snaps to Feb 28, 2011
    leap_2024 = date(2024, 2, 29)
    hist_date = to_historical_date(leap_2024)
    assert hist_date == date(2011, 2, 28)


def test_date_and_month_mapping():
    assert to_simulated_date("2013-05-15") == date(2026, 5, 15)
    assert to_historical_date(date(2026, 5, 15)) == date(2013, 5, 15)

    assert to_simulated_month("2013-09") == "2026-09"
    assert to_historical_month("2026-09") == "2013-09"


def test_map_date_range():
    start_sim, end_sim = date(2026, 1, 1), date(2026, 12, 31)
    hist_start, hist_end = map_date_range_to_historical(start_sim, end_sim)
    assert hist_start == date(2013, 1, 1)
    assert hist_end == date(2013, 12, 31)

    sim_start_res, sim_end_res = map_date_range_to_simulated(hist_start, hist_end)
    assert sim_start_res == start_sim
    assert sim_end_res == end_sim


def test_dataframe_transformation_preserves_values_and_original():
    original_df = pd.DataFrame({
        "household_id": ["MAC000002", "MAC000003"],
        "timestamp": pd.to_datetime(["2013-09-13 18:00:00", "2013-09-13 18:30:00"]),
        "date": ["2013-09-13", "2013-09-13"],
        "energy_kwh": [0.456, 0.789],
        "power_kw": [0.912, 1.578],
        "day_of_week": ["Friday", "Friday"],
        "is_weekend": [False, False],
    })
    copy_orig = original_df.copy(deep=True)

    transformed = to_simulated_dataframe(original_df)

    # 1. Verify original DataFrame is NOT mutated
    pd.testing.assert_frame_equal(original_df, copy_orig)

    # 2. Verify consumption and power values are IDENTICAL
    np.testing.assert_array_equal(transformed["energy_kwh"], original_df["energy_kwh"])
    np.testing.assert_array_equal(transformed["power_kw"], original_df["power_kw"])
    assert list(transformed["household_id"]) == ["MAC000002", "MAC000003"]

    # 3. Verify timestamps shifted to 2026
    assert transformed["timestamp"].iloc[0] == pd.Timestamp("2026-09-13 18:00:00")
    assert transformed["timestamp"].iloc[1] == pd.Timestamp("2026-09-13 18:30:00")
    assert transformed["date"].iloc[0] == "2026-09-13"

    # 4. Verify day_of_week and is_weekend align with 2026-09-13 (which is a Sunday!)
    assert transformed["day_of_week"].iloc[0] == "Sunday"
    assert transformed["is_weekend"].iloc[0] == True


def test_series_transformation():
    s = pd.Series(pd.to_datetime(["2013-01-01 00:00:00", "2013-06-15 12:00:00"]))
    s_sim = to_simulated_series(s)
    assert s_sim.iloc[0] == pd.Timestamp("2026-01-01 00:00:00")
    assert s_sim.iloc[1] == pd.Timestamp("2026-06-15 12:00:00")


def test_forecast_date_transformation():
    # Target date from model forecast mapped using true +13 year simulation offset
    assert transform_forecast_date("2014-02-28") == "2027-02-28"
    assert transform_forecast_date("2013-02-28") == "2026-02-28"
    assert transform_forecast_date("2013-09-13") == "2026-09-13"
