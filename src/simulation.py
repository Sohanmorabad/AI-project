"""Centralized 2026 Simulation Module.

Provides bidirectional mapping between historical smart-meter readings (2011-2014)
and the simulated 2026 calendar timeline, while preserving all actual consumption
readings, trained machine learning models, and anomaly scores.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime
from typing import Any, Sequence, Tuple, Union

import pandas as pd

DEFAULT_SIMULATION_YEAR: int = 2026
DEFAULT_BASE_YEAR: int = 2013


def get_year_offset(simulation_year: int = DEFAULT_SIMULATION_YEAR, base_year: int = DEFAULT_BASE_YEAR) -> int:
    """Return the year offset from the historical base year to the simulation year.

    Default: 2026 - 2013 = 13 years.
    """
    return simulation_year - base_year


def _shift_date_parts(year: int, month: int, day: int, year_offset: int) -> Tuple[int, int, int]:
    """Shift year by year_offset, safely snapping Feb 29 to Feb 28 in non-leap years."""
    target_year = year + year_offset
    if month == 2 and day == 29 and not calendar.isleap(target_year):
        target_day = 28
    else:
        target_day = day
    return target_year, month, target_day


def to_simulated_timestamp(
    ts: Any,
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> Any:
    """Transform a historical timestamp/date into its simulated counterpart.

    Preserves time-of-day information, leap-year safety, and original data types.
    Example: '2013-09-13 18:30:00' -> '2026-09-13 18:30:00'.
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return ts
    if isinstance(ts, pd.Timestamp) and pd.isna(ts):
        return pd.NaT

    offset = get_year_offset(simulation_year, base_year)

    if isinstance(ts, str):
        ts_clean = ts.strip()
        # Month string: 'YYYY-MM'
        if len(ts_clean) == 7 and ts_clean[4] == "-":
            y = int(ts_clean[:4])
            m = ts_clean[5:7]
            return f"{y + offset:04d}-{m}"
        # Date string: 'YYYY-MM-DD'
        if len(ts_clean) == 10 and ts_clean[4] == "-" and ts_clean[7] == "-":
            y = int(ts_clean[:4])
            m = int(ts_clean[5:7])
            d = int(ts_clean[8:10])
            ty, tm, td = _shift_date_parts(y, m, d, offset)
            return f"{ty:04d}-{tm:02d}-{td:02d}"
        # Full timestamp string
        dt = pd.to_datetime(ts_clean)
        ty, tm, td = _shift_date_parts(dt.year, dt.month, dt.day, offset)
        new_dt = dt.replace(year=ty, month=tm, day=td)
        # Preserve original string formatting pattern
        if "T" in ts_clean:
            return new_dt.strftime("%Y-%m-%dT%H:%M:%S")
        return new_dt.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(ts, pd.Timestamp):
        ty, tm, td = _shift_date_parts(ts.year, ts.month, ts.day, offset)
        return ts.replace(year=ty, month=tm, day=td)

    if isinstance(ts, datetime):
        ty, tm, td = _shift_date_parts(ts.year, ts.month, ts.day, offset)
        return ts.replace(year=ty, month=tm, day=td)

    if isinstance(ts, date):
        ty, tm, td = _shift_date_parts(ts.year, ts.month, ts.day, offset)
        return date(ty, tm, td)

    # Fallback via pandas Timestamp
    dt = pd.to_datetime(ts)
    ty, tm, td = _shift_date_parts(dt.year, dt.month, dt.day, offset)
    return dt.replace(year=ty, month=tm, day=td)


def to_historical_timestamp(
    ts: Any,
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> Any:
    """Reverse-map a simulated timestamp/date back to its original historical timestamp.

    Example: '2026-09-13 18:30:00' -> '2013-09-13 18:30:00'.
    """
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return ts
    if isinstance(ts, pd.Timestamp) and pd.isna(ts):
        return pd.NaT

    offset = -get_year_offset(simulation_year, base_year)

    if isinstance(ts, str):
        ts_clean = ts.strip()
        # Month string: 'YYYY-MM'
        if len(ts_clean) == 7 and ts_clean[4] == "-":
            y = int(ts_clean[:4])
            m = ts_clean[5:7]
            return f"{y + offset:04d}-{m}"
        # Date string: 'YYYY-MM-DD'
        if len(ts_clean) == 10 and ts_clean[4] == "-" and ts_clean[7] == "-":
            y = int(ts_clean[:4])
            m = int(ts_clean[5:7])
            d = int(ts_clean[8:10])
            ty, tm, td = _shift_date_parts(y, m, d, offset)
            return f"{ty:04d}-{tm:02d}-{td:02d}"
        dt = pd.to_datetime(ts_clean)
        ty, tm, td = _shift_date_parts(dt.year, dt.month, dt.day, offset)
        new_dt = dt.replace(year=ty, month=tm, day=td)
        if "T" in ts_clean:
            return new_dt.strftime("%Y-%m-%dT%H:%M:%S")
        return new_dt.strftime("%Y-%m-%d %H:%M:%S")

    if isinstance(ts, pd.Timestamp):
        ty, tm, td = _shift_date_parts(ts.year, ts.month, ts.day, offset)
        return ts.replace(year=ty, month=tm, day=td)

    if isinstance(ts, datetime):
        ty, tm, td = _shift_date_parts(ts.year, ts.month, ts.day, offset)
        return ts.replace(year=ty, month=tm, day=td)

    if isinstance(ts, date):
        ty, tm, td = _shift_date_parts(ts.year, ts.month, ts.day, offset)
        return date(ty, tm, td)

    dt = pd.to_datetime(ts)
    ty, tm, td = _shift_date_parts(dt.year, dt.month, dt.day, offset)
    return dt.replace(year=ty, month=tm, day=td)


def to_simulated_date(
    d: Union[date, datetime, pd.Timestamp, str],
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> date:
    """Convert any date representation to a simulated date object."""
    res = to_simulated_timestamp(d, simulation_year=simulation_year, base_year=base_year)
    if isinstance(res, (datetime, pd.Timestamp)):
        return res.date()
    if isinstance(res, str):
        return pd.to_datetime(res).date()
    if isinstance(res, date):
        return res
    raise TypeError(f"Cannot convert {type(d)} to simulated date")


def to_historical_date(
    d: Union[date, datetime, pd.Timestamp, str],
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> date:
    """Convert any date representation from simulated date to historical date object."""
    res = to_historical_timestamp(d, simulation_year=simulation_year, base_year=base_year)
    if isinstance(res, (datetime, pd.Timestamp)):
        return res.date()
    if isinstance(res, str):
        return pd.to_datetime(res).date()
    if isinstance(res, date):
        return res
    raise TypeError(f"Cannot convert {type(d)} to historical date")


def to_simulated_month(
    m_str: str,
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> str:
    """Shift month string 'YYYY-MM' to simulated timeline (e.g. '2013-09' -> '2026-09')."""
    return to_simulated_timestamp(m_str, simulation_year=simulation_year, base_year=base_year)


def to_historical_month(
    m_str: str,
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> str:
    """Shift month string 'YYYY-MM' from simulated timeline back to historical (e.g. '2026-09' -> '2013-09')."""
    return to_historical_timestamp(m_str, simulation_year=simulation_year, base_year=base_year)


def map_date_range_to_historical(
    start: Union[date, str],
    end: Union[date, str],
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> Tuple[date, date]:
    """Map a simulated (start, end) date pair into the corresponding historical date pair for SQL queries."""
    hist_start = to_historical_date(start, simulation_year=simulation_year, base_year=base_year)
    hist_end = to_historical_date(end, simulation_year=simulation_year, base_year=base_year)
    return hist_start, hist_end


def map_date_range_to_simulated(
    start: Union[date, str],
    end: Union[date, str],
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> Tuple[date, date]:
    """Map a historical (start, end) date pair into the corresponding simulated date pair for display."""
    sim_start = to_simulated_date(start, simulation_year=simulation_year, base_year=base_year)
    sim_end = to_simulated_date(end, simulation_year=simulation_year, base_year=base_year)
    return sim_start, sim_end


def to_simulated_series(
    series: pd.Series,
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> pd.Series:
    """Shift a Pandas Series of timestamps or dates to the simulated timeline.

    Returns a new Series without modifying the input Series.
    """
    if series.empty:
        return series.copy()

    offset = get_year_offset(simulation_year, base_year)

    # If datetime dtype, use DateOffset
    if pd.api.types.is_datetime64_any_dtype(series):
        return series + pd.DateOffset(years=offset)

    # If string or object series
    return series.apply(lambda x: to_simulated_timestamp(x, simulation_year=simulation_year, base_year=base_year))


def to_simulated_dataframe(
    df: pd.DataFrame,
    timestamp_cols: Sequence[str] | None = None,
    date_cols: Sequence[str] | None = None,
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> pd.DataFrame:
    """Transform specified timestamp and date columns in a DataFrame without mutating the original.

    Preserves all consumption values, numerical metrics, and metadata.
    Also updates 'day_of_week' and 'is_weekend' if present so they align with the 2026 calendar.
    """
    out = df.copy()
    if out.empty:
        return out

    offset = get_year_offset(simulation_year, base_year)

    # Detect timestamp columns if not explicitly provided
    if timestamp_cols is None:
        candidate_ts = ["timestamp", "received_at"]
        timestamp_cols = [c for c in candidate_ts if c in out.columns]

    # Detect date columns if not explicitly provided
    if date_cols is None:
        candidate_d = ["date", "target_date"]
        date_cols = [c for c in candidate_d if c in out.columns]

    reference_dt_col = None

    for col in timestamp_cols:
        if col in out.columns:
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                converted = pd.to_datetime(out[col])
                out[col] = converted + pd.DateOffset(years=offset)
            else:
                out[col] = out[col] + pd.DateOffset(years=offset)
            if reference_dt_col is None:
                reference_dt_col = out[col]

    for col in date_cols:
        if col in out.columns:
            if not pd.api.types.is_datetime64_any_dtype(out[col]):
                dt_series = pd.to_datetime(out[col]) + pd.DateOffset(years=offset)
                # Keep string format if original was string
                if isinstance(out[col].iloc[0], str):
                    out[col] = dt_series.dt.strftime("%Y-%m-%d")
                else:
                    out[col] = dt_series
            else:
                out[col] = out[col] + pd.DateOffset(years=offset)
            if reference_dt_col is None:
                reference_dt_col = pd.to_datetime(out[col])

    # Synchronize calendar-derived metadata columns if present
    if reference_dt_col is not None:
        ref = pd.to_datetime(reference_dt_col)
        if "day_of_week" in out.columns:
            out["day_of_week"] = ref.dt.day_name()
        if "is_weekend" in out.columns:
            orig_is_bool = pd.api.types.is_bool_dtype(df["is_weekend"])
            new_weekend = ref.dt.dayofweek >= 5
            out["is_weekend"] = new_weekend if orig_is_bool else new_weekend.astype(int)
        if "month" in out.columns:
            if pd.api.types.is_numeric_dtype(df["month"]):
                out["month"] = ref.dt.month
            else:
                out["month"] = ref.dt.strftime("%Y-%m")

    return out


def transform_forecast_date(
    target_date: Union[str, date, datetime, pd.Timestamp],
    simulation_year: int = DEFAULT_SIMULATION_YEAR,
    base_year: int = DEFAULT_BASE_YEAR,
) -> str:
    """Format forecast target date into the simulated timeline using the standard year offset.

    Example: '2014-02-28' + 13 years -> '2027-02-28'.
    Returns ISO date string 'YYYY-MM-DD'.
    """
    sim_d = to_simulated_date(target_date, simulation_year=simulation_year, base_year=base_year)
    return sim_d.strftime("%Y-%m-%d")
