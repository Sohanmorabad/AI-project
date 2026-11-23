"""Replay historical readings into the live API, as if they were arriving now.

1. Start the API in one window:      python src/live_api.py
2. Run this in another window:       python src/replay.py

Examples:
  python src/replay.py --households MAC000002,MAC000003 --start 2026-06-01 --days 2 --delay 0.5
The replayed readings are transformed to a 2026 simulation timeline and labelled source "replay".
"""
import argparse
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

from simulation import (
    to_historical_date,
    to_simulated_timestamp,
)

HISTORY_DB = "data/energy.db"
API_URL = "http://127.0.0.1:5000"


def load_readings(db_path, households, start, days, simulate_2026=False):
    """Historical readings for the chosen households, oldest first, as API-ready dicts.

    If simulate_2026 is True or start is a 2026 date, timestamps are mapped to 2026.
    """
    is_sim_start = str(start).startswith("2026")
    do_simulate = simulate_2026 or is_sim_start

    if is_sim_start:
        query_start_date = to_historical_date(start)
        start_dt = datetime.combine(query_start_date, datetime.min.time())
    else:
        start_dt = datetime.strptime(str(start)[:10], "%Y-%m-%d")

    end_dt = start_dt + timedelta(days=days)
    marks = ",".join("?" for _ in households)
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            f"""SELECT household_id, timestamp, power_kw FROM energy_readings
                WHERE household_id IN ({marks}) AND timestamp >= ? AND timestamp < ?
                ORDER BY timestamp, household_id""",
            [*households, start_dt.strftime("%Y-%m-%d %H:%M:%S"), end_dt.strftime("%Y-%m-%d %H:%M:%S")],
        ).fetchall()
    finally:
        con.close()

    results = []
    for h, ts, p in rows:
        ts_str = str(ts)[:19]
        if do_simulate:
            ts_str = to_simulated_timestamp(ts_str)
        results.append({
            "source": "replay",
            "device_id": h,
            "timestamp": ts_str,
            "power_kw": float(p),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--households", default="MAC000002", help="comma-separated household ids")
    parser.add_argument("--start", default="2026-06-01", help="first day to replay (YYYY-MM-DD, e.g. 2026-06-01)")
    parser.add_argument("--days", type=int, default=2, help="how many days to replay")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds to wait between readings")
    parser.add_argument("--api", default=API_URL, help="address of the live API")
    parser.add_argument("--no-simulate", dest="simulate_2026", action="store_false", default=True,
                        help="keep raw historical 2011-2014 timestamps instead of 2026 simulation")
    args = parser.parse_args()

    import requests  # imported here so the loading code above can be used without it

    households = [h.strip() for h in args.households.split(",") if h.strip()]
    readings = load_readings(HISTORY_DB, households, args.start, args.days, simulate_2026=args.simulate_2026)
    if not readings:
        sys.exit(f"No readings found for {households} from {args.start} for {args.days} day(s). "
                 "Check the household id and the dates.")

    headers = {}
    if os.environ.get("LIVE_API_KEY"):
        headers["X-API-Key"] = os.environ["LIVE_API_KEY"]

    sim_label = "2026 Simulation" if args.simulate_2026 else "Historical"
    print(f"Replaying {len(readings)} readings for {', '.join(households)} [{sim_label}] (Ctrl+C to stop)")
    sent = new = 0
    try:
        with requests.Session() as session:
            for r in readings:
                try:
                    resp = session.post(f"{args.api}/readings", json=r, headers=headers, timeout=5)
                except requests.ConnectionError:
                    sys.exit("Could not reach the API. Is it running? Start it with: python src/live_api.py")
                if resp.status_code != 201:
                    sys.exit(f"The API rejected a reading ({resp.status_code}): {resp.text}")
                sent += 1
                new += resp.json()["saved"]
                print(f"{r['timestamp']}  {r['device_id']}  {r['power_kw']:.3f} kW")
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"Sent {sent} readings, {new} new, {sent - new} already stored.")


if __name__ == "__main__":
    main()
