#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fredapi import Fred

FRED_KEY = os.getenv("FRED_API_KEY")
OUT_CSV = Path("data_csv/macro_liquidity.csv")
START_DATE = "2015-07-21"

SERIES = {
    "dxy": "DTWEXBGS",
    "yield10": "DGS10",
    "yield2": "DGS2",
}

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

fred = Fred(api_key=FRED_KEY)


def safe_get(series: str, start: str) -> pd.Series:
    attempt = 0
    while True:
        try:
            return fred.get_series(series, observation_start=start)
        except Exception as e:
            attempt += 1
            wait = 1 if attempt == 1 else 600
            logging.warning("%s  -> retry in %ss", e, wait)
            time.sleep(wait)


def last_timestamp(path: Path) -> Optional[int]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    *_, last = path.read_text().strip().splitlines()
    try:
        return int(last.split(",")[0])
    except:
        return None


def main() -> None:
    if OUT_CSV.exists():
        ts = last_timestamp(OUT_CSV)
        if ts:
            start_dt = datetime.fromtimestamp(ts, timezone.utc).date() + timedelta(
                days=1
            )
            start_str = start_dt.isoformat()
        else:
            start_str = START_DATE
    else:
        start_str = START_DATE
    logging.info("Fetch start date: %s", start_str)

    frames = []
    for col, code in SERIES.items():
        s = safe_get(code, start_str)
        s.name = col
        frames.append(s)

    if not frames or all(f.empty for f in frames):
        logging.info("No new data.")
        return

    df_new = pd.concat(frames, axis=1).astype(float)
    df_new["yield_spread"] = df_new["yield10"] - df_new["yield2"]
    df_new.index = pd.to_datetime(df_new.index).tz_localize("UTC")

    if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
        df_old = pd.read_csv(
            OUT_CSV,
            parse_dates=["timestamp"],
            date_parser=lambda x: pd.to_datetime(int(x), unit="s", utc=True),
        )
        df_old.set_index("timestamp", inplace=True)
        df = pd.concat([df_old, df_new[~df_new.index.isin(df_old.index)]])
    else:
        df = df_new

    df.sort_index(inplace=True)
    df.ffill(inplace=True)

    df_reset = df.reset_index().rename(columns={"index": "timestamp"})
    df_reset["timestamp"] = df_reset["timestamp"].astype("int64") // 10**9
    df_reset.to_csv(OUT_CSV, index=False)
    logging.info("Saved %d rows -> %s / %s", len(df_reset), OUT_CSV.name)


if __name__ == "__main__":
    main()
