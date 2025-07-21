#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nohup /Users/lijiheng/CS/Environs/python/python3.11/bin/python3.11 fetch_historical_candles.py > logs/out.log 2>&1 &

import requests
import time
import csv
import math

from datetime import datetime, timedelta, timezone
from tqdm import tqdm

PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "SUI-USD"]
GRANULARITY = 900
START_DATE = datetime(2013, 1, 1, tzinfo=timezone.utc)
PAGE_LIMIT = 300
REQUEST_PAUSE = 0.34
root_dir = "data_csv"

session = requests.Session()


def fetch_page(product: str, start: datetime, end: datetime):
    url = f"https://api.exchange.coinbase.com/products/{product}/candles"
    params = {
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
        "granularity": GRANULARITY,
    }
    resp = session.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_all(product: str):
    fname = f"{root_dir}/{product.lower().replace('-', '_')}_{GRANULARITY}s_candles.csv"
    with open(fname, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "low", "high", "open", "close", "volume"])

        now = datetime.now(timezone.utc)
        total_seconds = (now - START_DATE).total_seconds()
        seconds_per_page = GRANULARITY * PAGE_LIMIT
        total_pages = math.ceil(total_seconds / seconds_per_page)

        cursor = START_DATE
        pbar = tqdm(total=total_pages, desc=f"Fetch {product}", unit="page")

        while cursor < now:
            next_cursor = cursor + timedelta(seconds=seconds_per_page)
            if next_cursor > now:
                next_cursor = now

            try:
                data = fetch_page(product, cursor, next_cursor)
            except Exception as e:
                time.sleep(5)
                data = fetch_page(product, cursor, next_cursor)

            for candle in data:
                writer.writerow(candle)

            cursor = next_cursor
            pbar.update(1)
            time.sleep(REQUEST_PAUSE)

        pbar.close()


def main():
    for prod in PRODUCTS:
        print(f"\nStarting fetch for {prod} from {START_DATE.date()} to now...")
        fetch_all(prod)
        print(
            f"Finished {prod}, saved to {prod.lower().replace('-', '_')}_{GRANULARITY}s_candles.csv"
        )


if __name__ == "__main__":
    main()
