#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# nohup /Users/lijiheng/CS/Environs/python/python3.11/bin/python3.11 realtime_ticker_tracker.py > logs/out.log 2>&1 &

import time
import requests
import csv
import os
from datetime import datetime

PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD", "SUI-USD"]
INTERVAL_SECONDS = 10 * 60
root_dir = "../data_csv"

FIELDNAMES = [
    "timestamp",
    "trade_id",
    "price",
    "size",
    "bid",
    "ask",
    "volume",
]


def fetch_ticker(product):
    url = f"https://api.exchange.coinbase.com/products/{product}/ticker"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def init_csv(path):
    if not os.path.isfile(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_to_csv(path, row):
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def main():
    for product in PRODUCTS:
        fname = f"{root_dir}/{product.lower().replace('-', '_')}_ticker.csv"
        init_csv(fname)

    while True:
        for product in PRODUCTS:
            fname = f"{root_dir}/{product.lower().replace('-', '_')}_ticker.csv"
            try:
                ticker = fetch_ticker(product)
                row = {
                    "timestamp": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "trade_id": ticker.get("trade_id"),
                    "price": ticker.get("price"),
                    "size": ticker.get("size"),
                    "bid": ticker.get("bid"),
                    "ask": ticker.get("ask"),
                    "volume": ticker.get("volume"),
                }
                append_to_csv(fname, row)
                print(
                    f"[{row['timestamp']}] {product}: price={row['price']} bid={row['bid']} ask={row['ask']}"
                )
            except Exception as e:
                print(
                    f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {product} Error: {e}"
                )
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
