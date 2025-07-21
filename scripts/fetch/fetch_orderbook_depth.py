#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, csv, math, ccxt

from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm

EXCHANGE = "coinbaseadvanced"
PAIR = "BTC/USD"
OUT_CSV = Path("data_csv/orderbook_snapshots.csv")
INTERVAL = 15 * 60
DEPTH_LIM = 200

exchange_cls = dict(
    coinbaseadvanced=ccxt.coinbaseadvanced,
    binance=ccxt.binance,
    bybit=ccxt.bybit,
)
if EXCHANGE not in exchange_cls:
    raise ValueError(f"Unsupported exchange {EXCHANGE}")
ex = exchange_cls[EXCHANGE]({"enableRateLimit": True})


def next_slot(ts):
    return (math.floor(ts / INTERVAL) + 1) * INTERVAL


if OUT_CSV.exists() and OUT_CSV.stat().st_size > 0:
    *_, last = OUT_CSV.read_text().strip().splitlines()
    last_ts = (
        datetime.fromisoformat(last.split(",")[0])
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )
    next_t = next_slot(last_ts)
    mode = "a"
else:
    next_t = next_slot(time.time())
    mode = "w"

header = ["ts", "best_bid", "best_ask", "spread", "depth_bid_1pct", "depth_ask_1pct"]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
f = OUT_CSV.open(mode, newline="")
writer = csv.writer(f)
if mode == "w":
    writer.writerow(header)


def utc_iso():
    return datetime.now(timezone.utc).isoformat()


def depth_usd(side, limit_px, levels):
    total = 0.0
    for px, amt in levels:
        if (side == "bid" and px < limit_px) or (side == "ask" and px > limit_px):
            break
        total += px * amt
    return total


pbar = tqdm(total=0, dynamic_ncols=True, desc="OrderBook")
while True:
    now = time.time()
    if now >= next_t:
        try:
            ob = ex.fetch_order_book(PAIR, limit=DEPTH_LIM)
            bid, ask = ob["bids"][0][0], ob["asks"][0][0]
            mid = (bid + ask) / 2
            spread = (ask - bid) / mid

            limit_bid = mid * 0.99
            limit_ask = mid * 1.01
            depth_bid = depth_usd("bid", limit_bid, ob["bids"])
            depth_ask = depth_usd("ask", limit_ask, ob["asks"])

            row = [utc_iso(), bid, ask, spread, depth_bid, depth_ask]
            writer.writerow(row)
            f.flush()

            pbar.update(1)
            pbar.set_postfix(
                spread=f"{spread*100:.2f}%",
                depth=f"{depth_bid/1e6:.1f}M/{depth_ask/1e6:.1f}M",
            )
        except Exception as e:
            pbar.write(f"[WARN] {e}")
            time.sleep(5)
        next_t += INTERVAL
    else:
        time.sleep(min(5, next_t - now))
