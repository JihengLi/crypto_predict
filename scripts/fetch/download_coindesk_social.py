#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, csv, logging, requests, os

from pathlib import Path
from tqdm import tqdm

COIN_ID = 1182
START_TS = 1752713100
OUT_FILE = Path("data_csv/social_btc_daily.csv")
API_KEY = os.getenv("COINDESK_API_KEY")
if not API_KEY:
    raise RuntimeError("Environment variable COINDESK_API_KEY is missing!")
API_URL = "https://min-api.cryptocompare.com/data/social/coin/histo/day"
LIMIT = 2000
HEADERS = {
    "User-Agent": "CCSocialCrawler/1.0",
    "authorization": f"Apikey {API_KEY}",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def fetch_batch(to_ts: int):
    attempt = 0
    while True:
        params = {
            "coinId": COIN_ID,
            "limit": LIMIT,
            "toTs": to_ts,
        }
        try:
            r = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json().get("Data", [])
        except Exception as exc:
            attempt += 1
            wait = 1 if attempt == 1 else 600
            logging.warning(
                f"[toTs={to_ts}] {type(exc).__name__}: {exc}  | retry {wait}s"
            )
            time.sleep(wait)


def last_time(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)
        last = f.readline().decode()
    try:
        return int(last.split(",")[0])
    except Exception:
        return None


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    last_ts = last_time(OUT_FILE)
    ts_cursor = (last_ts - 1) if last_ts else START_TS
    mode = "a" if last_ts else "w"
    logging.info(f"{'Resuming' if last_ts else 'Starting'} at ts={ts_cursor} ({mode})")

    first_batch = fetch_batch(ts_cursor)
    if not first_batch:
        logging.info("Nothing returned, exit.")
        return
    header = list(first_batch[0].keys())

    with OUT_FILE.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == "w":
            writer.writeheader()
        batch = first_batch
        pbar = tqdm(desc="Downloading batches", unit="batch")
        while batch:
            for row in batch:
                writer.writerow(row)
            earliest = min(item["time"] for item in batch)
            ts_cursor = earliest - 1
            pbar.update(1)
            batch = fetch_batch(ts_cursor)
        pbar.close()

    logging.info(f"Done. File saved in {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
