#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, csv, logging, requests, os

from pathlib import Path
from tqdm import tqdm

START_TS = 1752713100
OUT_FILE = Path("data_csv/coindesk_news_sentiment.csv")
API_URL = "https://data-api.coindesk.com/news/v1/article/list"
LIMIT = 100
HEADERS = {"User-Agent": "NewsCrawler/1.0"}
LANG = "EN"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def get_batch(to_ts: int):
    attempt = 0
    while True:
        try:
            r = requests.get(
                API_URL,
                params={"lang": LANG, "limit": LIMIT, "to_ts": to_ts},
                headers=HEADERS,
                timeout=15,
            )
            r.raise_for_status()
            return r.json().get("Data", [])
        except Exception as exc:
            attempt += 1
            wait = 1 if attempt == 1 else 600
            logging.warning(f"[{to_ts}] {type(exc).__name__}: {exc} | retry {wait}s")
            time.sleep(wait)


def last_timestamp_from_file(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    with path.open("rb") as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b"\n":
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    try:
        return int(last_line.split(",")[1])
    except Exception:
        return None


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    last_ts = last_timestamp_from_file(OUT_FILE)
    ts_cursor = last_ts - 1 if last_ts else START_TS
    mode = "a" if last_ts else "w"
    logging.info(
        f"{'Resuming' if last_ts else 'Starting'} from ts={ts_cursor} "
        f"({'append' if mode=='a' else 'new file'})"
    )
    fieldnames = ["id", "published_on", "sentiment"]
    with OUT_FILE.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        total = (
            0 if mode == "w" else sum(1 for _ in open(OUT_FILE, encoding="utf-8")) - 1
        )
        pbar = tqdm(desc="Downloading batches", unit="batch")
        while True:
            batch = get_batch(ts_cursor)
            if not batch:
                break
            for art in batch:
                writer.writerow(
                    {
                        "id": art.get("ID", ""),
                        "published_on": art.get("PUBLISHED_ON", ""),
                        "sentiment": art.get("SENTIMENT", ""),
                    }
                )
            total += len(batch)
            earliest = min(art.get("PUBLISHED_ON", "") for art in batch)
            ts_cursor = int(earliest) - 1
            pbar.update(1)
            pbar.set_postfix(total_articles=total)
        pbar.close()
        logging.info(f"Finished. Total articles: {total:,d}")
        logging.info(f"Saved to {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
