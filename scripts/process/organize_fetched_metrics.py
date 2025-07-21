#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json, logging
from pathlib import Path
from typing import List

import pandas as pd

IN_DIR = Path("data_csv/btc_metrics_grass")
OUT = Path("data_csv/btc_metrics_grass_merged.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def load_one(file: Path) -> pd.DataFrame:
    with file.open() as f:
        js = json.load(f)
    if not js:
        logging.warning("empty %s", file.name)
        return pd.DataFrame()

    metric = file.stem

    if "v" in js[0]:
        df = pd.DataFrame(js).rename(columns={"t": "timestamp", "v": metric})
    else:
        df = pd.json_normalize(js)
        df.rename(columns={"t": "timestamp"}, inplace=True)
        df.columns = ["timestamp"] + [
            f"{metric}_{c.split('.', 1)[1]}" for c in df.columns if c != "timestamp"
        ]
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def main() -> None:
    frames: List[pd.DataFrame] = []
    for jf in IN_DIR.glob("*.json"):
        logging.info("reading %s", jf.name)
        df = load_one(jf)
        if not df.empty:
            frames.append(df)

    if not frames:
        logging.error("no json files parsed")
        return

    master = frames[0]
    for sub in frames[1:]:
        master = master.merge(sub, on="timestamp", how="outer")

    master.sort_values("timestamp", inplace=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT, index=False)
    logging.info(
        "Merged %d files -> %s  (%d rows and %d cols)",
        len(frames),
        OUT,
        len(master),
        len(master.columns),
    )


if __name__ == "__main__":
    main()
