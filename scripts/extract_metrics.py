#!/usr/bin/env python3
"""Extract per-update PPO metrics from every drl_logs/*.log file into a single
CSV. Each row is one (ws, config, seed, upd) tuple with all the columns the
training loop prints: pg, v, ent, aux, ret, fh_ratio, fh_components.

Usage:
    ./scripts/extract_metrics.py [--out /path/to/metrics.csv]

Defaults to writing $HOME/drl_logs/metrics.csv. Idempotent — safe to re-run
while the matrix is still training; just rewrites the CSV from current logs.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from glob import glob
from pathlib import Path

LINE_RE = re.compile(
    r"^upd (\d+)/(\d+) step (\d+) sps (\S+) pg (\S+) v (\S+) ent (\S+) "
    r"aux (\S+) ret (\S+) fh (\S+)/(\S+)"
)
# log filename schema: <ws>_<config>_s<seed>.log  e.g. ws4_oca_s0.log
FNAME_RE = re.compile(r"^(ws\d+)_(baseline|oca|dpr)_s(\d+)\.log$")


def parse_log(path: str):
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        return []
    ws, config, seed = m.group(1), m.group(2), int(m.group(3))
    rows = []
    with open(path) as fh:
        for line in fh:
            lm = LINE_RE.match(line)
            if not lm:
                continue
            rows.append({
                "ws": ws,
                "config": config,
                "seed": seed,
                "upd": int(lm.group(1)),
                "total_upd": int(lm.group(2)),
                "step": int(lm.group(3)),
                "sps": int(lm.group(4)),
                "pg": float(lm.group(5)),
                "v": float(lm.group(6)),
                "ent": float(lm.group(7)),
                "aux": float(lm.group(8)),
                "ret": float(lm.group(9)),
                "fh_ratio": float(lm.group(10)),
                "fh_comps": float(lm.group(11)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default=str(Path.home() / "drl_logs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_path = args.out or os.path.join(args.logs_dir, "metrics.csv")
    cols = ["ws", "config", "seed", "upd", "total_upd", "step", "sps",
            "pg", "v", "ent", "aux", "ret", "fh_ratio", "fh_comps"]
    all_rows = []
    for path in sorted(glob(os.path.join(args.logs_dir, "*.log"))):
        all_rows.extend(parse_log(path))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    by_config = {}
    for r in all_rows:
        by_config.setdefault(r["config"], 0)
        by_config[r["config"]] += 1
    print(f"wrote {len(all_rows)} rows to {out_path}")
    for c, n in sorted(by_config.items()):
        print(f"  {c}: {n} rows")


if __name__ == "__main__":
    main()
