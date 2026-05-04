#!/usr/bin/env python3
"""
Build a single labeled CSV for `cli.py evaluate` from the four class CSVs in data/.

Does not import ``llm_agent.train_ml`` (avoids pulling SHAP/XGBoost). Logic matches
that module for loading + dropping identifier columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

DROP_COLS = ("Src IP", "Dst IP", "Timestamp", "Attack Name")

_REPO = Path(__file__).resolve().parent.parent


def load_combined_dataframe(data_dir: Path, *, dos_tcp_nrows: int = 100_000) -> pd.DataFrame:
    data_dir = Path(data_dir)
    benign = pd.read_csv(data_dir / "Benign Traffic.csv")
    ddos = pd.read_csv(data_dir / "DDoS UDP Flood.csv")
    dos_tcp = pd.read_csv(data_dir / "DoS TCP Flood.csv", nrows=dos_tcp_nrows)
    recon = pd.read_csv(data_dir / "Recon Port Scan.csv")
    benign["Label"] = "BENIGN"
    ddos["Label"] = "DDOS_UDP"
    dos_tcp["Label"] = "DOS_TCP"
    recon["Label"] = "RECON"
    return pd.concat([benign, ddos, dos_tcp, recon], ignore_index=True)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=drop, errors="ignore").copy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Create eval_sample.csv from raw class CSVs")
    ap.add_argument("--data-dir", type=Path, default=_REPO / "data")
    ap.add_argument("--models-dir", type=Path, default=_REPO / "models")
    ap.add_argument("-o", "--output", type=Path, default=_REPO / "data" / "eval_sample.csv")
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--dos-tcp-rows", type=int, default=100_000)
    args = ap.parse_args()

    fc_path = args.models_dir / "feature_cols.joblib"
    if not fc_path.is_file():
        print(f"Missing {fc_path}. Train models first or set --models-dir.", file=sys.stderr)
        return 2

    feature_cols: list[str] = list(joblib.load(fc_path))

    df = load_combined_dataframe(args.data_dir, dos_tcp_nrows=args.dos_tcp_rows)
    df = preprocess_dataframe(df)

    rng = args.seed
    chunks: list[pd.DataFrame] = []
    for label in ("BENIGN", "DDOS_UDP", "DOS_TCP", "RECON"):
        sub = df.loc[df["Label"] == label]
        if len(sub) == 0:
            print(f"Warning: no rows for {label}", file=sys.stderr)
            continue
        take = min(args.per_class, len(sub))
        chunks.append(sub.sample(n=take, random_state=rng))
        rng += 1

    out_df = pd.concat(chunks, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    missing_fc = [c for c in feature_cols if c not in out_df.columns]
    if missing_fc:
        print("Missing feature columns vs model:", missing_fc[:10], file=sys.stderr)
        return 2

    export = out_df[feature_cols + ["Label"]].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(args.output, index=False)
    print(f"Wrote {args.output.resolve()} ({len(export)} rows, {len(feature_cols)} features + Label)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
