"""Stratified fixed-N-per-label downsampling for ``cli.py evaluate`` CSVs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

LABEL_ORDER = ("BENIGN", "DDOS_UDP", "DOS_TCP", "RECON")


def stratified_sample_for_eval(
    input_csv: Path,
    output_csv: Path,
    *,
    models_dir: Path,
    per_class: int,
    seed: int,
    write_meta: bool = True,
) -> Dict[str, Any]:
    """
    Read a CSV with ``Label`` + model feature columns; write ``per_class`` random rows per label.

    Uses independent RNG offsets per label (same pattern as ``scripts/make_eval_sample.py``).
    If a class has fewer than ``per_class`` rows, takes all rows and records ``requested``.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)
    models_dir = Path(models_dir)

    fc_path = models_dir / "feature_cols.joblib"
    if not fc_path.is_file():
        raise FileNotFoundError(f"Missing {fc_path}")
    feature_cols: List[str] = list(joblib.load(fc_path))

    df = pd.read_csv(input_csv)
    if "Label" not in df.columns:
        raise ValueError(f"{input_csv} must contain a 'Label' column.")

    missing_fc = [c for c in feature_cols if c not in df.columns]
    if missing_fc:
        raise ValueError(
            f"CSV missing {len(missing_fc)} model feature columns (showing up to 8): {missing_fc[:8]}"
        )

    df["Label"] = df["Label"].astype(str)
    chunks: List[pd.DataFrame] = []
    per_label_counts: Dict[str, Dict[str, int]] = {}
    rng_step = seed

    for label in LABEL_ORDER:
        sub = df.loc[df["Label"] == label]
        available = len(sub)
        take = min(per_class, available)
        per_label_counts[label] = {"available": available, "sampled": take, "requested": per_class}
        if take == 0:
            continue
        chunks.append(sub.sample(n=take, random_state=rng_step))
        rng_step += 1

    if not chunks:
        raise ValueError("No rows matched expected labels " + ", ".join(LABEL_ORDER))

    out_df = pd.concat(chunks, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    export = out_df[feature_cols + ["Label"]].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(output_csv, index=False)

    record: Dict[str, Any] = {
        "input_csv": str(input_csv.resolve()),
        "output_csv": str(output_csv.resolve()),
        "per_class_requested": per_class,
        "seed": seed,
        "total_rows_written": int(len(export)),
        "per_label": per_label_counts,
    }
    if write_meta:
        meta_path = output_csv.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        record["meta_json"] = str(meta_path.resolve())
    return record
