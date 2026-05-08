#!/usr/bin/env python3
"""Stratified sample N rows per class from a labeled evaluate CSV (e.g. test_split_eval.csv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from llm_agent.sample_eval_subset import stratified_sample_for_eval  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", type=Path, default=_REPO / "data" / "test_split_eval.csv")
    ap.add_argument("-o", "--output", type=Path, default=_REPO / "data" / "eval_balanced_4k.csv")
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--models-dir", type=Path, default=_REPO / "models")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--no-meta", action="store_true", help="Do not write output.meta.json")
    args = ap.parse_args()

    try:
        meta = stratified_sample_for_eval(
            args.input,
            args.output,
            models_dir=args.models_dir,
            per_class=args.per_class,
            seed=args.seed,
            write_meta=not args.no_meta,
        )
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 2

    print(f"Wrote {meta['output_csv']} ({meta['total_rows_written']} rows)")
    if meta.get("meta_json"):
        print(f"Meta: {meta['meta_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
