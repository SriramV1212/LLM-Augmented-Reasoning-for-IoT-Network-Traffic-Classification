#!/usr/bin/env python3
"""Export notebook-aligned SGKF test fold (wrapper around llm_agent.train_ml.export_sgkf_test_csv)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from llm_agent.train_ml import export_sgkf_test_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=_REPO / "data")
    ap.add_argument("-o", "--output", type=Path, default=_REPO / "data" / "test_split_eval.csv")
    ap.add_argument("--models-dir", type=Path, default=_REPO / "models")
    ap.add_argument("--no-check-models", action="store_true")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--sgkf-splits", type=int, default=4)
    ap.add_argument("--dos-tcp-rows", type=int, default=100_000)
    args = ap.parse_args()
    md = None if args.no_check_models else args.models_dir
    meta = export_sgkf_test_csv(
        data_dir=args.data_dir,
        output_csv=args.output,
        random_state=args.seed,
        sgkf_n_splits=args.sgkf_splits,
        dos_tcp_nrows=args.dos_tcp_rows,
        models_dir=md,
    )
    print(f"Wrote {meta['output_csv']} ({meta['n_test_rows']} test rows)")
    print(f"Meta: {meta['meta_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
