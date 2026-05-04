#!/usr/bin/env python3
"""
Download CIC-BCCC-NRC TabularIoTAttacks-2024 from Kaggle and stage the four CSVs
expected by `eda.ipynb`, `cli.py train`, and `llm_agent/train_ml.py`.

Prerequisites
--------------
  pip install kaggle

  Authentication (pick one):
  - Recommended: ``~/.kaggle/kaggle.json`` (from Kaggle → Settings → API → Create New Token).
  - Optional: add ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` to this repo's ``.env`` (gitignored here);
    this script loads ``.env`` from the project root before calling the Kaggle CLI.

Dataset: https://www.kaggle.com/datasets/kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

KAGGLE_SLUG = "kabeleswarpe/cic-bccc-nrc-tabulariotattacks-2024"

REQUIRED = [
    "Benign Traffic.csv",
    "DDoS UDP Flood.csv",
    "DoS TCP Flood.csv",
    "Recon Port Scan.csv",
]


def _run_kaggle_download(dest_dir: Path) -> Path:
    """
    Download dataset zip into ``dest_dir``.

    Uses ``KaggleApi`` (the ``kaggle`` PyPI package does not support ``python -m kaggle``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_SLUG!r} via KaggleApi → {dest_dir}")
    api.dataset_download_files(KAGGLE_SLUG, path=str(dest_dir), unzip=False, quiet=False)
    zips = list(dest_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No .zip found in {dest_dir} after download.")
    if len(zips) > 1:
        print("Warning: multiple zips; using newest:", zips)
    return max(zips, key=lambda p: p.stat().st_mtime)


def _unzip(zip_path: Path, extract_to: Path) -> None:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def _collect_csvs(root: Path) -> list[Path]:
    return sorted({p.resolve() for p in root.rglob("*") if p.suffix.lower() == ".csv"})


def _map_csvs(extract_root: Path, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    pool = _collect_csvs(extract_root)
    used: set[Path] = set()

    # 1) Exact filename match (case-insensitive)
    lower_to_path: dict[str, Path] = {}
    for p in pool:
        lower_to_path.setdefault(p.name.lower(), p)

    for target in REQUIRED:
        p = lower_to_path.get(target.lower())
        if p is not None:
            shutil.copy2(p, data_dir / target)
            used.add(p)
            print(f"OK {target} <- {p}")

    # 2) Heuristic match on basename for anything still missing
    heuristics: list[tuple[str, re.Pattern[str]]] = [
        ("Benign Traffic.csv", re.compile(r"benign", re.I)),
        ("DDoS UDP Flood.csv", re.compile(r"ddos.*udp|udp.*flood", re.I)),
        ("DoS TCP Flood.csv", re.compile(r"dos.*tcp|tcp.*flood", re.I)),
        ("Recon Port Scan.csv", re.compile(r"recon.*port|port.*scan", re.I)),
    ]
    for target, rx in heuristics:
        if (data_dir / target).is_file():
            continue
        for p in pool:
            if p in used:
                continue
            if rx.search(p.name):
                shutil.copy2(p, data_dir / target)
                used.add(p)
                print(f"OK {target} <- {p} (heuristic)")
                break

    missing = [t for t in REQUIRED if not (data_dir / t).is_file()]
    if missing:
        print("\nStill missing (please copy/rename manually):", missing)
        print("CSV files found in archive:")
        for p in pool:
            print(" ", p)
        raise SystemExit(1)

    print(f"\nAll four files are in: {data_dir.resolve()}")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)
    except ImportError:
        pass

    ap = argparse.ArgumentParser(description="Download TabularIoT 2024 from Kaggle into data/")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data",
        help="Where to write Benign Traffic.csv, etc.",
    )
    args = ap.parse_args()
    data_dir: Path = args.data_dir

    if importlib.util.find_spec("kaggle") is None:
        print("Install the Kaggle client: pip install kaggle", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        dl = tdp / "download"
        ex = tdp / "extract"
        z = _run_kaggle_download(dl)
        _unzip(z, ex)
        _map_csvs(ex, data_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
