"""Serialize CICFlowMeter feature dicts into LLM-friendly text."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Columns emphasized in project EDA / security reasoning (must match `feature_cols.joblib` names)
HIGHLIGHT_FEATURES: Tuple[str, ...] = (
    "Flow Duration",
    "Total Fwd Packet",
    "Total Bwd packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "SYN Flag Count",
    "ACK Flag Count",
    "Dst Port",
    "Protocol",
)


def log1p_sign_transform(values: Sequence[float]) -> List[float]:
    """Match training: sign(x) * log(1 + |x|)."""
    out: List[float] = []
    for v in values:
        fv = float(v)
        out.append(math.copysign(math.log1p(abs(fv)), fv))
    return out


def log_normalize_matrix(arr: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """Apply sign(x) * log1p(|x|) element-wise (same as ml_classifier notebook)."""
    a = np.asarray(arr, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def _fmt_num(x: float) -> str:
    if x == 0:
        return "0"
    ax = abs(x)
    if ax >= 1e6 or (ax < 1e-3 and ax > 0):
        return f"{x:.3e}"
    if ax >= 100:
        return f"{x:.2f}"
    return f"{x:.6g}"


def _annotate(name: str, raw: float) -> str:
    """Lightweight heuristics (no dataset stats required)."""
    notes: List[str] = []
    lname = name.lower()
    if "packet" in lname and "total" in lname and ("bwd" in lname or "backward" in lname):
        if raw == 0:
            notes.append("ALERT: zero backward packets (unidirectional)")
    if "flow packets/s" in lname:
        if raw > 1e5:
            notes.append("EXTREME rate")
        elif raw > 1e4:
            notes.append("very high rate")
    if "flow bytes/s" in lname:
        if raw > 1e7:
            notes.append("EXTREME throughput")
        elif raw > 1e6:
            notes.append("high throughput")
    if "flow duration" in lname:
        if raw > 0 and raw < 1e3:
            notes.append("very short flow")
    if "syn flag" in lname:
        if raw > 50:
            notes.append("elevated SYN activity")
        elif raw > 0:
            notes.append("some SYN activity")
    if notes:
        return " (" + "; ".join(notes) + ")"
    return ""


def serialize_flow_features(
    feature_dict: Mapping[str, float],
    feature_order: Sequence[str],
    *,
    mode: str = "verbose",
    max_other_features: int = 25,
) -> str:
    """
    Build a readable block from raw feature values (same scale as CSV / training).

    mode:
      - "verbose": highlighted features + additional high-magnitude columns
      - "compact": highlighted features only
    """
    lines: List[str] = []
    lines.append("### Network flow (CICFlowMeter features, raw values)")
    lines.append("")

    present = {k: float(feature_dict.get(k, 0.0)) for k in feature_order}

    lines.append("#### Key fields (EDA / security relevance)")
    for name in HIGHLIGHT_FEATURES:
        if name not in present:
            continue
        raw = present[name]
        lines.append(f"- **{name}** = {_fmt_num(raw)}{_annotate(name, raw)}")
    lines.append("")

    if mode == "compact":
        return "\n".join(lines)

    # Remaining features: sort by |value| descending, show top-N beyond highlights
    highlight_set = set(HIGHLIGHT_FEATURES)
    others = [(k, v) for k, v in present.items() if k not in highlight_set]
    others.sort(key=lambda kv: abs(kv[1]), reverse=True)
    lines.append(f"#### Other high-magnitude features (top {max_other_features} by |value|)")
    for k, v in others[:max_other_features]:
        lines.append(f"- {k} = {_fmt_num(v)}{_annotate(k, v)}")
    lines.append("")
    lines.append(
        f"_Total features provided: {len(feature_order)} "
        f"(identifiers such as Flow ID / IPs are excluded from the model.)_"
    )
    return "\n".join(lines)


def flow_vector_from_dict(
    feature_dict: Mapping[str, float],
    feature_order: Sequence[str],
) -> Tuple[List[str], List[float]]:
    """Ordered keys and raw values aligned to training columns."""
    keys = list(feature_order)
    vals = [float(feature_dict.get(k, 0.0)) for k in keys]
    return keys, vals


def build_feature_matrix_row(
    feature_dict: Mapping[str, float],
    feature_order: Sequence[str],
) -> np.ndarray:
    """Single-row float matrix for sklearn / xgboost (raw, then transform in ML path)."""
    _, vals = flow_vector_from_dict(feature_dict, feature_order)
    row = np.array([vals], dtype=float)
    return np.sign(row) * np.log1p(np.abs(row))
