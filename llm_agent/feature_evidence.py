"""Deterministic raw-feature bullets for LLM prompts and fallbacks (no SHAP)."""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple


def top_raw_pairs(raw_lookup: Mapping[str, float], top_k: int) -> List[Tuple[str, float]]:
    pairs = [(str(k), float(v)) for k, v in raw_lookup.items()]
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return pairs[:top_k]


def flow_evidence_markdown(
    predicted_label: str,
    confidence: float,
    raw_lookup: Mapping[str, float],
    *,
    top_k: int = 5,
) -> str:
    """Bullet list of highest-|value| raw features for narrative grounding."""
    lines = [
        f"**Flow evidence for predicted class `{predicted_label}`** "
        f"(confidence {confidence:.1%}; raw CICFlowMeter units):"
    ]
    for name, value in top_raw_pairs(raw_lookup, top_k):
        lines.append(f"- **{name}**: raw ≈ {value:.6g}.")
    return "\n".join(lines)


def mentions_top_feature_names(text: str, raw_lookup: Mapping[str, float], *, k: int = 2) -> bool:
    if not text:
        return False
    names = [name for name, _ in top_raw_pairs(raw_lookup, top_k=k)]
    return all(name in text for name in names)
