"""
Unified "why this label" digest for a single classification path.

Unit of analysis: CICFlowMeter rows are **flows** (aggregated statistics), not single packets.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from llm_agent.track_a_zero_shot import ZeroShotResult
from llm_agent.track_b_ml_explainer import TrackBResult, shap_to_evidence_list


def build_merged_why(
    *,
    flow_id: Optional[str],
    final_prediction: Optional[str],
    classification_mode: str,
    run_notes: str,
    zs: Optional[ZeroShotResult],
    tb: Optional[TrackBResult],
    raw_feature_dict: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """
    Structured explanation for API / CLI JSON.

    Clarifies that inputs are **flows** (CICFlowMeter), not single packets.
    """
    unit = (
        "Each input describes one **network flow** (CICFlowMeter): statistics aggregated "
        "over that flow (duration, packet counts, bytes/s, flags, etc.). "
        "It is **not** a single raw packet capture."
    )

    zeroshot_block: Optional[Dict[str, Any]] = None
    if zs is not None:
        zeroshot_block = {
            "predicted_class": zs.predicted_class,
            "confidence": zs.confidence,
            "reasoning_steps": zs.reasoning_steps,
            "short_rationale": zs.short_rationale,
        }

    raw_lookup: Mapping[str, float] = raw_feature_dict or {}

    ml_agent_block: Optional[Dict[str, Any]] = None
    if tb is not None:
        ml_agent_block = {
            "predicted_class": tb.prediction.predicted_class,
            "confidence": tb.prediction.confidence,
            "probabilities": tb.prediction.probabilities,
            "ml_backend": tb.ml_model,
            "top_evidence_shap": shap_to_evidence_list(
                tb.top_shap, raw_lookup, top_k=len(tb.top_shap)
            ),
            "plausibility_llm": tb.plausibility,
            "llm_explanation": tb.explanation,
            "security_takeaway_llm": tb.security_takeaway,
            "rule_based_summary": tb.rule_based_summary,
        }

    narrative_parts: List[str] = [unit]
    mode_label = (
        "ML + agent (tree model + SHAP + optional LLM explainer)"
        if classification_mode == "ml_agent"
        else "Zero-shot LLM"
    )
    if final_prediction:
        narrative_parts.append(
            f"Mode: **{mode_label}**. The predicted label is **{final_prediction}**. "
            f"{run_notes}".strip()
        )
    else:
        narrative_parts.append(
            f"Mode: **{mode_label}**. No label was produced. {run_notes}".strip()
        )
    if zs is not None and zs.short_rationale:
        narrative_parts.append(f"LLM rationale: {zs.short_rationale}")
    if tb is not None and tb.rule_based_summary:
        narrative_parts.append(
            "ML + SHAP evidence (first line): " + tb.rule_based_summary.split("\n")[0]
        )

    return {
        "unit_of_analysis": unit,
        "flow_id": flow_id,
        "classification_mode": classification_mode,
        "final_prediction": final_prediction,
        "run_notes": run_notes,
        "zeroshot": zeroshot_block,
        "ml_agent": ml_agent_block,
        "narrative": "\n\n".join(narrative_parts),
    }
