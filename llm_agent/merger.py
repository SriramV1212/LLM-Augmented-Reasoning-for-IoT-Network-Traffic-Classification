"""Build a single-mode classification report (ML + explainer, or zero-shot LLM only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional

from llm_agent.config import AgentConfig, ClassifyMode
from llm_agent.flow_explanation import build_merged_why
from llm_agent.track_a_zero_shot import ZeroShotResult
from llm_agent.track_b_ml_explainer import TrackBResult


@dataclass
class MergedFlowReport:
    flow_id: Optional[str]
    classification_mode: str
    zeroshot: Optional[Dict[str, Any]]
    ml_agent: Optional[Dict[str, Any]]
    final_prediction: Optional[str]
    run_notes: str
    why: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _zeroshot_dict(z: ZeroShotResult) -> Dict[str, Any]:
    return {
        "prediction": z.predicted_class,
        "confidence": z.confidence,
        "plausibility": z.plausibility,
        "reasoning_steps": z.reasoning_steps,
        "short_rationale": z.short_rationale,
        "llm_model": z.model,
    }


def _ml_agent_dict(t: TrackBResult) -> Dict[str, Any]:
    return {
        "prediction": t.prediction.predicted_class,
        "confidence": t.prediction.confidence,
        "probabilities": t.prediction.probabilities,
        "ml_model": t.ml_model,
        "plausibility": t.plausibility,
        "explanation": t.explanation,
        "security_takeaway": t.security_takeaway,
        "rule_based_summary": t.rule_based_summary,
        "top_shap": [
            {"feature": s.feature, "log1p_sign_value": s.raw_value, "shap": s.shap_value}
            for s in t.top_shap
        ],
        "llm_model": t.llm_model,
    }


def merge_results(
    *,
    flow_id: Optional[str],
    classification_mode: ClassifyMode,
    zs: Optional[ZeroShotResult],
    tb: Optional[TrackBResult],
    config: AgentConfig,
    raw_features: Optional[Mapping[str, float]] = None,
) -> MergedFlowReport:
    """
    One classification path per call:

    - ``ml_agent``: ML prediction + optional LLM narrative (+ separate SHAP-backed artifacts on ``tb``). ``tb`` required.
    - ``zeroshot``: LLM-only label + chain-of-thought. ``zs`` required.
    """
    _ = config  # reserved for future options
    zs_dict = _zeroshot_dict(zs) if zs else None
    ml_dict = _ml_agent_dict(tb) if tb else None

    final: Optional[str] = None
    notes = ""

    if classification_mode == "ml_agent":
        if tb is None:
            raise ValueError("ml_agent mode requires a TrackBResult (ML path).")
        final = tb.prediction.predicted_class
        notes = "ML + agent mode: tree model prediction with optional LLM narrative (SHAP summary may appear separately)."
    elif classification_mode == "zeroshot":
        if zs is None:
            raise ValueError("zeroshot mode requires a ZeroShotResult (LLM path).")
        final = zs.predicted_class
        notes = "Zero-shot LLM mode: label and rationale from the language model only."
    else:
        raise ValueError(f"Unknown classification_mode: {classification_mode}")

    why = build_merged_why(
        flow_id=flow_id,
        final_prediction=final,
        classification_mode=classification_mode,
        run_notes=notes,
        zs=zs,
        tb=tb,
        raw_feature_dict=raw_features,
    )

    return MergedFlowReport(
        flow_id=flow_id,
        classification_mode=classification_mode,
        zeroshot=zs_dict,
        ml_agent=ml_dict,
        final_prediction=final,
        run_notes=notes,
        why=why,
    )
