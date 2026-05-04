"""Track B: ML prediction (RF / XGBoost) + SHAP + LLM explanation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import numpy as np
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from llm_agent.config import AgentConfig
from llm_agent.feature_serializer import build_feature_matrix_row, serialize_flow_features
from llm_agent.llm_json import extract_json_object
from llm_agent.openrouter_client import OpenRouterClient


@dataclass
class MLArtifacts:
    feature_cols: List[str]
    label_encoder_classes: np.ndarray
    rf: Optional[RandomForestClassifier]
    xgb_clf: Optional[xgb.XGBClassifier]


@dataclass
class MLPrediction:
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    pred_idx: int


@dataclass
class SHAPDetail:
    feature: str
    raw_value: float
    shap_value: float


def _shap_direction_label(shap_val: float) -> str:
    if shap_val > 1e-9:
        return "increased support for the predicted class"
    if shap_val < -1e-9:
        return "decreased support for the predicted class"
    return "had negligible marginal effect"


def build_rule_based_ml_explanation(
    prediction: MLPrediction,
    shap_details: Sequence[SHAPDetail],
    raw_feature_dict: Mapping[str, float],
    *,
    top_k: int = 8,
) -> str:
    """
    Deterministic narrative from class probabilities + SHAP (no LLM).

    SHAP values are for the predicted class index (TreeExplainer convention).
    """
    lines: List[str] = []
    lines.append(
        f"The model assigns **{prediction.predicted_class}** with "
        f"{prediction.confidence:.1%} estimated probability (argmax over four classes)."
    )
    sorted_probs = sorted(
        prediction.probabilities.items(), key=lambda kv: kv[1], reverse=True
    )
    prob_bits = ", ".join(f"{lbl} {p:.1%}" for lbl, p in sorted_probs[:4])
    lines.append(f"Full class distribution: {prob_bits}.")

    lines.append(
        "The features below had the largest SHAP influence on the score for the "
        f"predicted class (**{prediction.predicted_class}**). "
        "Values shown are the original flow statistics; SHAP is computed in log1p-sign feature space."
    )
    for d in list(shap_details)[:top_k]:
        raw = raw_feature_dict.get(d.feature, float("nan"))
        direction = _shap_direction_label(d.shap_value)
        lines.append(
            f"- **{d.feature}**: raw value ≈ {raw:.6g}; SHAP = {d.shap_value:.6g} "
            f"({direction})."
        )
    return "\n".join(lines)


def build_security_takeaway_ml(
    prediction: MLPrediction, shap_details: Sequence[SHAPDetail]
) -> str:
    """One-line operational summary from ML + SHAP only."""
    top = shap_details[0] if shap_details else None
    if top is None:
        return f"ML verdict: {prediction.predicted_class} ({prediction.confidence:.0%})."
    return (
        f"ML verdict: {prediction.predicted_class} ({prediction.confidence:.0%}); "
        f"strongest driver: {top.feature} (SHAP {top.shap_value:.4g})."
    )


def shap_to_evidence_list(
    shap_details: Sequence[SHAPDetail],
    raw_feature_dict: Mapping[str, float],
    *,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Structured list for JSON `why` blocks."""
    out: List[Dict[str, Any]] = []
    for d in list(shap_details)[:top_k]:
        raw = raw_feature_dict.get(d.feature)
        out.append(
            {
                "feature": d.feature,
                "raw_flow_value": None if raw is None else float(raw),
                "feature_transform_log1p_sign": float(d.raw_value),
                "shap_for_predicted_class": float(d.shap_value),
                "effect": _shap_direction_label(d.shap_value),
            }
        )
    return out


@dataclass
class TrackBResult:
    ml_model: str
    prediction: MLPrediction
    top_shap: List[SHAPDetail]
    explanation: str
    security_takeaway: str
    raw_llm_response: str
    llm_model: str
    rule_based_summary: str = ""
    plausibility: str = ""


def load_ml_models(models_dir: Path) -> MLArtifacts:
    """Load sklearn RF, XGBoost JSON, label encoder, and feature column list."""
    models_dir = Path(models_dir)
    feature_cols: List[str] = list(joblib.load(models_dir / "feature_cols.joblib"))
    le = joblib.load(models_dir / "label_encoder.joblib")
    classes = np.asarray(le.classes_)

    rf_path = models_dir / "random_forest.joblib"
    rf: Optional[RandomForestClassifier] = None
    if rf_path.is_file():
        rf = joblib.load(rf_path)

    xgb_clf: Optional[xgb.XGBClassifier] = None
    xgb_path = models_dir / "xgboost.json"
    if xgb_path.is_file():
        xgb_clf = xgb.XGBClassifier()
        xgb_clf.load_model(str(xgb_path))

    return MLArtifacts(
        feature_cols=feature_cols,
        label_encoder_classes=classes,
        rf=rf,
        xgb_clf=xgb_clf,
    )


def predict_with_ml(
    feature_dict: Mapping[str, float],
    artifacts: MLArtifacts,
    backend: str,
) -> Tuple[MLPrediction, np.ndarray, Any]:
    """
    Return prediction details, transformed row (1, n_features), and fitted model.

    `backend` is 'rf' or 'xgboost'.
    """
    row = build_feature_matrix_row(feature_dict, artifacts.feature_cols)
    if backend == "xgboost":
        if artifacts.xgb_clf is None:
            raise FileNotFoundError("xgboost.json not found in models_dir")
        model = artifacts.xgb_clf
    else:
        if artifacts.rf is None:
            raise FileNotFoundError("random_forest.joblib not found in models_dir")
        model = artifacts.rf

    proba = model.predict_proba(row)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = str(artifacts.label_encoder_classes[pred_idx])
    probs = {str(c): float(p) for c, p in zip(artifacts.label_encoder_classes, proba)}

    return (
        MLPrediction(
            predicted_class=pred_label,
            confidence=float(proba[pred_idx]),
            probabilities=probs,
            pred_idx=pred_idx,
        ),
        row,
        model,
    )


def compute_shap_for_prediction(
    model: Any,
    X_row: np.ndarray,
    pred_idx: int,
    feature_names: Sequence[str],
    top_k: int,
) -> List[SHAPDetail]:
    """TreeExplainer SHAP values for one row, ranked for predicted class."""
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_row)

    if isinstance(shap_raw, list):
        # multiclass: list index matches class index
        contrib = np.asarray(shap_raw[pred_idx])[0]
    else:
        arr = np.asarray(shap_raw)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            contrib = arr[0, :, pred_idx]
        elif arr.ndim == 2:
            contrib = arr[0]
        else:
            contrib = np.asarray(shap_raw).reshape(-1)

    raw_row = X_row[0]
    pairs = list(zip(feature_names, raw_row, contrib))
    pairs.sort(key=lambda t: abs(float(t[2])), reverse=True)
    out: List[SHAPDetail] = []
    for name, xval, sval in pairs[:top_k]:
        # xval is log1p-sign space; report inverse for readability in explanation
        out.append(SHAPDetail(feature=str(name), raw_value=float(xval), shap_value=float(sval)))
    return out


def _top_raw_feature_lines(
    raw_lookup: Mapping[str, float],
    *,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    pairs = [(k, float(v)) for k, v in raw_lookup.items()]
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return pairs[:top_k]


def _concrete_feature_evidence_block(
    prediction: MLPrediction,
    raw_lookup: Mapping[str, float],
    *,
    top_k: int = 5,
) -> str:
    """Deterministic raw-feature evidence block (no SHAP dependency)."""
    lines = [
        f"**Flow evidence for predicted class `{prediction.predicted_class}`** "
        f"(confidence {prediction.confidence:.1%}; raw CICFlowMeter units):"
    ]
    for name, value in _top_raw_feature_lines(raw_lookup, top_k=top_k):
        lines.append(f"- **{name}**: raw ≈ {value:.6g}.")
    return "\n".join(lines)


def _mentions_top_features(text: str, raw_lookup: Mapping[str, float], *, k: int = 2) -> bool:
    if not text:
        return False
    top_names = [name for name, _ in _top_raw_feature_lines(raw_lookup, top_k=k)]
    return all(name in text for name in top_names)


def _default_plausibility_sentence(
    prediction: MLPrediction,
    raw_lookup: Mapping[str, float],
    ground_truth: Optional[str],
) -> str:
    top = _top_raw_feature_lines(raw_lookup, top_k=1)
    if not top:
        return f"Model chose **{prediction.predicted_class}** at {prediction.confidence:.1%} from flow statistics."
    fname, raw0 = top[0]
    base = (
        f"Top observed signal is **{fname}** ≈ {raw0:.6g}, and the model confidence for "
        f"**{prediction.predicted_class}** is {prediction.confidence:.1%}."
    )
    if ground_truth:
        agree = ground_truth == prediction.predicted_class
        tag = "matches" if agree else "does **not** match"
        return f"Ground truth **{ground_truth}**: prediction {tag} label. {base}"
    return base


def _default_takeaway_sentence(
    prediction: MLPrediction,
    raw_lookup: Mapping[str, float],
) -> str:
    top = _top_raw_feature_lines(raw_lookup, top_k=3)
    if not top:
        return f"Correlate alerts for **{prediction.predicted_class}** with flow-level volume and timing features."
    names = ", ".join(name for name, _ in top)
    return f"Operational check: compare **{names}** against your baseline when scoring similar flows as **{prediction.predicted_class}**."


def generate_llm_explanation(
    *,
    prediction: MLPrediction,
    raw_feature_dict: Mapping[str, float],
    feature_order: Sequence[str],
    model_name: str,
    client: OpenRouterClient,
    config: AgentConfig,
    llm_model: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> Tuple[str, str, str, str, str]:
    """
    Call LLM to narrate SHAP + probabilities.

    Returns (explanation, takeaway, plausibility, raw, model_id).
    """
    prompts_dir = Path(config.prompts_dir)
    system_t = (prompts_dir / "explainer_system.txt").read_text(encoding="utf-8")
    user_t = (prompts_dir / "explainer_user.txt").read_text(encoding="utf-8")

    if ground_truth:
        ground_truth_section = (
            "Offline evaluation — labeled ground truth for this row:\n"
            f"- True class (from dataset `Label`): **{ground_truth}**\n"
            "Say whether the model’s predicted class matches this label and whether that agreement "
            "is intuitive or surprising given the flow statistics.\n\n"
        )
    else:
        ground_truth_section = ""

    prob_lines = "\n".join(
        f"- {k}: {v:.4f}" for k, v in sorted(prediction.probabilities.items())
    )

    serialized = serialize_flow_features(
        raw_feature_dict, feature_order, mode="verbose", max_other_features=28
    )

    user_content = user_t.format(
        model_name=model_name,
        predicted_class=prediction.predicted_class,
        confidence=prediction.confidence,
        ground_truth_section=ground_truth_section,
        probability_lines=prob_lines,
        serialized_features=serialized,
    )

    messages = [
        {"role": "system", "content": system_t},
        {"role": "user", "content": user_content},
    ]
    # Keep explainer outputs short to avoid truncation (`finish_reason=length`).
    explainer_max_tokens = min(int(config.max_tokens), 700)
    res = client.chat_completion(
        messages,
        model=llm_model,
        max_tokens=explainer_max_tokens,
    )
    parsed = extract_json_object(res.content) or {}
    expl = str(parsed.get("explanation", "") or "").strip()
    take = str(parsed.get("security_takeaway", "") or "").strip()
    plaus = str(parsed.get("plausibility", "") or "").strip()
    if not expl:
        expl = (res.content or "").strip()

    appendix = _concrete_feature_evidence_block(prediction, raw_feature_dict, top_k=5)
    weak = len(expl) < 80 and len(plaus) < 40 and len(take) < 25
    if not expl:
        expl = appendix
    elif weak or not _mentions_top_features(expl, raw_feature_dict, k=2):
        expl = f"{expl.rstrip()}\n\n{appendix}".strip()
    if not plaus:
        plaus = _default_plausibility_sentence(prediction, raw_feature_dict, ground_truth)
    if not take:
        take = _default_takeaway_sentence(prediction, raw_feature_dict)
    return expl, take, plaus, res.content, res.model


def run_track_b(
    feature_dict: Mapping[str, float],
    artifacts: MLArtifacts,
    client: Optional[OpenRouterClient],
    config: AgentConfig,
    *,
    ml_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    skip_llm: bool = False,
    ground_truth: Optional[str] = None,
) -> TrackBResult:
    """End-to-end Track B for one flow dict (raw feature values)."""
    backend = (ml_backend or config.resolved_ml_backend()).lower()
    if backend in ("xgb", "xgboost"):
        backend = "xgboost"
        mname = "XGBoost"
        if artifacts.xgb_clf is None:
            raise FileNotFoundError("XGBoost model missing")
    else:
        backend = "rf"
        mname = "RandomForest"
        if artifacts.rf is None:
            raise FileNotFoundError("RandomForest model missing")

    pred, X_row, model = predict_with_ml(feature_dict, artifacts, backend)
    shap_details = compute_shap_for_prediction(
        model,
        X_row,
        pred.pred_idx,
        artifacts.feature_cols,
        top_k=config.shap_top_k,
    )
    rule_summary = build_rule_based_ml_explanation(
        pred, shap_details, feature_dict, top_k=config.shap_top_k
    )
    takeaway_ml = build_security_takeaway_ml(pred, shap_details)
    if skip_llm:
        return TrackBResult(
            ml_model=backend,
            prediction=pred,
            top_shap=shap_details,
            explanation=rule_summary,
            security_takeaway=takeaway_ml,
            raw_llm_response="",
            llm_model="none",
            rule_based_summary=rule_summary,
            plausibility="",
        )
    if client is None:
        raise ValueError(
            "OpenRouter client is required for Track B LLM explanations. "
            "Pass skip_llm=True for ML + SHAP only."
        )

    expl, take, plaus, raw, res_model = generate_llm_explanation(
        prediction=pred,
        raw_feature_dict=feature_dict,
        feature_order=artifacts.feature_cols,
        model_name=mname,
        client=client,
        config=config,
        llm_model=llm_model,
        ground_truth=ground_truth,
    )
    return TrackBResult(
        ml_model=backend,
        prediction=pred,
        top_shap=shap_details,
        explanation=expl,
        security_takeaway=take,
        raw_llm_response=raw,
        llm_model=res_model,
        rule_based_summary=rule_summary,
        plausibility=plaus,
    )
