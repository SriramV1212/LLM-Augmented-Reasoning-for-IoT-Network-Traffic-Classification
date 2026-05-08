"""Batch evaluation: ML+agent metrics or zero-shot LLM metrics (one mode per run)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from llm_agent.config import AgentConfig, ClassifyMode
from llm_agent.merger import merge_results
from llm_agent.openrouter_client import OpenRouterClient
from llm_agent.track_a_zero_shot import classify_flow_zero_shot
from llm_agent.track_b_ml_explainer import load_ml_models, predict_with_ml


@dataclass
class EvalSummary:
    """Metrics for one evaluation mode (either ML or zero-shot LLM)."""

    classification_mode: str
    n_rows: int
    ml_accuracy: Optional[float]
    ml_macro_f1: Optional[float]
    ml_classification_report: Optional[Dict[str, Any]]
    ml_confusion_matrix: Optional[List[List[int]]]
    zeroshot_accuracy: Optional[float]
    zeroshot_macro_f1: Optional[float]
    zeroshot_classification_report: Optional[Dict[str, Any]]
    zeroshot_confusion_matrix: Optional[List[List[int]]]
    labels: List[str]
    # ml_agent: optional LLM narrative samples (first K rows; metrics use all rows)
    ml_agent_llm_explainer_samples: Optional[List[Dict[str, Any]]] = None
    ml_agent_llm_explainer_note: Optional[str] = None
    zeroshot_llm_explainer_samples: Optional[List[Dict[str, Any]]] = None
    zeroshot_llm_explainer_note: Optional[str] = None

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_csv(
    csv_path: Path,
    config: AgentConfig,
    client: Optional[OpenRouterClient],
    *,
    max_rows: Optional[int] = None,
    random_state: int = 43,
    classification_mode: ClassifyMode = "ml_agent",
    ml_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_explainer: bool = True,
    explain_max_rows: int = 25,
    store_eval_samples: bool = True,
) -> EvalSummary:
    """
    Evaluate on a CSV containing all `feature_cols` plus a `Label` column.

    - ``ml_agent``: ML predictions vs labels on **all** rows. By default, also runs the
      **OpenRouter LLM explainer** on the first ``explain_max_rows`` rows only
      (set ``llm_explainer=False`` to skip API calls for narratives).
    - ``zeroshot``: one OpenRouter call per row — slow and costly. Optionally keeps narrative samples for the first ``explain_max_rows`` rows (when ``store_eval_samples``).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    artifacts = load_ml_models(Path(config.models_dir))
    df = pd.read_csv(csv_path)
    if "Label" not in df.columns:
        raise ValueError("CSV must contain a 'Label' column with ground truth.")

    missing = [c for c in artifacts.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing {len(missing)} feature columns (showing up to 5): {missing[:5]}")

    if max_rows is not None and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=random_state)

    y_true = df["Label"].astype(str).tolist()
    rows = df[artifacts.feature_cols].to_dict(orient="records")

    backend = (ml_backend or config.resolved_ml_backend()).lower()
    if backend in ("xgb", "xgboost"):
        backend = "xgboost"
    else:
        backend = "rf"

    if classification_mode == "ml_agent":
        from llm_agent.track_b_ml_explainer import run_track_b

        preds: List[str] = []
        for rec in rows:
            pred_b, _, _ = predict_with_ml(rec, artifacts, backend)
            preds.append(pred_b.predicted_class)
        labels_sorted = sorted(set(y_true) | set(preds))
        acc = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, average="macro", labels=labels_sorted, zero_division=0))
        report = classification_report(
            y_true, preds, labels=labels_sorted, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true, preds, labels=labels_sorted).tolist()

        expl_samples: Optional[List[Dict[str, Any]]] = None
        expl_note: Optional[str] = None
        if llm_explainer:
            if client is None:
                expl_note = (
                    "LLM explainer not run: no OpenRouter client (set OPENROUTER_API_KEY or pass "
                    "--no-llm-explainer for metrics-only)."
                )
            else:
                limit = min(len(rows), max(0, explain_max_rows))
                if limit == 0:
                    expl_note = "LLM explainer skipped: explain_max_rows is 0."
                else:
                    expl_samples = []
                    for i in range(limit):
                        rec = rows[i]
                        tb = run_track_b(
                            rec,
                            artifacts,
                            client,
                            config,
                            ml_backend=ml_backend,
                            llm_model=llm_model,
                            skip_llm=False,
                            ground_truth=y_true[i],
                        )
                        expl_samples.append(
                            {
                                "row_index": i,
                                "true_label": y_true[i],
                                "ml_prediction": preds[i],
                                "llm_model": tb.llm_model,
                                "plausibility": tb.plausibility,
                                "explanation": tb.explanation,
                                "security_takeaway": tb.security_takeaway,
                                "rule_based_summary": tb.rule_based_summary,
                            }
                        )
                    expl_note = (
                        f"OpenRouter LLM explainer on first {limit} row(s); "
                        f"metrics use all {len(rows)} rows."
                    )

        return EvalSummary(
            classification_mode="ml_agent",
            n_rows=len(rows),
            ml_accuracy=acc,
            ml_macro_f1=f1,
            ml_classification_report=report,
            ml_confusion_matrix=cm,
            zeroshot_accuracy=None,
            zeroshot_macro_f1=None,
            zeroshot_classification_report=None,
            zeroshot_confusion_matrix=None,
            labels=labels_sorted,
            ml_agent_llm_explainer_samples=expl_samples,
            ml_agent_llm_explainer_note=expl_note,
            zeroshot_llm_explainer_samples=None,
            zeroshot_llm_explainer_note=None,
        )

    # zeroshot
    if client is None:
        raise ValueError("zeroshot evaluation requires an OpenRouterClient")

    sample_limit = 0
    if store_eval_samples:
        sample_limit = min(len(rows), max(0, explain_max_rows))

    zs_samples: Optional[List[Dict[str, Any]]] = None
    zs_note: Optional[str] = None
    if not store_eval_samples:
        zs_note = (
            "Zero-shot narrative samples omitted (store_eval_samples=False / CLI --no-llm-explainer)."
        )
    elif sample_limit == 0:
        zs_note = "Zero-shot narrative samples skipped: explain_max_rows is 0."
    else:
        zs_samples = []

    preds_z: List[str] = []
    for i, rec in enumerate(rows):
        gt_ctx = y_true[i] if zs_samples is not None and i < sample_limit else None
        zs_one = classify_flow_zero_shot(
            rec,
            artifacts.feature_cols,
            client,
            config,
            model=llm_model,
            ground_truth=gt_ctx,
        )
        preds_z.append(zs_one.predicted_class)
        if zs_samples is not None and i < sample_limit:
            zs_samples.append(
                {
                    "row_index": i,
                    "true_label": y_true[i],
                    "zeroshot_prediction": zs_one.predicted_class,
                    "confidence": zs_one.confidence,
                    "llm_model": zs_one.model,
                    "plausibility": zs_one.plausibility,
                    "reasoning_steps": zs_one.reasoning_steps,
                    "short_rationale": zs_one.short_rationale,
                }
            )

    if zs_samples is not None and zs_note is None:
        zs_note = (
            f"Zero-shot LLM narratives on first {sample_limit} row(s); "
            f"metrics use all {len(rows)} rows."
        )

    labels_sorted = sorted(set(y_true) | set(preds_z))
    acc_z = float(accuracy_score(y_true, preds_z))
    f1_z = float(f1_score(y_true, preds_z, average="macro", labels=labels_sorted, zero_division=0))
    report_z = classification_report(
        y_true, preds_z, labels=labels_sorted, output_dict=True, zero_division=0
    )
    cm_z = confusion_matrix(y_true, preds_z, labels=labels_sorted).tolist()
    return EvalSummary(
        classification_mode="zeroshot",
        n_rows=len(rows),
        ml_accuracy=None,
        ml_macro_f1=None,
        ml_classification_report=None,
        ml_confusion_matrix=None,
        zeroshot_accuracy=acc_z,
        zeroshot_macro_f1=f1_z,
        zeroshot_classification_report=report_z,
        zeroshot_confusion_matrix=cm_z,
        labels=labels_sorted,
        ml_agent_llm_explainer_samples=None,
        ml_agent_llm_explainer_note=None,
        zeroshot_llm_explainer_samples=zs_samples,
        zeroshot_llm_explainer_note=zs_note,
    )


def write_eval_artifacts(summary: EvalSummary, out_json: Path, out_md: Path) -> None:
    """Write JSON report and short markdown summary."""
    out_json = Path(out_json)
    out_md = Path(out_md)
    out_json.write_text(json.dumps(summary.to_json_dict(), indent=2), encoding="utf-8")
    lines = [
        "# Evaluation summary",
        "",
        f"- Mode: **{summary.classification_mode}**",
        f"- Rows: **{summary.n_rows}**",
    ]
    if summary.classification_mode == "ml_agent" and summary.ml_accuracy is not None:
        lines.append(f"- ML accuracy: **{summary.ml_accuracy:.4f}**")
        lines.append(f"- ML macro-F1: **{summary.ml_macro_f1:.4f}**")
    if summary.classification_mode == "zeroshot" and summary.zeroshot_accuracy is not None:
        lines.append(f"- Zero-shot accuracy: **{summary.zeroshot_accuracy:.4f}**")
        lines.append(f"- Zero-shot macro-F1: **{summary.zeroshot_macro_f1:.4f}**")
    if summary.ml_agent_llm_explainer_note:
        lines.append(f"- LLM explainer: {summary.ml_agent_llm_explainer_note}")
    if summary.ml_agent_llm_explainer_samples:
        lines.append(
            f"- LLM explainer samples: **{len(summary.ml_agent_llm_explainer_samples)}** (see JSON)"
        )
    if summary.zeroshot_llm_explainer_note:
        lines.append(f"- Zero-shot narratives: {summary.zeroshot_llm_explainer_note}")
    if summary.zeroshot_llm_explainer_samples:
        lines.append(
            f"- Zero-shot narrative samples: **{len(summary.zeroshot_llm_explainer_samples)}** (see JSON)"
        )
    lines.extend(["", "## Labels (evaluation order)", "", ", ".join(summary.labels)])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_flow_pair_for_debug(
    feature_dict: dict,
    config: AgentConfig,
    client: OpenRouterClient,
    *,
    classification_mode: ClassifyMode = "ml_agent",
    ml_backend: Optional[str] = None,
    llm_model: Optional[str] = None,
    flow_id: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> dict:
    """Run one classification path + report (for debugging)."""
    from llm_agent.track_b_ml_explainer import run_track_b

    artifacts = load_ml_models(Path(config.models_dir))
    if classification_mode == "zeroshot":
        zs = classify_flow_zero_shot(
            feature_dict,
            artifacts.feature_cols,
            client,
            config,
            model=llm_model,
            ground_truth=ground_truth,
        )
        merged = merge_results(
            flow_id=flow_id,
            classification_mode="zeroshot",
            zs=zs,
            tb=None,
            config=config,
            raw_features=feature_dict,
        )
    else:
        tb = run_track_b(
            feature_dict,
            artifacts,
            client,
            config,
            ml_backend=ml_backend,
            llm_model=llm_model,
            ground_truth=ground_truth,
        )
        merged = merge_results(
            flow_id=flow_id,
            classification_mode="ml_agent",
            zs=None,
            tb=tb,
            config=config,
            raw_features=feature_dict,
        )
    return merged.to_json_dict()
