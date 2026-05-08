"""Track A: zero-shot LLM classification with chain-of-thought JSON output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from llm_agent.config import AgentConfig
from llm_agent.feature_evidence import flow_evidence_markdown, mentions_top_feature_names, top_raw_pairs
from llm_agent.feature_serializer import serialize_flow_features
from llm_agent.llm_json import extract_json_object
from llm_agent.openrouter_client import OpenRouterClient

VALID_CLASSES = ("BENIGN", "DDOS_UDP", "DOS_TCP", "RECON")

# Zero-shot answers include reasoning_steps; cap completion tokens to reduce truncation.
_ZERO_SHOT_COMPLETION_CAP = 720


@dataclass
class ZeroShotResult:
    predicted_class: str
    confidence: float
    reasoning_steps: List[str]
    short_rationale: str
    plausibility: str
    raw_response: str
    model: str
    parsed: Dict[str, Any]


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


def _default_plausibility_zero_shot(
    predicted_class: str,
    confidence: float,
    raw_present: Mapping[str, float],
    ground_truth: Optional[str],
) -> str:
    top = top_raw_pairs(raw_present, top_k=1)
    if not top:
        base = f"Zero-shot label **{predicted_class}** at {confidence:.1%} from flow statistics."
    else:
        fname, raw0 = top[0]
        base = (
            f"Dominant numeric signal **{fname}** ≈ {raw0:.6g} aligns with **{predicted_class}** "
            f"at {confidence:.1%} confidence."
        )
    if ground_truth:
        agree = ground_truth == predicted_class
        tag = "matches" if agree else "does **not** match"
        return f"Ground truth **{ground_truth}**: prediction {tag} label. {base}"
    return base


def classify_flow_zero_shot(
    feature_dict: Mapping[str, float],
    feature_order: Sequence[str],
    client: OpenRouterClient,
    config: AgentConfig,
    *,
    model: Optional[str] = None,
    serializer_mode: str = "verbose",
    ground_truth: Optional[str] = None,
    serializer_max_other: int = 28,
) -> ZeroShotResult:
    """Run Track A: serialize features, call LLM, parse JSON classification."""
    prompts_dir = Path(config.prompts_dir)
    system_t = _read_prompt(prompts_dir / "zero_shot_system.txt")
    user_t = _read_prompt(prompts_dir / "zero_shot_user.txt")

    serialized = serialize_flow_features(
        feature_dict,
        feature_order,
        mode=serializer_mode,
        max_other_features=serializer_max_other,
    )

    if ground_truth:
        ground_truth_section = (
            "Offline evaluation — labeled ground truth for this row:\n"
            f"- True class (from dataset `Label`): **{ground_truth}**\n"
            "After you choose `predicted_class`, say whether it agrees with this label and whether "
            "that agreement is intuitive or surprising given the flow statistics.\n\n"
        )
    else:
        ground_truth_section = ""

    user_content = user_t.format(
        ground_truth_section=ground_truth_section,
        serialized_features=serialized,
    )

    messages = [
        {"role": "system", "content": system_t},
        {"role": "user", "content": user_content},
    ]
    zs_max_tokens = min(int(config.max_tokens), _ZERO_SHOT_COMPLETION_CAP)
    res = client.chat_completion(messages, model=model, max_tokens=zs_max_tokens)
    parsed = extract_json_object(res.content) or {}

    pred = str(parsed.get("predicted_class", "")).strip()
    if pred not in VALID_CLASSES:
        upper = pred.upper().replace(" ", "_")
        alias = {
            "DDOS_UDP_FLOOD": "DDOS_UDP",
            "DOS_TCP_FLOOD": "DOS_TCP",
            "RECONNAISSANCE": "RECON",
            "RECONNAISSANCE_PORT_SCAN": "RECON",
        }
        pred = alias.get(upper, pred)

    conf_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    steps = parsed.get("reasoning_steps") or []
    if not isinstance(steps, list):
        steps = [str(steps)]
    else:
        steps = [str(s) for s in steps]

    rationale = str(parsed.get("short_rationale", "") or "").strip()
    plaus = str(parsed.get("plausibility", "") or "").strip()

    final_pred = pred if pred in VALID_CLASSES else "UNKNOWN"
    present = {k: float(feature_dict.get(k, 0.0)) for k in feature_order}

    combined_text = " ".join(steps) + " " + rationale + " " + plaus
    appendix = flow_evidence_markdown(final_pred, confidence, present, top_k=5)
    weak = len(rationale) < 70 and len(plaus) < 40 and len(steps) < 3
    if not rationale:
        rationale = appendix
    elif weak or not mentions_top_feature_names(combined_text, present, k=2):
        rationale = f"{rationale.rstrip()}\n\n{appendix}".strip()

    if not plaus:
        plaus = _default_plausibility_zero_shot(final_pred, confidence, present, ground_truth)

    return ZeroShotResult(
        predicted_class=final_pred,
        confidence=confidence,
        reasoning_steps=steps,
        short_rationale=rationale,
        plausibility=plaus,
        raw_response=res.content,
        model=res.model,
        parsed=parsed,
    )
