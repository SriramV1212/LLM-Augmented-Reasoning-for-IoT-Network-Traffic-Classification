"""Track A: zero-shot LLM classification with chain-of-thought JSON output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from llm_agent.config import AgentConfig
from llm_agent.feature_serializer import serialize_flow_features
from llm_agent.llm_json import extract_json_object
from llm_agent.openrouter_client import OpenRouterClient

VALID_CLASSES = ("BENIGN", "DDOS_UDP", "DOS_TCP", "RECON")


@dataclass
class ZeroShotResult:
    predicted_class: str
    confidence: float
    reasoning_steps: List[str]
    short_rationale: str
    raw_response: str
    model: str
    parsed: Dict[str, Any]


def _read_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8")


def classify_flow_zero_shot(
    feature_dict: Mapping[str, float],
    feature_order: Sequence[str],
    client: OpenRouterClient,
    config: AgentConfig,
    *,
    model: Optional[str] = None,
    serializer_mode: str = "verbose",
) -> ZeroShotResult:
    """Run Track A: serialize features, call LLM, parse JSON classification."""
    prompts_dir = Path(config.prompts_dir)
    system_t = _read_prompt(prompts_dir / "zero_shot_system.txt")
    user_t = _read_prompt(prompts_dir / "zero_shot_user.txt")

    serialized = serialize_flow_features(
        feature_dict, feature_order, mode=serializer_mode
    )
    user_content = user_t.format(serialized_features=serialized)

    messages = [
        {"role": "system", "content": system_t},
        {"role": "user", "content": user_content},
    ]
    res = client.chat_completion(messages, model=model)
    parsed = extract_json_object(res.content) or {}

    pred = str(parsed.get("predicted_class", "")).strip()
    if pred not in VALID_CLASSES:
        # tolerate minor formatting
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

    rationale = str(parsed.get("short_rationale", "")).strip()

    return ZeroShotResult(
        predicted_class=pred if pred in VALID_CLASSES else "UNKNOWN",
        confidence=confidence,
        reasoning_steps=steps,
        short_rationale=rationale,
        raw_response=res.content,
        model=res.model,
        parsed=parsed,
    )
