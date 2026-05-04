"""Configuration for the LLM traffic agent (OpenRouter + ML paths)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

from dotenv import load_dotenv

ClassifyMode = Literal["ml_agent", "zeroshot"]
MLBackend = Literal["rf", "xgb", "xgboost"]


def _default_fallback_models() -> List[str]:
    return [
        "openai/gpt-4o-mini",
        "meta-llama/llama-3.1-70b-instruct",
    ]


def _env_classify_mode() -> str:
    """Resolve LLM_AGENT_CLASSIFY_MODE with legacy LLM_AGENT_TRACK_MODE fallback."""
    raw = os.environ.get("LLM_AGENT_CLASSIFY_MODE", "").strip().lower()
    if raw in ("ml_agent", "ml-agent", "ml+agent"):
        return "ml_agent"
    if raw in ("zeroshot", "zero-shot", "llm"):
        return "zeroshot"
    legacy = os.environ.get("LLM_AGENT_TRACK_MODE", "").strip().lower()
    if legacy in ("a", "track_a", "zero-shot", "zeroshot"):
        return "zeroshot"
    if legacy in ("b", "track_b", "both", "all", ""):
        return "ml_agent"
    return "ml_agent"


@dataclass
class AgentConfig:
    """Runtime configuration loaded from env + optional overrides."""

    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    default_model: str = "tencent/hy3-preview:free"
    fallback_models: List[str] = field(default_factory=_default_fallback_models)
    ml_classifier: MLBackend = "rf"
    classification_mode: ClassifyMode = "ml_agent"
    temperature: float = 0.1
    max_tokens: int = 2048
    request_timeout_s: float = 120.0
    max_retries: int = 3
    retry_backoff_s: float = 1.5
    models_dir: Path = field(default_factory=lambda: Path("models"))
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    shap_top_k: int = 10
    site_url: Optional[str] = None  # optional OpenRouter attribution
    site_name: Optional[str] = None

    def resolved_ml_backend(self) -> str:
        if self.ml_classifier in ("xgb", "xgboost"):
            return "xgboost"
        return "rf"


def load_config(
    *,
    env_file: Optional[Path | str] = None,
    require_api_key: bool = True,
    **overrides: object,
) -> AgentConfig:
    """
    Load `.env` if present, then build AgentConfig.

    Environment variables:
      OPENROUTER_API_KEY (required)
      OPENROUTER_BASE_URL (optional)
      OPENROUTER_DEFAULT_MODEL
      OPENROUTER_FALLBACK_MODELS (comma-separated)
      LLM_AGENT_ML_CLASSIFIER (rf | xgb)
      LLM_AGENT_CLASSIFY_MODE (ml_agent | zeroshot) — default ml_agent
      LLM_AGENT_MODELS_DIR
      LLM_AGENT_PROMPTS_DIR
    """
    if env_file is not None:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv(override=False)

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    key_from_override = overrides.get("openrouter_api_key")
    resolved_key = (
        str(key_from_override).strip()
        if key_from_override is not None
        else api_key
    )

    if require_api_key and not resolved_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    fallbacks_raw = os.environ.get("OPENROUTER_FALLBACK_MODELS", "")
    fallback_models = (
        [m.strip() for m in fallbacks_raw.split(",") if m.strip()]
        if fallbacks_raw
        else _default_fallback_models()
    )

    models_dir = Path(os.environ.get("LLM_AGENT_MODELS_DIR", "models"))
    prompts_dir = Path(os.environ.get("LLM_AGENT_PROMPTS_DIR", "prompts"))

    ml_c = (
        overrides["ml_classifier"]
        if "ml_classifier" in overrides
        else os.environ.get("LLM_AGENT_ML_CLASSIFIER", "rf").lower()
    )
    mode_raw = (
        overrides["classification_mode"]
        if "classification_mode" in overrides
        else _env_classify_mode()
    )
    mode_s = str(mode_raw).strip().lower().replace("-", "_")
    if mode_s in ("ml_agent", "mlagent"):
        cls_mode: ClassifyMode = "ml_agent"
    elif mode_s in ("zeroshot", "zero_shot", "llm"):
        cls_mode = "zeroshot"
    else:
        cls_mode = "ml_agent"

    cfg = AgentConfig(
        openrouter_api_key=resolved_key,
        openrouter_base_url=os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        default_model=os.environ.get(
            "OPENROUTER_DEFAULT_MODEL", "tencent/hy3-preview:free"
        ),
        fallback_models=fallback_models,
        ml_classifier=ml_c,  # type: ignore[arg-type]
        classification_mode=cls_mode,
        models_dir=models_dir,
        prompts_dir=prompts_dir,
    )

    for k, v in overrides.items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)

    return cfg
