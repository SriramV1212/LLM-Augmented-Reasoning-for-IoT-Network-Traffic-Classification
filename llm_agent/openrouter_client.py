"""OpenRouter-compatible chat client (OpenAI SDK base_url pattern)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from llm_agent.config import AgentConfig

logger = logging.getLogger(__name__)


def _truncate_for_log(text: str, max_chars: int) -> str:
    """Single-line preview for logs (newlines escaped)."""
    if not text:
        return ""
    t = text.replace("\r", " ").replace("\n", "\\n")
    if len(t) <= max_chars:
        return t
    return f"{t[:max_chars]}…({len(text)} chars)"


def _messages_log_summary(messages: Sequence[Dict[str, str]], preview_chars: int) -> str:
    parts: List[str] = []
    for i, msg in enumerate(messages):
        role = str(msg.get("role", "?"))
        content = str(msg.get("content", "") or "")
        prev = _truncate_for_log(content, preview_chars)
        parts.append(f"[{i}] {role} len={len(content)} «{prev}»")
    return " | ".join(parts)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class ChatResult:
    content: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: Optional[Dict[str, Any]] = None


class OpenRouterClient:
    """
    Thin wrapper around the OpenAI client pointed at OpenRouter.

    Supports retries, optional fallback models, and non-streaming chat.
    Streaming yields text deltas for optional CLI UX.
    """

    def __init__(self, config: AgentConfig):
        self._config = config
        self._client = OpenAI(
            base_url=config.openrouter_base_url,
            api_key=config.openrouter_api_key,
            timeout=config.request_timeout_s,
        )
        self._total_usage = TokenUsage()

    @property
    def cumulative_usage(self) -> TokenUsage:
        return self._total_usage

    def chat_completion(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> ChatResult:
        """Single non-streaming completion with fallbacks on transient errors."""
        primary = model or self._config.default_model
        chain = [primary, *self._config.fallback_models]
        seen: set[str] = set()
        models_to_try = [m for m in chain if not (m in seen or seen.add(m))]

        last_err: Optional[Exception] = None
        preview_chars = 1600 if logger.isEnabledFor(logging.DEBUG) else 400
        for m in models_to_try:
            for attempt in range(self._config.max_retries):
                try:
                    headers = dict(extra_headers or {})
                    if self._config.site_url:
                        headers.setdefault("HTTP-Referer", self._config.site_url)
                    if self._config.site_name:
                        headers.setdefault("X-Title", self._config.site_name)

                    temp = temperature if temperature is not None else self._config.temperature
                    mtoks = max_tokens if max_tokens is not None else self._config.max_tokens
                    url = (self._config.openrouter_base_url or "").rstrip("/") + "/chat/completions"
                    logger.info(
                        "OpenRouter request: POST %s | model=%s attempt=%s/%s | temperature=%s max_tokens=%s | messages: %s",
                        url,
                        m,
                        attempt + 1,
                        self._config.max_retries,
                        temp,
                        mtoks,
                        _messages_log_summary(messages, preview_chars),
                    )
                    t0 = time.perf_counter()
                    resp = self._client.chat.completions.create(
                        model=m,
                        messages=list(messages),  # type: ignore[arg-type]
                        temperature=temp,
                        max_tokens=mtoks,
                        extra_headers=headers or None,
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    choice = resp.choices[0]
                    text = (choice.message.content or "").strip()
                    usage = TokenUsage()
                    if resp.usage:
                        usage.prompt_tokens = resp.usage.prompt_tokens or 0
                        usage.completion_tokens = resp.usage.completion_tokens or 0
                        usage.total_tokens = resp.usage.total_tokens or (
                            usage.prompt_tokens + usage.completion_tokens
                        )
                    self._total_usage.add(usage)
                    out_model = resp.model or m
                    logger.info(
                        "OpenRouter response: model=%s | %.0f ms | tokens in/out/total=%s/%s/%s | finish_reason=%s | content_preview=%s",
                        out_model,
                        elapsed_ms,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        getattr(choice, "finish_reason", None),
                        _truncate_for_log(text, preview_chars),
                    )
                    return ChatResult(
                        content=text,
                        model=out_model,
                        usage=usage,
                        raw=resp.model_dump() if hasattr(resp, "model_dump") else None,
                    )
                except (RateLimitError, APITimeoutError, APIError, TimeoutError) as e:
                    last_err = e
                    wait = self._config.retry_backoff_s * (2**attempt)
                    logger.warning("OpenRouter error model=%s attempt=%s: %s", m, attempt + 1, e)
                    time.sleep(wait)
                except Exception as e:  # noqa: BLE001 — surface after fallbacks
                    last_err = e
                    logger.exception("Unexpected OpenRouter failure on model=%s", m)
                    break
        raise RuntimeError(f"OpenRouter chat failed after fallbacks: {last_err}") from last_err

    def chat_completion_stream(
        self,
        messages: Sequence[Dict[str, str]],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Iterator[str]:
        """Stream completion text chunks (best-effort token usage not aggregated)."""
        m = model or self._config.default_model
        headers = dict(extra_headers or {})
        if self._config.site_url:
            headers.setdefault("HTTP-Referer", self._config.site_url)
        if self._config.site_name:
            headers.setdefault("X-Title", self._config.site_name)

        preview_chars = 1600 if logger.isEnabledFor(logging.DEBUG) else 400
        url = (self._config.openrouter_base_url or "").rstrip("/") + "/chat/completions"
        logger.info(
            "OpenRouter stream request: POST %s | model=%s stream=True | messages: %s",
            url,
            m,
            _messages_log_summary(messages, preview_chars),
        )
        t0 = time.perf_counter()
        stream = self._client.chat.completions.create(
            model=m,
            messages=list(messages),  # type: ignore[arg-type]
            temperature=temperature if temperature is not None else self._config.temperature,
            max_tokens=max_tokens if max_tokens is not None else self._config.max_tokens,
            stream=True,
            extra_headers=headers or None,
        )
        chunks = 0
        for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            if delta and delta.content:
                chunks += 1
                yield delta.content
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "OpenRouter stream response: finished in %.0f ms (text chunks yielded: %s)",
            elapsed_ms,
            chunks,
        )

    def list_models(self) -> List[str]:
        """Return model ids from OpenRouter (requires API access)."""
        base = (self._config.openrouter_base_url or "").rstrip("/")
        logger.info("OpenRouter request: GET %s/models", base)
        t0 = time.perf_counter()
        data = self._client.models.list()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ids = sorted({m.id for m in data.data})
        logger.info(
            "OpenRouter response: models list | %.0f ms | count=%s",
            elapsed_ms,
            len(ids),
        )
        return ids
