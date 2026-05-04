"""Extract JSON objects from LLM outputs (markdown fences tolerated)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the first JSON object in `text`.

    Handles optional ```json ... ``` fences and trailing prose.
    """
    if not text:
        return None
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = s[start : i + 1]
                try:
                    return json.loads(chunk)
                except json.JSONDecodeError:
                    return None
    return None
