import re
from typing import Any

from pydantic import BaseModel


class RequestFeatures(BaseModel):
    input_tokens: int
    has_code: bool
    has_image: bool


def preprocess(messages: list[dict[str, Any]]) -> RequestFeatures:
    text_parts: list[str] = []
    has_image = False
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                if part.get("type") in {"image_url", "input_image"}:
                    has_image = True

    text = "\n".join(text_parts)
    input_tokens = max(1, (len(text) + 3) // 4)
    has_code = "```" in text or bool(
        re.search(r"(?m)^\s{4,}\S|\b(def|class|function|const|import)\s+\w+", text)
    )
    return RequestFeatures(
        input_tokens=input_tokens,
        has_code=has_code,
        has_image=has_image,
    )