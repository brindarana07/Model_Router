import re
from typing import Any

from pydantic import BaseModel


class RequestFeatures(BaseModel):
    input_tokens: int
    has_code: bool
    has_image: bool
    query_text: str = ""


def preprocess(messages: list[dict[str, Any]]) -> RequestFeatures:
    text_parts: list[str] = []
    query_parts: list[str] = []
    has_image = False
    for message in messages:
        content = message.get("content", "")
        message_text_parts: list[str] = []
        if isinstance(content, str):
            message_text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    message_text_parts.append(part["text"])
                if part.get("type") in {"image_url", "input_image"}:
                    has_image = True
        text_parts.extend(message_text_parts)
        if message.get("role") == "user":
            query_parts.extend(message_text_parts)

    text = "\n".join(text_parts)
    query_text = "\n".join(query_parts)
    input_tokens = max(1, (len(text) + 3) // 4)
    has_code = "```" in text or bool(
        re.search(r"(?m)^\s{4,}\S|\b(def|class|function|const|import)\s+\w+", text)
    )
    return RequestFeatures(
        input_tokens=input_tokens,
        has_code=has_code,
        has_image=has_image,
        query_text=query_text,
    )
