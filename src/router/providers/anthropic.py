import os
from typing import Any

import httpx

from router.core.config import ModelConfig


class AnthropicProvider:
    async def generate(self, model: ModelConfig, request: dict[str, Any]) -> dict[str, Any]:
        api_key_env = model.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"environment variable {api_key_env} is not set")

        messages = request["messages"]
        system_messages = [
            message["content"] for message in messages if message["role"] == "system"
        ]
        api_messages = [message for message in messages if message["role"] != "system"]
        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": api_messages,
            "max_tokens": request.get("max_tokens", 1024),
        }
        if system_messages:
            payload["system"] = "\n".join(system_messages)
        for key in ("temperature", "top_p"):
            if key in request:
                payload[key] = request[key]

        base_url = (model.api_base or "https://api.anthropic.com/v1").rstrip("/")
        async with httpx.AsyncClient(timeout=model.timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/messages",
                json=payload,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
            )
            response.raise_for_status()
            result = response.json()

        usage = result.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        text = "".join(
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        )
        stop_reason = result.get("stop_reason")
        finish_reason = "length" if stop_reason == "max_tokens" else "stop"
        return {
            "id": result.get("id", "anthropic-chat-completion"),
            "object": "chat.completion",
            "created": 0,
            "model": model.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }