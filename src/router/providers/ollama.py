from typing import Any

import httpx

from router.core.config import ModelConfig


class OllamaProvider:
    async def generate(self, model: ModelConfig, request: dict[str, Any]) -> dict[str, Any]:
        base_url = (model.api_base or "http://localhost:11434").rstrip("/")
        payload: dict[str, Any] = {
            "model": model.model_id,
            "messages": request["messages"],
            "stream": False,
            "options": {
                key: request[key]
                for key in ("temperature", "top_p", "seed")
                if key in request
            },
        }
        if "max_tokens" in request:
            payload["options"]["num_predict"] = request["max_tokens"]

        async with httpx.AsyncClient(timeout=model.timeout_seconds) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            result = response.json()

        prompt_tokens = result.get("prompt_eval_count", 0)
        completion_tokens = result.get("eval_count", 0)
        return {
            "id": result.get("id", "ollama-chat-completion"),
            "object": "chat.completion",
            "created": 0,
            "model": model.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": result.get("message", {"role": "assistant", "content": ""}),
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }