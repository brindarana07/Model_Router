import os
from typing import Any

import httpx

from router.core.config import ModelConfig


class OpenAIProvider:
    async def generate(self, model: ModelConfig, request: dict[str, Any]) -> dict[str, Any]:
        base_url = (model.api_base or "https://api.openai.com/v1").rstrip("/")
        headers: dict[str, str] = {}
        if model.api_key_env:
            api_key = os.getenv(model.api_key_env)
            if not api_key:
                raise RuntimeError(f"environment variable {model.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {**request, "model": model.model_id, "stream": False}
        async with httpx.AsyncClient(timeout=model.timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()