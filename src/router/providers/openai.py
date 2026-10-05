import os
from typing import Any

import httpx

from router.core.config import ModelConfig

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "groq": "https://api.groq.com/openai/v1",
}

DEFAULT_API_KEY_ENVS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


class OpenAIProvider:
    async def generate(self, model: ModelConfig, request: dict[str, Any]) -> dict[str, Any]:
        base_url = (model.api_base or DEFAULT_BASE_URLS[model.provider]).rstrip("/")
        headers: dict[str, str] = {}
        api_key_env = model.api_key_env or DEFAULT_API_KEY_ENVS.get(model.provider)
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"environment variable {api_key_env} is not set")
            headers["Authorization"] = "Bearer " + api_key

        payload = {**request, "model": model.model_id, "stream": False}
        async with httpx.AsyncClient(timeout=model.timeout_seconds) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("upstream response must be a JSON object")
            return {str(key): value for key, value in result.items()}
