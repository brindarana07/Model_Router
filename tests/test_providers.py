from typing import Any

import httpx
import pytest

from router.api.app import _default_provider
from router.core.config import ModelConfig, RouterConfig
from router.providers.openai import OpenAIProvider


@pytest.mark.parametrize(
    ("provider", "key_env", "base_url"),
    [
        (
            "gemini",
            "GEMINI_API_KEY",
            "https://generativelanguage.googleapis.com/v1beta/openai",
        ),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    ],
)
async def test_openai_compatible_provider_uses_default_url_and_auth(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_env: str,
    base_url: str,
) -> None:
    observed: dict[str, Any] = {}

    class MockAsyncClient:
        def __init__(self, timeout: float) -> None:
            observed["timeout"] = timeout

        async def __aenter__(self) -> "MockAsyncClient":
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def post(
            self,
            url: str,
            *,
            json: dict[str, Any],
            headers: dict[str, str],
        ) -> httpx.Response:
            observed.update(url=url, payload=json, headers=headers)
            return httpx.Response(200, json={"choices": []}, request=httpx.Request("POST", url))

    monkeypatch.setenv(key_env, "test-api-key")
    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    model = ModelConfig(
        name=provider,
        provider=provider,
        model_id="test-model",
        max_context=1000,
    )

    assert isinstance(_default_provider(model), OpenAIProvider)
    await OpenAIProvider().generate(model, {"messages": [{"role": "user", "content": "hi"}]})

    assert observed["url"] == f"{base_url}/chat/completions"
    assert observed["headers"] == {"Authorization": "Bearer test-api-key"}
    assert observed["payload"]["model"] == "test-model"


@pytest.mark.parametrize(
    ("provider", "key_env"),
    [("gemini", "GEMINI_API_KEY"), ("groq", "GROQ_API_KEY")],
)
async def test_openai_compatible_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    key_env: str,
) -> None:
    monkeypatch.delenv(key_env, raising=False)
    model = ModelConfig(
        name=provider,
        provider=provider,
        model_id="test-model",
        max_context=1000,
    )

    with pytest.raises(RuntimeError, match=f"{key_env} is not set"):
        await OpenAIProvider().generate(model, {"messages": []})


def test_example_config_includes_gemini_and_groq() -> None:
    config = RouterConfig.load("config/models.example.yaml")

    assert config.model_by_name("gemini-flash").provider == "gemini"
    assert config.model_by_name("groq-fast").provider == "groq"
