from typing import Any

from fastapi.testclient import TestClient

from router.api.app import create_app
from router.core.config import ModelConfig, RouterConfig


def sample_config() -> RouterConfig:
    return RouterConfig.model_validate(
        {
            "models": [
                {
                    "name": "cheap",
                    "provider": "openai",
                    "model_id": "small",
                    "max_context": 1000,
                },
                {
                    "name": "strong",
                    "provider": "anthropic",
                    "model_id": "large",
                    "max_context": 2000,
                },
            ],
            "routing": {
                "default": "cheap",
                "fallbacks": ["strong"],
                "rules": [{"if": {"has_code": True}, "use": "strong"}],
            },
        }
    )


class FakeProvider:
    def __init__(self, model_name: str, calls: list[str]) -> None:
        self.model_name = model_name
        self.calls = calls

    async def generate(self, model: ModelConfig, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(self.model_name)
        if self.model_name == "strong":
            raise TimeoutError("upstream timed out")
        return {
            "id": "test-response",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }


def test_endpoint_falls_back_and_returns_openai_shape() -> None:
    calls: list[str] = []
    app = create_app(sample_config(), lambda model: FakeProvider(model.name, calls))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "```py\nprint(1)```"}]},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "cheap"
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert calls == ["strong", "cheap"]


def test_streaming_is_rejected_until_implemented() -> None:
    app = create_app(sample_config(), lambda model: FakeProvider(model.name, []))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"stream": True, "messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 400


def test_context_limit_skips_model_and_uses_fallback() -> None:
    config = sample_config()
    config.model_by_name("strong").max_context = 1
    calls: list[str] = []
    app = create_app(config, lambda model: FakeProvider(model.name, calls))

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "```py\nprint(1)```"}]},
        )

    assert response.status_code == 200
    assert response.json()["model"] == "cheap"
    assert calls == ["cheap"]