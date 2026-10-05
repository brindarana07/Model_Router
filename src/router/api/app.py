import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from router.core.config import ModelConfig, RouterConfig
from router.core.preprocessor import preprocess
from router.core.routing import choose_route
from router.providers.anthropic import AnthropicProvider
from router.providers.base import ChatProvider
from router.providers.ollama import OllamaProvider
from router.providers.openai import OpenAIProvider

logger = logging.getLogger("router.decisions")
load_dotenv()


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False


class RouterApplication(FastAPI):
    router_config: RouterConfig
    providers: dict[str, ChatProvider]


def _default_provider(model: ModelConfig) -> ChatProvider:
    if model.provider in {"openai", "gemini", "groq"}:
        return OpenAIProvider()
    if model.provider == "anthropic":
        return AnthropicProvider()
    if model.provider == "ollama":
        return OllamaProvider()
    raise ValueError(f"unsupported provider: {model.provider}")


def _cost(model: ModelConfig, usage: dict[str, Any]) -> float:
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    return (
        input_tokens * model.cost_per_1k_input + output_tokens * model.cost_per_1k_output
    ) / 1000


def create_app(
    config: RouterConfig | None = None,
    provider_factory: Callable[[ModelConfig], ChatProvider] = _default_provider,
) -> RouterApplication:
    if config is None:
        config = RouterConfig.load(
            os.getenv("MODEL_ROUTER_CONFIG", "config/models.example.yaml")
        )

    app = RouterApplication(title="Model Router", version="0.1.0")
    app.router_config = config
    app.providers = {model.name: provider_factory(model) for model in config.models}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {"id": model.name, "object": "model", "owned_by": "model-router"}
                for model in config.models
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> dict[str, Any]:
        if request.stream:
            raise HTTPException(status_code=400, detail="streaming is not supported yet")

        features = preprocess([message.model_dump() for message in request.messages])
        decision = choose_route(config, features)
        request_data = request.model_dump(exclude_none=True, exclude={"model", "stream"})
        failures: list[str] = []
        request_id = uuid.uuid4()

        for model_name in decision.candidates:
            model = config.model_by_name(model_name)
            if features.input_tokens > model.max_context:
                failures.append(f"{model_name}: ContextLimitExceeded")
                logger.warning(
                    "request_id=%s selected=%s used=%s reason=%s status=skipped "
                    "error=ContextLimitExceeded input_tokens=%s max_context=%s",
                    request_id,
                    decision.selected_model,
                    model_name,
                    decision.reason,
                    features.input_tokens,
                    model.max_context,
                )
                continue
            provider = app.providers[model_name]
            for attempt in range(model.max_retries + 1):
                started_at = time.perf_counter()
                try:
                    result = await provider.generate(model, request_data)
                    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
                    usage = result.get("usage", {})
                    cost = _cost(model, usage)
                    logger.info(
                        "request_id=%s selected=%s used=%s reason=%s latency_ms=%s "
                        "input_tokens=%s output_tokens=%s estimated_cost=%s status=success",
                        request_id,
                        decision.selected_model,
                        model_name,
                        decision.reason,
                        elapsed_ms,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        cost,
                    )
                    result["model"] = model_name
                    return result
                except Exception as error:
                    upstream_status = (
                        error.response.status_code
                        if isinstance(error, httpx.HTTPStatusError)
                        else None
                    )
                    failure = (
                        f"{model_name}: HTTP {upstream_status}"
                        if upstream_status is not None
                        else f"{model_name}: {type(error).__name__}"
                    )
                    failures.append(failure)
                    logger.warning(
                        "request_id=%s selected=%s used=%s reason=%s attempt=%s status=error "
                        "latency_ms=%s upstream_status=%s error=%s",
                        request_id,
                        decision.selected_model,
                        model_name,
                        decision.reason,
                        attempt + 1,
                        round((time.perf_counter() - started_at) * 1000, 2),
                        upstream_status,
                        type(error).__name__,
                    )

        raise HTTPException(
            status_code=502,
            detail={"message": "all configured models failed", "failures": failures},
        )

    return app


app = create_app()