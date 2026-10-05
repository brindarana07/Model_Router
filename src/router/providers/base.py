from typing import Any, Protocol

from router.core.config import ModelConfig


class ChatProvider(Protocol):
    async def generate(
        self,
        model: ModelConfig,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...