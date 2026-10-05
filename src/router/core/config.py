from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: Literal["openai", "anthropic", "ollama", "gemini", "groq"]
    model_id: str
    api_base: str | None = None
    api_key_env: str | None = None
    cost_per_1k_input: float = Field(default=0, ge=0)
    cost_per_1k_output: float = Field(default=0, ge=0)
    max_context: int = Field(gt=0)
    tier: str = "default"
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=0, ge=0, le=5)


class RoutingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conditions: dict[str, Any] = Field(alias="if")
    use: str

    @model_validator(mode="after")
    def validate_conditions(self) -> "RoutingRule":
        for feature, expected in self.conditions.items():
            if feature in {"has_code", "has_image"}:
                if not isinstance(expected, bool):
                    raise ValueError(f"{feature} rule values must be boolean")
            elif feature == "input_tokens_gt":
                if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
                    raise ValueError("input_tokens_gt rule values must be non-negative integers")
            elif feature == "query_contains_any":
                if (
                    not isinstance(expected, list)
                    or not expected
                    or any(not isinstance(term, str) or not term.strip() for term in expected)
                ):
                    raise ValueError("query_contains_any rule values must be non-empty strings")
            else:
                raise ValueError(f"unsupported routing feature: {feature}")
        return self


class RoutingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: Literal["rules"] = "rules"
    default: str
    fallbacks: list[str] = Field(default_factory=list)
    rules: list[RoutingRule] = Field(default_factory=list)


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[ModelConfig]
    routing: RoutingConfig

    @model_validator(mode="after")
    def validate_model_references(self) -> "RouterConfig":
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        known = set(names)
        references = [self.routing.default, *self.routing.fallbacks]
        references.extend(rule.use for rule in self.routing.rules)
        unknown = sorted(set(references) - known)
        if unknown:
            raise ValueError(f"routing references unknown models: {', '.join(unknown)}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "RouterConfig":
        with Path(path).open(encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
        if not isinstance(raw_config, dict):
            raise ValueError("configuration must be a YAML mapping")
        return cls.model_validate(raw_config)

    def model_by_name(self, name: str) -> ModelConfig:
        return next(model for model in self.models if model.name == name)
