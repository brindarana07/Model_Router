import pytest
from pydantic import ValidationError

from router.core.config import RouterConfig
from router.core.preprocessor import preprocess
from router.core.routing import choose_route


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


def test_routes_code_prompts_to_strong_model() -> None:
    features = preprocess(
        [{"role": "user", "content": "Please review this: ```python\nprint(1)```"}]
    )

    decision = choose_route(sample_config(), features)

    assert decision.selected_model == "strong"
    assert decision.candidates == ["strong", "cheap"]
    assert decision.reason == "rule:strong"


def test_default_route_and_image_detection() -> None:
    features = preprocess(
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    )

    assert features.has_image is True
    assert choose_route(sample_config(), features).selected_model == "cheap"


def test_config_rejects_unknown_route_target() -> None:
    config = sample_config().model_dump(by_alias=True)
    config["routing"]["default"] = "missing"

    with pytest.raises(ValidationError, match="unknown models"):
        RouterConfig.model_validate(config)


def test_config_rejects_unknown_rule_feature() -> None:
    config = sample_config().model_dump(by_alias=True)
    config["routing"]["rules"] = [{"if": {"unknown": True}, "use": "strong"}]

    with pytest.raises(ValidationError, match="unsupported routing feature"):
        RouterConfig.model_validate(config)