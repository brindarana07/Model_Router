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


def test_routes_matching_query_keywords_case_insensitively() -> None:
    raw_config = sample_config().model_dump(by_alias=True)
    raw_config["routing"]["rules"].insert(
        0, {"if": {"query_contains_any": ["step by step", "compare"]}, "use": "strong"}
    )
    config = RouterConfig.model_validate(raw_config)
    features = preprocess([{"role": "user", "content": "Compare these options, please."}])

    decision = choose_route(config, features)

    assert decision.selected_model == "strong"
    assert decision.reason == "rule:strong"


def test_query_keyword_rules_match_whole_words() -> None:
    raw_config = sample_config().model_dump(by_alias=True)
    raw_config["routing"]["rules"] = [{"if": {"query_contains_any": ["analyze"]}, "use": "strong"}]
    config = RouterConfig.model_validate(raw_config)
    features = preprocess([{"role": "user", "content": "Which analysis is useful?"}])

    assert choose_route(config, features).selected_model == "cheap"


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


def test_config_rejects_empty_query_keyword_list() -> None:
    config = sample_config().model_dump(by_alias=True)
    config["routing"]["rules"] = [{"if": {"query_contains_any": []}, "use": "strong"}]

    with pytest.raises(ValidationError, match="query_contains_any"):
        RouterConfig.model_validate(config)


def test_example_config_routes_by_query_type() -> None:
    config = RouterConfig.load("config/models.example.yaml")

    ordinary_query = preprocess([{"role": "user", "content": "Say hello briefly."}])
    complex_query = preprocess([{"role": "user", "content": "Compare the options step by step."}])
    code_query = preprocess([{"role": "user", "content": "Write Python code to sort a list."}])

    assert choose_route(config, ordinary_query).selected_model == "groq-fast"
    assert choose_route(config, complex_query).selected_model == "gemini-flash"
    assert choose_route(config, code_query).selected_model == "gemini-flash"
