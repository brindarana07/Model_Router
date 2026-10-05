from dataclasses import dataclass

from router.core.config import RouterConfig
from router.core.preprocessor import RequestFeatures


@dataclass(frozen=True)
class RouteDecision:
    selected_model: str
    candidates: list[str]
    reason: str


def _matches(conditions: dict[str, object], features: RequestFeatures) -> bool:
    values: dict[str, object] = features.model_dump()
    for key, expected in conditions.items():
        if key.endswith("_gt"):
            feature_name = key[:-3]
            if feature_name not in values or not isinstance(values[feature_name], (int, float)):
                return False
            if not values[feature_name] > expected:  # type: ignore[operator]
                return False
        elif key not in values or values[key] != expected:
            return False
    return True


def choose_route(config: RouterConfig, features: RequestFeatures) -> RouteDecision:
    selected = config.routing.default
    reason = "default"
    for rule in config.routing.rules:
        if _matches(rule.conditions, features):
            selected = rule.use
            reason = f"rule:{rule.use}"
            break

    candidates = list(dict.fromkeys([selected, *config.routing.fallbacks]))
    return RouteDecision(selected_model=selected, candidates=candidates, reason=reason)