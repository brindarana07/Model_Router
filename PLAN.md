# Project Plan

## Goals

- Provide one OpenAI-compatible `/v1/chat/completions` endpoint.
- Route requests across providers based on lightweight features and configurable strategies.
- Reduce cost while tracking model choice, latency, usage, cost, and failures.
- Retry transient failures and fall back to another configured model.

## Non-goals for v0.1

- Hosting or training models, billing/multi-tenant accounts, or a web dashboard.

## Architecture

Client -> FastAPI API -> preprocessor -> routing strategy -> provider adapters -> response and decision logs.

| Component | Responsibility |
| --- | --- |
| API | Validate OpenAI-compatible requests and return unified responses |
| Preprocessor | Estimate input tokens, detect code and images, and retain query text for configured keyword rules |
| Router | Select the first matching model and its fallback chain |
| Providers | Translate requests for OpenAI-compatible APIs, Anthropic, and Ollama |
| Config | Validate model IDs, prices, limits, rules, retries, and endpoints from YAML |
| Logs | Record selection reason, attempts, latency, usage, estimated cost, and status |

##**Phases**

### Phase 0: Setup

- [x] Python package, Ruff, mypy, pytest, and GitHub Actions CI
- [x] README, MIT license, and ignore rules

### Phase 1: Skeleton

- [x] Common async provider interface and three provider adapters
- [x] Non-streaming `/v1/chat/completions` and `/v1/models`
- [x] YAML model/routing configuration with Pydantic validation
- [ ] Provider contract tests against live/simulated upstream responses

### Phase 2: Basic routing

- [x] Token estimate, code detection, image detection, and configurable query-keyword detection
- [x] Config-driven static rules
- [x] Per-model retries and ordered fallback
- [x] Unit tests for routing and fallback behavior
- [ ] Context-limit enforcement and provider-specific error classification

### Phase 3: Observability

- [x] Log decisions, attempts, latency, token usage, and estimated cost
- [ ] Persist request records in SQLite and add `/stats` or `/metrics`
- [ ] Add streaming support

### Phase 4: Smarter routing

- [ ] Heuristic complexity scoring
- [ ] Cascade strategy with configurable quality checks
- [ ] Select `rules`, `complexity`, or `cascade` in config

### Phase 5: Evaluation

- [ ] Create a labeled evaluation set and runner
- [ ] Compare quality, cost, and latency with always-cheap and always-best baselines
- [ ] Publish results in `docs/results.md`

### Phase 6: Release

- [x] Docker image and Compose example
- [x] Quickstart and configuration guide
- [ ] Contributing guide and issue templates
- [ ] Validate release criteria and tag v0.1.0

## Risks

| Risk | Mitigation |
| --- | --- |
| A weak model reduces answer quality | Build evaluation data early and add cascade checks |
| Routing increases latency | Keep feature extraction local and measure overhead |
| Provider APIs change | Isolate adapters and add contract tests |
| API keys leak | Read keys from environment variables; ignore `.env` |
| Prices or model IDs become stale | Keep them in user-maintained YAML, not code |
