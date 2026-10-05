# Model Router

A config-driven LLM gateway with an OpenAI-compatible chat completions endpoint. It routes requests using lightweight prompt features and can fall back across OpenAI-compatible APIs, Anthropic, and Ollama.

## Quickstart

Requires Python 3.11 or newer. Create an environment and install the project:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item config\models.example.yaml config\models.yaml
Copy-Item .env.example .env
```

Add the keys for the providers you use to `.env`. Ollama must be running locally for its configured fallback to work. Then launch the API:

```powershell
$env:MODEL_ROUTER_CONFIG = "config/models.yaml"
uvicorn router.api.app:app --app-dir src --reload
```

Point an OpenAI SDK client at `http://localhost:8000/v1` and use one of the model names from the config:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-required")
response = client.chat.completions.create(
    model="fast-small",
    messages=[{"role": "user", "content": "Explain a hash table briefly."}],
)
print(response.choices[0].message.content)
```

The request's `model` is accepted for compatibility; routing is controlled by `config/models.yaml`. The preprocessor uses an approximate character-based token estimate, so use it for routing thresholds rather than billing. Each attempt and successful decision is emitted through Python logging; prompt contents and secrets are not logged.

## Configuration

`config/models.example.yaml` shows model IDs, provider endpoints, key environment-variable names, context limits, per-1K token prices, retry counts, and ordered routing rules. Supported provider values are `openai`, `anthropic`, `ollama`, `gemini`, and `groq`. Gemini and Groq use their OpenAI-compatible chat completions APIs; the router supplies their default API URLs and reads `GEMINI_API_KEY` or `GROQ_API_KEY` unless `api_key_env` is configured. Model IDs, context limits, and pricing should match your provider account. The example routes ordinary queries to Groq and sends image, code, long, or keyword-matched queries to Gemini. Configure keywords with `query_contains_any`; matches are case-insensitive whole words or phrases, and all conditions in a rule must match. Rules are evaluated in order and the first match wins. Routing is deterministic and does not call an LLM to classify the query. The selected model is tried first, then the configured default and fallback models, with duplicates removed.

The API currently supports non-streaming chat completions and `/v1/models`. Streaming, persistent metrics, tool-call translation, complexity/cascade strategies, evaluation, and learned routing are not implemented yet.

## Development

```powershell
ruff check .
ruff format --check .
mypy src
pytest
```

Run the service in Docker with `docker compose up --build`. See [PLAN.md](PLAN.md) for milestone status.