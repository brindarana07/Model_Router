FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY config ./config

EXPOSE 8000
CMD ["uvicorn", "router.api.app:app", "--host", "0.0.0.0", "--port", "8000"]