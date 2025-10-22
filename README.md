# Kliver.AI Chat API

A FastAPI-based REST API for chat interactions with multiple AI providers (OpenAI, Google Gemini, Anthropic Claude) using LangChain.

## Features

- **Multi-Provider Support**: OpenAI GPT, Google Gemini, Anthropic Claude
- **Structured Output**: JSON schema-based responses for structured data
- **Token Usage Tracking**: Detailed token consumption metrics
- **Telemetry**: Azure Application Insights integration
- **Docker Support**: Containerized deployment with Docker Compose
- **Kubernetes Ready**: CI/CD pipeline for Scaleway Kubernetes deployment

## Quick Start

### Local Development

```bash
# Install dependencies
poetry install

# Run the server
poetry run python run.py
```

Server runs at `http://localhost:8000` with docs at `http://localhost:8000/docs`.

### Docker

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## API Usage

### POST /api/chat

Send messages to AI models with automatic provider detection.

**Request:**
```json
{
  "model": "gpt-4",
  "api_key": "sk-...",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": {
    "content": "Hello! How can I help you today?",
    "model": "gpt-4",
    "role": "assistant"
  },
  "token_usage": {
    "input_token_count": 10,
    "output_token_count": 15,
    "total_token_count": 25
  }
}
```

### Structured Output

Add `output_schema` for JSON responses:

```json
{
  "model": "gpt-4",
  "api_key": "sk-...",
  "messages": [{"role": "user", "content": "Analyze this product"}],
  "output_schema": {
    "name": "string",
    "category": "string",
    "rating": "string"
  }
}
```

## Supported Models

### OpenAI
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-4`, `gpt-4-turbo-preview`
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano` (reasoning models)
- `o1`, `o1-mini`, `o3`, `o4-mini`

### Google Gemini
- `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.0-flash-exp`

### Anthropic Claude
- `claude-3-7-sonnet-20250219`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
- `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# API Keys
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
ANTHROPIC_API_KEY=your-anthropic-key

# Optional: Telemetry
APPLICATIONINSIGHTS_CONNECTION_STRING=your-connection-string

# Optional: Tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
```

## Deployment

### Docker Compose (Development)
```bash
docker-compose up -d
```

### Kubernetes (Production)
Automatic deployment via GitHub Actions on push to main branch.

### Manual Kubernetes
```bash
kubectl apply -f k8s/
```

## Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "Kliver.AI Chat API",
  "version": "3.0.0",
  "telemetry_enabled": true
}
```

## Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest

# Format code
black .
isort .
```

## Architecture

- **FastAPI**: Web framework with automatic OpenAPI docs
- **LangChain**: AI provider abstraction and message handling
- **Pydantic**: Request/response validation
- **Azure Application Insights**: Telemetry and monitoring
- **Docker**: Containerization
- **Kubernetes**: Orchestration

## License

MIT
