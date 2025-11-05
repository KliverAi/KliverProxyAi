# Quick Start Guide

## 🚀 Quick Start

### 1. Install dependencies

```bash
make install
```

### 2. Run the application

```bash
# Normal mode
make run

# Development mode (with auto-reload)
make dev
```

The application will be available at: `http://localhost:8000`

### 3. View API documentation

```bash
make docs
# Or visit: http://localhost:8000/docs
```

### 4. Run tests

```bash
# All tests
make test

# With coverage report
make test-cov

# View coverage in browser
make coverage
```

## 📋 Available Commands

```bash
make help              # View all commands
make install           # Install dependencies
make test              # Run tests
make test-cov          # Tests with coverage
make run               # Run the app
make dev               # Run in development mode
make clean             # Clean temporary files
make lint              # Run linter
make format            # Format code
make health            # Check if service is running
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Azure Application Insights (optional)
APPLICATIONINSIGHTS_CONNECTION_STRING=your_connection_string

# LangSmith Tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=your_project_name
```

## 📡 API Usage

### 1. Create a Cache (Gemini)

```bash
curl -X POST "http://localhost:8000/api/cache/create" \
  -F "api_key=YOUR_GEMINI_API_KEY" \
  -F "model=gemini-2.0-flash-exp" \
  -F "system_instruction=You are an expert assistant" \
  -F "display_name=My cache" \
  -F "files=@document.txt"
```

Response:
```json
{
  "cache_name": "cachedContents/abc123..."
}
```

### 2. Chat without Cache

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "api_key": "YOUR_OPENAI_API_KEY",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

### 3. Chat with Cache (Gemini)

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash-exp",
    "api_key": "YOUR_GEMINI_API_KEY",
    "context_cache_name": "cachedContents/abc123...",
    "messages": [
      {"role": "user", "content": "Summarize the document"}
    ]
  }'
```

## 🎯 Supported Providers

### OpenAI
- Models: `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`, `o1`, `o3`
- Auto-detected by model name

### Google Gemini
- Models: `gemini-2.0-flash-exp`, `gemini-pro`, etc.
- Supports Context Caching
- Auto-detected by model name

### Anthropic Claude
- Models: `claude-3-opus`, `claude-3-sonnet`, etc.
- Auto-detected by model name

## 🧪 Testing

- **Total: 20 tests**
- **Passing: 20/20 (100%)**
- **Coverage: 54%**

See more details in [TESTING.md](TESTING.md)

## 📁 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # Main application
│   └── models.py        # Pydantic models
├── tests/
│   ├── conftest.py      # Fixtures
│   ├── test_health.py   # Health tests
│   ├── test_chat.py     # Chat tests
│   └── test_cache.py    # Cache tests
├── run.py               # Script to run the app
├── Makefile             # Useful commands
├── pytest.ini           # Pytest configuration
└── pyproject.toml       # Dependencies
```

## 🛠 Development

### Add new dependency

```bash
poetry add package-name
```

### Run specific tests

```bash
# One file
poetry run pytest tests/test_chat.py

# Specific test
poetry run pytest tests/test_chat.py::test_name

# Tests matching pattern
poetry run pytest -k "cache"
```

### View logs

Logs include:
- Request information
- LLM call duration
- Token usage
- Errors and warnings

## 🔍 Verify Everything Works

```bash
# 1. Install
make install

# 2. Run tests
make test

# 3. Start the app
make dev

# 4. In another terminal, check health
make health

# You should see:
# {
#   "status": "healthy",
#   "service": "Kliver.AI Chat API",
#   "version": "3.0.0",
#   ...
# }
```

## 📚 Resources

- **API Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **Testing Guide**: [TESTING.md](TESTING.md)

## ❓ Troubleshooting

### Error: "Failed to initialize Gemini client"
- Verify your API key is valid
- Make sure you're using a Gemini model for cache

### Error: "python-multipart required"
- Run: `make install`

### Tests fail
- Run: `make clean && make install && make test`

### Port 8000 busy
- Change port in `run.py` or use: `kill $(lsof -t -i:8000)`

## 🤝 Contributing

1. Create a branch: `git checkout -b feature/new-feature`
2. Make your changes
3. Run tests: `make test`
4. Commit: `git commit -m "feat: description"`
5. Push: `git push origin feature/new-feature`
6. Create a Pull Request
