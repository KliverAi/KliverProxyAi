"""Main FastAPI application"""
import os
import sys
import warnings

# Silence gRPC/ALTS warnings BEFORE any imports
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to devnull temporarily to suppress C++ level warnings
import io
_original_stderr = sys.stderr
sys.stderr = io.StringIO()

warnings.filterwarnings('ignore', category=Warning)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import time
from aiocache import Cache

from app.config import settings, logger
from app.routes import chat, cache

# Restore stderr after imports
sys.stderr = _original_stderr

# Configure aiocache
Cache.MEMORY = Cache.MEMORY or Cache(Cache.MEMORY)

# Metadata for tags
tags_metadata = [
    {
        "name": "root",
        "description": "Basic information endpoints for the API service.",
    },
    {
        "name": "health",
        "description": "Health check endpoints for monitoring and load balancers.",
    },
    {
        "name": "chat",
        "description": """Operations with AI chat models. 
        
**Supported Providers:**
- 🤖 **OpenAI** (GPT-4, GPT-4-turbo, GPT-3.5-turbo, o1-preview, o1-mini)
- 🌟 **Google Gemini** (gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp)
- 🎭 **Anthropic Claude** (claude-3-5-sonnet, claude-3-opus, claude-3-sonnet)

**Key Features:**
- Multi-turn conversations with history
- Structured JSON output with schemas
- File processing (images, videos, audio, documents)
- Context caching for large documents
- Automatic response caching (2 hours TTL)
        """,
        "externalDocs": {
            "description": "LangChain Documentation",
            "url": "https://python.langchain.com/docs/",
        },
    },
    {
        "name": "cache",
        "description": """Cache management operations for performance optimization.
        
**Features:**
- 📦 Create Gemini Context Caches for large documents
- 📊 View cache statistics
- 🧹 Clear cache when needed
- ⚡ Automatic request/response caching

**Benefits:**
- Reduced API costs
- Faster response times
- Efficient large document processing
        """,
    },
]

# Create FastAPI app with enhanced documentation
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    openapi_tags=tags_metadata,
    contact={
        "name": "Germán Küber",
        "email": "german.kuber@outlook.com",
    },
    license_info={
        "name": "MIT",
    },
    docs_url="/swagger",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(cache.router)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        method = request.method
        path = request.url.path
        try:
            response = await call_next(request)
            status = response.status_code
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(f"{method} {path} -> {status} in {duration_ms:.1f}ms")
            return response
        except Exception as e:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception(f"{method} {path} -> 500 in {duration_ms:.1f}ms: {e}")
            raise


app.add_middleware(RequestLoggingMiddleware)


@app.get(
    "/",
    tags=["root"],
    summary="Root endpoint",
    description="Get basic information about the API service including version and documentation URL.",
    response_description="Service information and status"
)
async def root():
    """
    Root endpoint that provides basic service information.
    
    Returns:
        dict: Service name, status, version, and documentation URL
    """
    logger.info("Root endpoint accessed")
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "version": settings.APP_VERSION,
        "docs": "/swagger"
    }


@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Verify the API service health and telemetry status.",
    response_description="Health status and service information"
)
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        dict: Health status, service name, version, and telemetry configuration
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "telemetry_enabled": bool(settings.APPLICATIONINSIGHTS_CONNECTION_STRING)
    }


@app.on_event("startup")
async def on_startup():
    logger.info("🚀 FastAPI app startup complete. Docs at /swagger")


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("🛑 FastAPI app shutdown")


if __name__ == "__main__":
    import os
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
