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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from aiocache import Cache

from app.config import settings, logger
from app.routes import chat, cache

# Restore stderr after imports
sys.stderr = _original_stderr

# Configure aiocache
Cache.MEMORY = Cache.MEMORY or Cache(Cache.MEMORY)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
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


@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "service": settings.APP_NAME,
        "status": "running",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "telemetry_enabled": bool(settings.APPLICATIONINSIGHTS_CONNECTION_STRING)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
