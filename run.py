"""Script to run the FastAPI application"""
import os
import uvicorn

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,  # Enable auto-reload during development
        log_level=os.getenv("LOG_LEVEL", "debug"),
        access_log=True,
        proxy_headers=True,
    )
