"""Application configuration and telemetry setup"""
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings"""

    # Application info
    APP_NAME: str = "Kliver.AI Chat API"
    APP_VERSION: str = "3.0.0"
    APP_DESCRIPTION: str = "API for chat interactions using LangChain with OpenAI, Google Gemini, and Anthropic Claude support"

    # Azure Monitor Application Insights
    APPLICATIONINSIGHTS_CONNECTION_STRING: str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")

    # LangSmith Configuration
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "")


settings = Settings()


def configure_logging():
    """Configure application logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Silence gRPC warnings
    logging.getLogger("absl").setLevel(logging.ERROR)

    return logging.getLogger("kliver.ai")


def configure_telemetry():
    """Configure Azure Monitor Application Insights and return tracer"""
    if settings.APPLICATIONINSIGHTS_CONNECTION_STRING:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor
            from opentelemetry import trace

            configure_azure_monitor(logger_name="kliver.ai")
            tracer = trace.get_tracer("kliver-ai")
            print("✅ Application Insights configured successfully with automatic distributed tracing")
            return tracer
        except Exception as e:
            print(f"❌ Failed to configure Application Insights: {e}")
            return None
    else:
        print("⚠️ Application Insights connection string not found - telemetry disabled")
        return None


def configure_langsmith():
    """Configure LangSmith tracing"""
    if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        if settings.LANGSMITH_PROJECT:
            os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
            print(f"✅ LangSmith tracing enabled - Project: {settings.LANGSMITH_PROJECT}")
        else:
            print("✅ LangSmith tracing enabled - Using default project")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        for key in ["LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_PROJECT", "LANGSMITH_PROJECT", "LANGCHAIN_ENDPOINT"]:
            os.environ.pop(key, None)
        print("⚠️ LangSmith tracing disabled")


# Initialize configuration
logger = configure_logging()
tracer = configure_telemetry()
configure_langsmith()
