"""LLM service for managing AI model interactions"""
import mimetypes
import os
from typing import Union, List, Optional
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from app.models import AIProvider, ChatRole, ChatMessage, TokenAiServiceUsageInfo
from app.config import logger


def convert_to_langchain_message(chat_message: ChatMessage, provider: AIProvider):
    """Convert ChatMessage to LangChain message format with file support"""
    role_to_message = {
        ChatRole.SYSTEM: SystemMessage,
        ChatRole.USER: HumanMessage,
        ChatRole.ASSISTANT: AIMessage,
        ChatRole.DEVELOPER: SystemMessage,
    }

    message_class = role_to_message.get(chat_message.role)
    if not message_class:
        raise ValueError(f"Unsupported role: {chat_message.role}")

    # Si hay file_url, creamos contenido multimodal para Gemini
    if chat_message.file_url and provider == AIProvider.GEMINI:
        # Detectar MIME type si no está especificado
        mime_type = chat_message.file_mime_type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(chat_message.file_url)
            if not mime_type:
                mime_type = "application/octet-stream"
        
        logger.info(f"📎 Adding file from URL: {chat_message.file_url} (type: {mime_type})")
        
        # Para Gemini, usamos el formato de contenido multimodal
        # Ver: https://python.langchain.com/docs/integrations/chat/google_generative_ai
        content = [
            {"type": "text", "text": chat_message.content},
            {
                "type": "image_url" if mime_type.startswith("image/") else "file_url",
                "file_url": chat_message.file_url,
            }
        ]
        return message_class(content=content)
    
    # Si no hay archivo o no es Gemini, mensaje normal
    return message_class(content=chat_message.content)


def extract_token_usage(response, provider: AIProvider) -> TokenAiServiceUsageInfo:
    """
    Extract token usage information from LangChain response based on provider.
    """
    try:
        # Try usage_metadata first (standardized across providers)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            thoughts_tokens = usage.get('thoughts_token_count') or usage.get('thoughtsTokenCount')
            return TokenAiServiceUsageInfo(
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                total_tokens=usage.get('total_tokens', 0),
                thoughts_tokens=thoughts_tokens
            )

        # Fallback to response_metadata for provider-specific formats
        if hasattr(response, 'response_metadata') and response.response_metadata:
            metadata = response.response_metadata

            if provider == AIProvider.OPENAI:
                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    return TokenAiServiceUsageInfo(
                        input_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('completion_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0)
                    )

            elif provider == AIProvider.CLAUDE:
                if 'usage' in metadata:
                    usage = metadata['usage']
                    return TokenAiServiceUsageInfo(
                        input_tokens=usage.get('input_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0)
                    )

        logger.warning(f"No token usage information found for {provider.value}")
        return TokenAiServiceUsageInfo(input_tokens=0, output_tokens=0)

    except Exception as e:
        logger.error(f"Error extracting token usage for {provider.value}: {str(e)}")
        return TokenAiServiceUsageInfo(input_tokens=0, output_tokens=0)


def create_llm(
    provider: AIProvider,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    context_cache_name: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_version: str = "2024-08-01-preview",
    provider_endpoint: str | None = None,
    vertex_project: Optional[str] = None,
    vertex_location: Optional[str] = None,
    oauth_token: Optional[str] = None,
):
    """
    Create the appropriate LLM client based on the provider.
    """
    logger.debug(f"Creating LLM client - Provider: {provider.value}, Model: {model}")

    if provider == AIProvider.OPENAI:
        # Check if using Azure OpenAI (Azure Foundry)
        # Priority: 1. Request parameter, 2. Environment variable
        endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = azure_api_version or os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        if endpoint:
            # Azure OpenAI usando AzureChatOpenAI
            logger.debug(f"Using Azure OpenAI - Endpoint: {endpoint}, Deployment: {model}")
            
            return AzureChatOpenAI(
                azure_endpoint=endpoint,
                azure_deployment=model,
                api_key=api_key,
                api_version=api_version,
                temperature=temperature,
                timeout=240.0,  # 4 minutes timeout
                max_retries=1,  # Reducir reintentos para fallar rápido
            )
        else:
            # Standard OpenAI API
            model_lower = model.lower()
            if model_lower.startswith("gpt-5") or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
                logger.info(f"Detected reasoning model: {model} - Using low reasoning effort")
                # OpenAI Reasoning models (o1, o3, o4, gpt-5) use "low", "medium", or "high" for reasoning effort
                # Using "low" favors speed and economical token usage
                model_kwargs = {"reasoning_effort": "low"}
                kwargs = dict(
                    model=model,
                    api_key=api_key,
                    temperature=1.0,
                    model_kwargs=model_kwargs,
                    timeout=240.0,
                    max_retries=1,
                )
                if provider_endpoint:
                    kwargs["base_url"] = provider_endpoint
                try:
                    return ChatOpenAI(**kwargs)
                except TypeError:
                    if provider_endpoint:
                        logger.warning("provider_endpoint not supported for OpenAI in this LangChain version; proceeding without it")
                        kwargs.pop("base_url", None)
                        return ChatOpenAI(**kwargs)
                    raise
            else:
                kwargs = dict(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=240.0,
                    max_retries=1,
                )
                if provider_endpoint:
                    kwargs["base_url"] = provider_endpoint
                try:
                    return ChatOpenAI(**kwargs)
                except TypeError:
                    if provider_endpoint:
                        logger.warning("provider_endpoint not supported for OpenAI in this LangChain version; proceeding without it")
                        kwargs.pop("base_url", None)
                        return ChatOpenAI(**kwargs)
                    raise

    elif provider == AIProvider.GEMINI:
        # GEMINI LOGIC WITH CACHE
        # Prefer Vertex AI path if explicitly requested or inferred
        use_vertex = False
        vertex_hint = (provider_endpoint or "").find("aiplatform.googleapis.com") != -1 if provider_endpoint else False
        if oauth_token or vertex_project or vertex_hint:
            use_vertex = True

        if use_vertex:
            try:
                from langchain_google_vertexai import ChatVertexAI
            except Exception as e:
                raise ValueError(
                    "Vertex AI support requires 'langchain-google-vertexai'. "
                    "Add it to dependencies and install."
                ) from e

            # Build credentials if oauth_token provided; else rely on ADC
            credentials = None
            if oauth_token:
                try:
                    from google.oauth2.credentials import Credentials
                    credentials = Credentials(token=oauth_token)
                except Exception as e:
                    logger.warning(f"Could not construct OAuth credentials from token: {e}. Falling back to ADC.")

            project = vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
            location = vertex_location or os.getenv("GOOGLE_CLOUD_REGION") or "us-central1"

            if not project:
                raise ValueError("Vertex AI selected but no project provided. Set vertex_project or GOOGLE_CLOUD_PROJECT.")

            logger.info(f"Using Vertex AI Gemini model via project '{project}' in '{location}'")

            return ChatVertexAI(
                model=model,
                project=project,
                location=location,
                temperature=temperature,
                max_retries=1,
                request_timeout=240.0,
                credentials=credentials,
            )

        # Otherwise, use Google AI Studio client
        # Configure thinking parameters based on model type
        model_lower = model.lower()
        thinking_config = {}

        # Gemini 3.x models use thinking_level instead of thinking_budget
        if "gemini-3" in model_lower or "gemini-flash-3" in model_lower:
            # gemini-3-flash-preview and other Gemini 3 models use thinking_level
            # Options: minimal, low, medium, high
            # Use minimal for speed and cost efficiency
            thinking_config["thinking_level"] = "minimal"
            logger.debug(f"Gemini 3 model detected - Setting thinking_level: minimal")
        elif "pro" in model_lower and "flash" not in model_lower and "lite" not in model_lower:
            # gemini-2.5-pro uses thinking budget for complex reasoning
            thinking_config["thinking_budget"] = 8192  # Balanced budget for complex tasks
            logger.debug(f"Gemini Pro model detected - Setting thinking_budget: 8192 tokens")

        # Default parameters for Gemini models optimized for speed and quality
        # Based on official Google documentation:
        # - candidate_count: 1 (single response for faster generation)
        # - top_k: 40 (balanced - not too restrictive)
        # - top_p: 0.95 (Google's recommended default)
        # Note: temperature is NOT set here to allow model's default (1.0 for Gemini 3)
        gemini_defaults = {
            "candidate_count": 1,
            "top_k": 40,
            "top_p": 0.95
        }
        logger.debug(f"Applying Gemini defaults optimized for speed: {gemini_defaults}")

        if provider_endpoint:
            logger.warning("provider_endpoint is not supported for Gemini AI Studio client; ignoring this parameter")

        if context_cache_name:
            logger.debug(f"Using Gemini Context Cache: {context_cache_name}")
            # When using cached content, we pass it via model_kwargs
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                cached_content=context_cache_name,
                timeout=240.0,
                max_retries=1,
                **thinking_config,
                **gemini_defaults
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                timeout=240.0,
                max_retries=1,
                **thinking_config,
                **gemini_defaults
            )

    elif provider == AIProvider.CLAUDE:
        # Allow custom base URL for proxies if provided
        kwargs = dict(
            model=model,
            api_key=api_key,
            temperature=temperature,
            timeout=240.0,
            max_retries=1,
        )
        if provider_endpoint:
            kwargs_with_base = dict(kwargs)
            kwargs_with_base["base_url"] = provider_endpoint
            try:
                return ChatAnthropic(**kwargs_with_base)
            except TypeError:
                logger.warning("provider_endpoint not supported for Anthropic in this LangChain version; proceeding without it")
                # fall through to default
        return ChatAnthropic(**kwargs)
    else:
        logger.error(f"Unsupported provider requested: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")
