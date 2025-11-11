"""LLM service for managing AI model interactions"""
import mimetypes
from typing import Union, List
from langchain_openai import ChatOpenAI
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
    context_cache_name: str | None = None
):
    """
    Create the appropriate LLM client based on the provider.
    """
    logger.info(f"Creating LLM client - Provider: {provider.value}, Model: {model}, Temperature: {temperature}, Cache: {context_cache_name}")

    if provider == AIProvider.OPENAI:
        model_lower = model.lower()
        if model_lower.startswith("gpt-5") or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            logger.info(f"Detected reasoning model: {model} - Using minimal reasoning_effort and low verbosity")
            model_kwargs = {"reasoning_effort": "minimal"}
            if model_lower.startswith("gpt-5"):
                model_kwargs["verbosity"] = "low"
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=1.0,
                model_kwargs=model_kwargs,
            )
        else:
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )

    elif provider == AIProvider.GEMINI:
        # GEMINI LOGIC WITH CACHE
        # Always use ChatGoogleGenerativeAI with API key

        # Configure thinking_budget for Pro models (not Flash or Lite)
        model_lower = model.lower()
        thinking_config = {}

        if "pro" in model_lower and "flash" not in model_lower and "lite" not in model_lower:
            # gemini-2.5-pro uses thinking budget for complex reasoning
            thinking_config["thinking_budget"] = 8192  # Balanced budget for complex tasks
            logger.info(f"Gemini Pro model detected - Setting thinking_budget: 8192 tokens")

        if context_cache_name:
            logger.info(f"Using Gemini Context Cache: {context_cache_name}")
            # When using cached content, we pass it via model_kwargs
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                cached_content=context_cache_name,
                **thinking_config
            )
        else:
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                **thinking_config
            )

    elif provider == AIProvider.CLAUDE:
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
    else:
        logger.error(f"Unsupported provider requested: {provider}")
        raise ValueError(f"Unsupported provider: {provider}")
