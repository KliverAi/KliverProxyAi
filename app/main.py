import logging
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from pydantic import BaseModel, create_model, Field
from dotenv import load_dotenv
import os

from app.models import ChatRequest, ChatResponse, ChatRole, AIProvider, AiResponse, TokenAiServiceUsageInfo

# Load environment variables from .env file
load_dotenv()

# Configure Azure Monitor Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPLICATIONINSIGHTS_CONNECTION_STRING:
    from azure.monitor.opentelemetry import configure_azure_monitor
    configure_azure_monitor(
        connection_string=APPLICATIONINSIGHTS_CONNECTION_STRING,
        instrumentation_options={
            "fastapi": {"enabled": True},
            "requests": {"enabled": True},
        }
    )
    print("✅ Application Insights configured successfully")
else:
    print("⚠️ Application Insights connection string not found - telemetry disabled")

# Configure LangSmith (optional)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    if LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        print(f"✅ LangSmith tracing enabled - Project: {LANGSMITH_PROJECT}")
    else:
        print("✅ LangSmith tracing enabled - Using default project")
else:
    # Disable LangSmith tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    if "LANGSMITH_API_KEY" in os.environ:
        del os.environ["LANGSMITH_API_KEY"]
    print("⚠️ LangSmith tracing disabled")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("kliver.ai")

app = FastAPI(
    title="Kliver.AI Chat API",
    description="API for chat interactions using LangChain with OpenAI, Google Gemini, and Anthropic Claude support",
    version="3.0.0"
)

# Configure CORS for C# client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_to_langchain_message(chat_message):
    """Convert ChatMessage to LangChain message format"""
    role_to_message = {
        ChatRole.SYSTEM: SystemMessage,
        ChatRole.USER: HumanMessage,
        ChatRole.ASSISTANT: AIMessage,
        ChatRole.DEVELOPER: SystemMessage,  # Developer maps to System in LangChain
    }

    message_class = role_to_message.get(chat_message.role)
    if not message_class:
        raise ValueError(f"Unsupported role: {chat_message.role}")

    return message_class(content=chat_message.content)


def extract_token_usage(response, provider: AIProvider) -> TokenAiServiceUsageInfo:
    """
    Extract token usage information from LangChain response based on provider.
    
    Args:
        response: LangChain AI response object
        provider: The AI provider (OpenAI, Gemini, or Claude)
        
    Returns:
        TokenAiServiceUsageInfo with token counts
    """
    try:
        # Try usage_metadata first (standardized across providers)
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            return TokenAiServiceUsageInfo(
                input_tokens=usage.get('input_tokens', 0),
                output_tokens=usage.get('output_tokens', 0),
                total_tokens=usage.get('total_tokens', 0)
            )
        
        # Fallback to response_metadata for provider-specific formats
        if hasattr(response, 'response_metadata') and response.response_metadata:
            metadata = response.response_metadata
            
            if provider == AIProvider.OPENAI:
                # OpenAI format: response_metadata['token_usage']
                if 'token_usage' in metadata:
                    usage = metadata['token_usage']
                    return TokenAiServiceUsageInfo(
                        input_tokens=usage.get('prompt_tokens', 0),
                        output_tokens=usage.get('completion_tokens', 0),
                        total_tokens=usage.get('total_tokens', 0)
                    )
            
            elif provider == AIProvider.CLAUDE:
                # Claude format: response_metadata['usage']
                if 'usage' in metadata:
                    usage = metadata['usage']
                    return TokenAiServiceUsageInfo(
                        input_tokens=usage.get('input_tokens', 0),
                        output_tokens=usage.get('output_tokens', 0)
                    )
            
        # If no usage information is available, return zero counts
        logger.warning(f"No token usage information found for {provider.value}")
        return TokenAiServiceUsageInfo(input_tokens=0, output_tokens=0)
        
    except Exception as e:
        logger.error(f"Error extracting token usage for {provider.value}: {str(e)}")
        return TokenAiServiceUsageInfo(input_tokens=0, output_tokens=0)


def create_llm(provider: AIProvider, model: str, api_key: str, temperature: float = 0.7):
    """
    Create the appropriate LLM client based on the provider.

    Args:
        provider: The AI provider (OpenAI, Gemini, or Claude)
        model: The model name
        api_key: The API key
        temperature: The temperature for response generation

    Returns:
        LLM client instance

    Raises:
        ValueError: If the provider is not supported
    """
    logger.info(f"Creating LLM client - Provider: {provider.value}, Model: {model}, Temperature: {temperature}")

    if provider == AIProvider.OPENAI:
        model_lower = model.lower()

        # Check if it's a GPT-5 or o1 reasoning model
        if model_lower.startswith("gpt-5") or model_lower.startswith("o1") or model_lower.startswith("o3") or model_lower.startswith("o4"):
            logger.info(f"Detected reasoning model: {model} - Using minimal reasoning_effort and low verbosity")

            # GPT-5 and o-series models use special parameters
            # Hardcoded to minimum: reasoning_effort="minimal", verbosity="low"
            model_kwargs = {
                "reasoning_effort": "minimal",  # Minimal reasoning for fastest response
            }

            # GPT-5 models support verbosity parameter
            if model_lower.startswith("gpt-5"):
                model_kwargs["verbosity"] = "low"  # Short, concise answers
                logger.info(f"GPT-5 model - Added verbosity='low'")

            return ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=1.0,  # Reasoning models work best with temperature=1
                model_kwargs=model_kwargs,
            )
        else:
            # Standard GPT models
            return ChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
    elif provider == AIProvider.GEMINI:
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
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


@app.get("/")
async def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {
        "service": "Kliver.AI Chat API",
        "status": "running",
        "version": "3.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    logger.info("Health check endpoint accessed")
    
    health_status = {
        "status": "healthy",
        "service": "Kliver.AI Chat API",
        "version": "3.0.0",
        "timestamp": time.time(),
        "environment": {
            "telemetry_enabled": bool(APPLICATIONINSIGHTS_CONNECTION_STRING),
            "langsmith_tracing": LANGSMITH_TRACING,
            "langsmith_project": LANGSMITH_PROJECT if LANGSMITH_TRACING else None
        },
        "providers": {
            "openai": "available",
            "gemini": "available", 
            "claude": "available"
        }
    }
    
    return health_status


@app.post(
    "/api/chat",
    response_model=AiResponse,
    status_code=status.HTTP_200_OK,
    summary="Send chat messages to AI (supports structured output)",
    description="Process a list of chat messages using OpenAI, Google Gemini, or Anthropic Claude models. If 'schema' field is provided, returns structured JSON output."
)
async def chat(request: ChatRequest) -> AiResponse:
    """
    Process chat messages using LangChain with OpenAI, Google Gemini, or Anthropic Claude.
    Automatically detects if structured output is requested based on the presence of 'schema' field.

    Args:
        request: ChatRequest containing model, api_key, messages, and optional provider and schema

    Returns:
        AiResponse with the AI's response and token usage information
        If schema is provided, returns structured JSON output

    Raises:
        HTTPException: If there's an error processing the request
    """
    request_start_time = time.time()

    # Detect if this is a structured output request
    is_structured = request.output_schema is not None
    request_type = "Structured chat" if is_structured else "Regular chat"

    logger.info("=" * 80)
    logger.info(f"{request_type} request received")
    logger.info(f"Model: {request.model}")
    logger.info(f"Temperature: {request.temperature}")
    logger.info(f"Number of messages: {len(request.messages)}")
    if is_structured:
        logger.info(f"Schema fields: {list(request.output_schema.keys())}")

    try:
        # Use the model and api_key from the request directly
        model = request.model
        api_key = request.api_key

        # Detect or get the provider
        provider = request.get_provider()
        logger.info(f"Provider detected: {provider.value}")

        # Initialize the appropriate LLM client
        llm = create_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=request.temperature,
        )

        # Convert ChatMessage objects to LangChain message format
        langchain_messages = [
            convert_to_langchain_message(msg)
            for msg in request.messages
        ]

        logger.info(f"Messages converted to LangChain format")

        # Handle structured output if schema is provided
        if is_structured:
            # 🔹 1. Crear modelo Pydantic dinámicamente según el schema del request
            fields = {
                name: (str, Field(..., description=f"Field {name} of type {typ}"))
                for name, typ in request.output_schema.items()
            }

            DynamicModel = create_model("DynamicStructuredModel", **fields)
            logger.info(f"Dynamic Pydantic model created with fields: {list(fields.keys())}")

            # 🔹 2. Usar .with_structured_output() que es el método recomendado
            structured_llm = llm.with_structured_output(DynamicModel)

            # Convert ChatMessage objects to LangChain message format  
            langchain_messages = [
                convert_to_langchain_message(msg)
                for msg in request.messages
            ]

            # 🔹 3. Ejecutar directamente con structured output
            llm_start_time = time.time()
            logger.info(f"Calling {provider.value} LLM with structured output...")

            response_obj = await structured_llm.ainvoke(langchain_messages)

            llm_end_time = time.time()
            llm_duration = llm_end_time - llm_start_time

            logger.info(f"Structured LLM response received in {llm_duration:.3f}s")
            logger.info(f"Parsed object type: {type(response_obj)}")

            # For structured output, we create minimal token usage since we don't have access to raw response
            token_usage = TokenAiServiceUsageInfo(input_tokens=0, output_tokens=0)

            response_content = response_obj.model_dump_json(indent=2)

        else:
            # Regular chat flow
            llm_start_time = time.time()
            logger.info(f"Calling {provider.value} LLM...")

            response = await llm.ainvoke(langchain_messages)

            llm_end_time = time.time()
            llm_duration = llm_end_time - llm_start_time

            logger.info(f"LLM response received in {llm_duration:.3f}s")
            logger.info(f"Response length: {len(response.content)} characters")

            # Extract token usage information
            token_usage = extract_token_usage(response, provider)
            logger.info(f"Token usage - Input: {token_usage.input_token_count}, Output: {token_usage.output_token_count}, Total: {token_usage.total_token_count}")

            response_content = response.content

        # Return the response with token usage
        request_duration = time.time() - request_start_time
        logger.info(f"Total request duration: {request_duration:.3f}s")
        logger.info(f"{request_type} completed successfully")
        logger.info("=" * 80)

        return AiResponse(
            response=ChatResponse(
                content=response_content,
                model=model,  # Use the actual model used
                role=ChatRole.ASSISTANT
            ),
            token_usage=token_usage
        )

    except ValueError as e:
        request_duration = time.time() - request_start_time
        logger.error(f"ValueError after {request_duration:.3f}s: {str(e)}")
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        request_duration = time.time() - request_start_time
        logger.error(f"Exception after {request_duration:.3f}s: {str(e)}", exc_info=True)
        logger.info("=" * 80)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
