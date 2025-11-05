import logging
import time
from datetime import timedelta
from enum import Enum
from fastapi import FastAPI, HTTPException, status, Request, UploadFile, File, Form
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
from typing import List, Optional

# Assuming app.models now includes CacheCreationRequest, CacheCreationResponse, etc.
# from app.models import ... 
# NOTE: Replace 'from app.models import ...' with your actual imports from models.py
# For this example, we'll assume the necessary models (ChatRequest, AIProvider, etc.) are available.
# Since I don't have your models file, I must define dummy classes for compilation.
# >>> START DUMMY MODEL DEFINITION (REPLACE WITH REAL IMPORTS) <<<
class ChatRole(str, Enum): SYSTEM = "system"; USER = "user"; ASSISTANT = "assistant"; TOOL = "tool"; DEVELOPER = "developer"
class AIProvider(str, Enum): OPENAI = "openai"; GEMINI = "gemini"; CLAUDE = "claude"
class ChatMessage(BaseModel): role: ChatRole; content: str
class TokenAiServiceUsageInfo(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @property
    def input_token_count(self): return self.input_tokens
    @property
    def output_token_count(self): return self.output_tokens
    @property
    def total_token_count(self): return self.total_tokens
class ChatResponse(BaseModel): content: str; model: str; role: ChatRole
class AiResponse(BaseModel): response: ChatResponse; token_usage: Optional[TokenAiServiceUsageInfo]
class CacheCreationRequest(BaseModel): api_key: str; model: str; contents_to_cache: List[str]; system_instruction: Optional[str]; display_name: Optional[str]
class CacheCreationResponse(BaseModel): cache_name: str
class ChatRequest(BaseModel): 
    model: str; api_key: str; messages: List[ChatMessage]; provider: Optional[AIProvider]; temperature: float = 0.7; output_schema: Optional[dict] = None; context_cache_name: Optional[str] = None
    def get_provider(self) -> AIProvider: 
        model_lower = self.model.lower()
        if self.provider: return self.provider
        elif model_lower.startswith("gemini"): return AIProvider.GEMINI
        elif model_lower.startswith("claude"): return AIProvider.CLAUDE
        else: return AIProvider.OPENAI
# >>> END DUMMY MODEL DEFINITION <<<


# Load environment variables from .env file
load_dotenv()

# Configure Azure Monitor Application Insights
APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if APPLICATIONINSIGHTS_CONNECTION_STRING:
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor
        from opentelemetry import trace
        
        configure_azure_monitor(
            logger_name="kliver.ai",
        )
        
        tracer = trace.get_tracer("kliver-ai")
        print("✅ Application Insights configured successfully with automatic distributed tracing")
    except Exception as e:
        print(f"❌ Failed to configure Application Insights: {e}")
        tracer = None
else:
    print("⚠️ Application Insights connection string not found - telemetry disabled")
    tracer = None

# Configure LangSmith (optional)
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")

if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    if LANGSMITH_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
        print(f"✅ LangSmith tracing enabled - Project: {LANGSMITH_PROJECT}")
    else:
        print("✅ LangSmith tracing enabled - Using default project")
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    for key in ["LANGCHAIN_API_KEY", "LANGSMITH_API_KEY", "LANGCHAIN_PROJECT", "LANGSMITH_PROJECT", "LANGCHAIN_ENDPOINT"]:
        os.environ.pop(key, None)
    print("⚠️ LangSmith tracing disabled")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.monitor.opentelemetry.exporter.export._base").setLevel(logging.WARNING)

logger = logging.getLogger("kliver.ai")

app = FastAPI(
    title="Kliver.AI Chat API",
    description="API for chat interactions using LangChain with OpenAI, Google Gemini, and Anthropic Claude support",
    version="3.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---

def convert_to_langchain_message(chat_message):
    """Convert ChatMessage to LangChain message format"""
    role_to_message = {
        ChatRole.SYSTEM: SystemMessage,
        ChatRole.USER: HumanMessage,
        ChatRole.ASSISTANT: AIMessage,
        ChatRole.DEVELOPER: SystemMessage,
    }

    message_class = role_to_message.get(chat_message.role)
    if not message_class:
        raise ValueError(f"Unsupported role: {chat_message.role}")

    return message_class(content=chat_message.content)


def extract_token_usage(response, provider: AIProvider) -> TokenAiServiceUsageInfo:
    """
    Extract token usage information from LangChain response based on provider.
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
        if context_cache_name:
            logger.info(f"Using Gemini Context Cache: {context_cache_name}")
            # When using cached content, we pass it via model_kwargs
            return ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=temperature,
                cached_content=context_cache_name
            )
        else:
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


# --- Core Endpoints ---

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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Kliver.AI Chat API",
        "version": "3.0.0",
        "telemetry_enabled": bool(APPLICATIONINSIGHTS_CONNECTION_STRING)
    }

# --- NEW ENDPOINT: Cache Creation ---

@app.post(
    "/api/cache/create",
    response_model=CacheCreationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Gemini Context Cache resource for large documents.",
    description="Processes large text content once and stores it for repeated, low-cost use in subsequent chat calls. Supports file uploads and text content."
)
async def create_context_cache_endpoint(
    api_key: str = Form(...),
    model: str = Form(...),
    system_instruction: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None),
    contents_to_cache: Optional[str] = Form(None),  # JSON array of strings or single string
    files: Optional[List[UploadFile]] = File(None)
) -> CacheCreationResponse:
    if model is None or not model.lower().startswith('gemini'):
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Context Caching is only supported for Gemini models."
        )

    logger.info(f"Attempting to create Context Cache: {display_name} using {model}...")

    # NOTE: Google Gemini API supports context caching via the REST API
    # We need to use the Google GenAI client to create a cache
    # Reference: https://ai.google.dev/gemini-api/docs/caching

    try:
        import google.generativeai as genai
        import json

        # Configure the API key
        genai.configure(api_key=api_key)

        # Prepare content for caching
        content_list = []

        # Process text contents if provided
        if contents_to_cache:
            try:
                # Try to parse as JSON array
                parsed_contents = json.loads(contents_to_cache)
                if isinstance(parsed_contents, list):
                    content_list.extend(parsed_contents)
                else:
                    content_list.append(str(parsed_contents))
            except json.JSONDecodeError:
                # If not JSON, treat as single string
                content_list.append(contents_to_cache)

        # Process uploaded files
        if files:
            for file in files:
                try:
                    # Read file content
                    file_content = await file.read()

                    # Decode based on file type
                    if file.filename.endswith(('.txt', '.md', '.json', '.csv', '.py', '.js', '.html', '.xml')):
                        # Text files
                        text_content = file_content.decode('utf-8')
                        content_list.append(f"File: {file.filename}\n\n{text_content}")
                    elif file.filename.endswith('.pdf'):
                        # Extract text from PDF using PyPDF2
                        try:
                            from PyPDF2 import PdfReader
                            import io

                            logger.info(f"Processing PDF file: {file.filename}")

                            # Create a file-like object from bytes
                            pdf_file = io.BytesIO(file_content)

                            # Read the PDF
                            pdf_reader = PdfReader(pdf_file)

                            # Extract text from all pages
                            pdf_text = []
                            for page_num, page in enumerate(pdf_reader.pages, 1):
                                page_text = page.extract_text()
                                if page_text.strip():
                                    pdf_text.append(f"--- Page {page_num} ---\n{page_text}")

                            if pdf_text:
                                combined_text = "\n\n".join(pdf_text)
                                content_list.append(f"File: {file.filename}\n\n{combined_text}")
                                logger.info(f"Successfully extracted text from PDF: {file.filename} ({len(pdf_reader.pages)} pages)")
                            else:
                                logger.warning(f"No text extracted from PDF: {file.filename}")
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=f"Could not extract text from PDF {file.filename}. The PDF might be image-based or encrypted."
                                )
                        except ImportError:
                            logger.error("PyPDF2 not installed")
                            raise HTTPException(
                                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="PDF support requires PyPDF2 package. Please install it."
                            )
                        except Exception as pdf_error:
                            logger.error(f"Error parsing PDF {file.filename}: {pdf_error}")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Error parsing PDF {file.filename}: {str(pdf_error)}"
                            )
                    else:
                        # Try to decode as UTF-8 for other files
                        try:
                            text_content = file_content.decode('utf-8')
                            content_list.append(f"File: {file.filename}\n\n{text_content}")
                        except UnicodeDecodeError:
                            logger.warning(f"Cannot decode file {file.filename} as text")
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"File {file.filename} is not a supported text format"
                            )
                except Exception as e:
                    logger.error(f"Error processing file {file.filename}: {e}")
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Error processing file {file.filename}: {str(e)}"
                    )

        if not content_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No content provided. Please provide either 'contents_to_cache' or upload files."
            )

        logger.info(f"Total content items to cache: {len(content_list)}")

        # Create the cache using the Google GenAI SDK
        cache = genai.caching.CachedContent.create(
            model=model,
            display_name=display_name or f"cache_{int(time.time())}",
            system_instruction=system_instruction,
            contents=content_list,
            ttl=timedelta(hours=1)  # Cache expires in 1 hour
        )

        logger.info(f"✅ Context Cache successfully created: {cache.name}")

        return CacheCreationResponse(cache_name=cache.name)

    except ImportError:
        logger.error("google-generativeai package not installed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Missing required package: google-generativeai. Please install it."
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Gemini Context Cache creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during Gemini Context Cache creation: {str(e)}"
        )

# --- Modified Chat Endpoint ---

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
    """
    request_start_time = time.time()

    is_structured = request.output_schema is not None
    request_type = "Structured chat" if is_structured else "Regular chat"

    try:
        model = request.model
        api_key = request.api_key
        provider = request.get_provider()
        
        request_info = f"🚀 {provider.value} {model} request"
        if is_structured:
            request_info += f" (structured: {list(request.output_schema.keys())})"
        if request.context_cache_name:
            request_info += f" (Cache: {request.context_cache_name})"
        logger.info(request_info)

        # Initialize the appropriate LLM client, passing the cache name if present
        llm = create_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            temperature=request.temperature,
            context_cache_name=request.context_cache_name, 
        )

        # Convert ChatMessage objects to LangChain message format
        langchain_messages = [
            convert_to_langchain_message(msg)
            for msg in request.messages
        ]

        # --- Structured Output Logic ---
        if is_structured:
            # ... (Structured output code logic)
            json_schema = request.output_schema
            schema_name = "StructuredOutput"
            
            if "output_schema" in json_schema and isinstance(json_schema["output_schema"], dict):
                schema_name = json_schema.get("name", "StructuredOutput")
                json_schema = json_schema["output_schema"]
                
            if "title" not in json_schema: json_schema["title"] = schema_name
            if "description" not in json_schema: json_schema["description"] = f"Structured output schema for {schema_name}"
            
            structured_llm = llm.with_structured_output(json_schema, method="json_schema")
            
            run_config = {
                "run_name": f"{schema_name}",
                "tags": [provider.value, model, "structured_output", schema_name],
                "metadata": {
                    "provider": provider.value, "model": model, "schema_name": schema_name, "temperature": request.temperature
                }
            }
            
            llm_start_time = time.time()
            
            # Execute with telemetry
            if tracer:
                with tracer.start_as_current_span(f"ai_structured_call_{provider.value}") as span:
                    span.set_attribute("ai.provider", provider.value)
                    # ... (Telemetry attributes)
                    response_obj = await structured_llm.ainvoke(langchain_messages, config=run_config)
                    llm_duration = time.time() - llm_start_time
                    # ... (More telemetry)
            else:
                response_obj = await structured_llm.ainvoke(langchain_messages, config=run_config)
                llm_duration = time.time() - llm_start_time

            logger.info(f"✅ Structured response in {llm_duration:.2f}s")
            
            token_usage = None # Often unavailable for structured output
            
            import json
            if isinstance(response_obj, (dict, list)):
                response_content = json.dumps(response_obj, indent=2)
            elif hasattr(response_obj, 'model_dump_json'):
                response_content = response_obj.model_dump_json(indent=2)
            else:
                response_content = str(response_obj)

        # --- Regular Chat Flow ---
        else:
            run_config = {
                "run_name": f"{provider.value}_{model}_chat",
                "tags": [provider.value, model, "chat"],
                "metadata": {
                    "provider": provider.value,
                    "model": model,
                    "temperature": request.temperature,
                    "message_count": len(langchain_messages),
                    "context_cache_name": request.context_cache_name or "N/A"
                }
            }
            
            llm_start_time = time.time()
            
            # Execute with telemetry
            if tracer:
                with tracer.start_as_current_span(f"ai_chat_call_{provider.value}") as span:
                    span.set_attribute("ai.provider", provider.value)
                    # ... (Telemetry attributes)
                    response = await llm.ainvoke(langchain_messages, config=run_config)
                    llm_end_time = time.time()
                    llm_duration = llm_end_time - llm_start_time
                    # ... (More telemetry)
            else:
                response = await llm.ainvoke(langchain_messages, config=run_config)
                llm_end_time = time.time()
                llm_duration = llm_end_time - llm_start_time

            token_usage = extract_token_usage(response, provider)
            logger.info(f"✅ Response in {llm_duration:.2f}s | Tokens: {token_usage.input_token_count}→{token_usage.output_token_count} ({token_usage.total_token_count})")

            response_content = response.content

        # Log final summary with telemetry
        request_duration = time.time() - request_start_time
        
        if tracer:
            with tracer.start_as_current_span("ai_request_summary") as span:
                span.set_attribute("ai.total_duration_seconds", request_duration)
                span.set_attribute("ai.provider", provider.value)
                span.set_attribute("ai.model", model)
                span.set_attribute("ai.total_tokens", token_usage.total_token_count if token_usage else 0)

        return AiResponse(
            response=ChatResponse(
                content=response_content,
                model=model,
                role=ChatRole.ASSISTANT
            ),
            token_usage=token_usage
        )

    except ValueError as e:
        request_duration = time.time() - request_start_time
        # ... (Error telemetry and raise HTTPException)
        logger.error(f"❌ ValueError: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        request_duration = time.time() - request_start_time
        # ... (Error telemetry and raise HTTPException)
        logger.error(f"❌ {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)