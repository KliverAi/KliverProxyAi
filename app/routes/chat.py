"""Chat routes for the API"""
from fastapi import APIRouter, HTTPException, status
from aiocache import cached
from aiocache.serializers import PickleSerializer

from app.models import ChatRequest, AiResponse
from app.models.base import AIProvider
from app.services.chat_service import process_chat_request
from app.config import logger

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=AiResponse,
    status_code=status.HTTP_200_OK,
    summary="Send chat messages to AI models",
    description="""
    Process chat messages using multiple AI providers through LangChain.
    
    **Supported Models:**
    - OpenAI: gpt-4, gpt-4-turbo, gpt-3.5-turbo, o1-preview, o1-mini
    - Google Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp
    - Anthropic Claude: claude-3-5-sonnet, claude-3-opus, claude-3-sonnet
    
    **Features:**
    - ✅ Multi-turn conversations with message history
    - ✅ Structured JSON output (provide a JSON schema)
    - ✅ Automatic caching (2 hours TTL)
    - ✅ Support for system messages and user/assistant roles
    - ✅ Context caching for large documents (Gemini)
    
    **Example Request:**
    ```json
    {
        "model": "gpt-4",
        "api_key": "your-api-key",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"}
        ]
    }
    ```
    """,
    response_description="AI model response with content and metadata"
)
# Cache deshabilitado
# @cached(ttl=14200, serializer=PickleSerializer(), noself=True)
async def chat(chat_request: ChatRequest) -> AiResponse:
    """
    Process chat messages using LangChain with OpenAI, Google Gemini, or Anthropic Claude.
    
    Args:
        chat_request: Chat request with model, API key, messages, and optional schema
        
    Returns:
        AiResponse: Response containing AI-generated content and metadata
        
    Raises:
        HTTPException 400: Invalid request parameters
        HTTPException 500: Error processing chat request
    """
    try:
        # Early validation for common credential mismatches
        if chat_request.get_provider() == AIProvider.GEMINI:
            key = (chat_request.api_key or "").strip()
            is_oauth = key.startswith("ya29.") or (chat_request.oauth_token or "").startswith("ya29.")
            is_vertex_endpoint = (chat_request.provider_endpoint or "").find("aiplatform.googleapis.com") != -1
            vertex_params = chat_request.vertex_project is not None

            if is_oauth and not (is_vertex_endpoint or vertex_params):
                raise ValueError(
                    "Token OAuth detectado (ya29...) pero no se indicó Vertex (provider_endpoint de aiplatform o vertex_project). "
                    "Agrega vertex_project/location o provider_endpoint de Vertex, o usa API key AIza... de AI Studio."
                )
        return await process_chat_request(chat_request)

    except ValueError as e:
        logger.exception("❌ ValueError while processing chat request")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        # Capture full stack trace to aid debugging
        msg = str(e)
        logger.exception(f"❌ Unexpected {type(e).__name__} while processing chat request")

        # Map common provider auth errors to 401 for clarity
        auth_markers = [
            "API key not valid",              # Google Gemini
            "API_KEY_INVALID",                # Google error code
            "Incorrect API key provided",     # OpenAI
            "invalid_api_key",                # Generic marker
        ]
        if any(m in msg for m in auth_markers):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {msg}"
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {msg}"
        )
