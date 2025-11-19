"""Chat routes for the API"""
from fastapi import APIRouter, HTTPException, status
from aiocache import cached
from aiocache.serializers import PickleSerializer

from app.models import ChatRequest, AiResponse
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
        return await process_chat_request(chat_request)

    except ValueError as e:
        logger.error(f"❌ ValueError: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )
