"""Chat routes for the API"""
from fastapi import APIRouter, HTTPException, status

from app.models import ChatRequest, AiResponse
from app.services.chat_service import process_chat_request
from app.config import logger

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=AiResponse,
    status_code=status.HTTP_200_OK,
    summary="Send chat messages to AI (supports structured output)",
    description="Process a list of chat messages using OpenAI, Google Gemini, or Anthropic Claude models. If 'schema' field is provided, returns structured JSON output."
)
async def chat(request: ChatRequest) -> AiResponse:
    """
    Process chat messages using LangChain with OpenAI, Google Gemini, or Anthropic Claude.
    """
    try:
        return await process_chat_request(request)

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
