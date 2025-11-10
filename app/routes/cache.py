"""Cache routes for the API"""
from typing import List, Optional
from fastapi import APIRouter, status, Form, UploadFile, File
from aiocache import caches

from app.models import CacheCreationResponse
from app.services.cache_service import create_gemini_context_cache

router = APIRouter(prefix="/api/cache", tags=["cache"])


@router.post(
    "/create",
    response_model=CacheCreationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new Gemini Context Cache resource for large documents.",
    description="Uploads files to Gemini File API and creates a context cache. Supports PDFs, documents, images, audio, video, and code files up to 5GB each."
)
async def create_context_cache_endpoint(
    api_key: str = Form(...),
    model: str = Form(...),
    system_instruction: Optional[str] = Form(None),
    display_name: Optional[str] = Form(None),
    contents_to_cache: Optional[str] = Form(None),  # JSON array of strings or single string
    files: Optional[List[UploadFile]] = File(None),
    use_file_api: bool = Form(True)  # New parameter to choose upload method
) -> CacheCreationResponse:
    """Create a Gemini context cache"""
    return await create_gemini_context_cache(
        api_key=api_key,
        model=model,
        system_instruction=system_instruction,
        display_name=display_name,
        contents_to_cache=contents_to_cache,
        files=files,
        use_file_api=use_file_api
    )


@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
    summary="Get memory cache statistics",
    description="Returns the current size and stats of the in-memory request/response cache"
)
async def get_memory_cache_stats():
    """Get statistics about the in-memory cache"""
    cache = caches.get('default')
    # aiocache doesn't provide a direct way to count items
    # This is a limitation of the library
    return {
        "cache_type": "aiocache",
        "backend": "SimpleMemoryCache",
        "ttl": 7200,
        "message": "Cache is active with 2 hour TTL"
    }


@router.delete(
    "/clear",
    status_code=status.HTTP_200_OK,
    summary="Clear the in-memory cache",
    description="Removes all cached request/response pairs from memory"
)
async def clear_memory_cache():
    """Clear all entries from the in-memory cache"""
    cache = caches.get('default')
    await cache.clear()
    
    return {
        "message": "Cache cleared successfully"
    }
