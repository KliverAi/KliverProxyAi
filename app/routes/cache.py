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
    summary="Create Gemini Context Cache",
    description="""
    Create a Gemini Context Cache resource for efficient processing of large documents.
    
    **Features:**
    - 📁 Upload files up to 5GB each
    - 🎯 Supports multiple file formats: PDF, documents, images, audio, video, code
    - ⚡ Reduced latency and cost for repeated queries
    - 💾 Cache expires after 1 hour of inactivity
    
    **Supported File Types:**
    - Documents: PDF, TXT, HTML, Markdown
    - Images: PNG, JPEG, WEBP, GIF
    - Audio: WAV, MP3, AIFF, AAC, OGG, FLAC
    - Video: MP4, MPEG, MOV, AVI, FLV, MPG, WEBM, WMV, 3GPP
    - Code: Python, JavaScript, Java, C++, and more
    
    **Use Cases:**
    - Long document Q&A
    - Video/audio analysis
    - Large codebase understanding
    """,
    response_description="Cache creation response with cache name and metadata"
)
async def create_context_cache_endpoint(
    api_key: str = Form(..., description="Google AI API key"),
    model: str = Form(..., description="Gemini model name (e.g., gemini-1.5-pro)"),
    system_instruction: Optional[str] = Form(None, description="System instruction for the model"),
    display_name: Optional[str] = Form(None, description="Display name for the cache"),
    contents_to_cache: Optional[str] = Form(None, description="JSON array of strings or single string to cache"),
    files: Optional[List[UploadFile]] = File(None, description="Files to upload and cache"),
    use_file_api: bool = Form(True, description="Use File API for upload (recommended for files > 20MB)")
) -> CacheCreationResponse:
    """
    Create a Gemini context cache for large document processing.
    
    Args:
        api_key: Google AI API key
        model: Gemini model name
        system_instruction: Optional system instruction
        display_name: Optional display name for the cache
        contents_to_cache: Text content to cache
        files: Files to upload and cache
        use_file_api: Whether to use File API for upload
        
    Returns:
        CacheCreationResponse: Cache creation details including cache name
    """
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
    summary="Get cache statistics",
    description="""
    Get statistics and information about the in-memory cache system.
    
    **Returns information about:**
    - 📊 Cache type and backend
    - ⏱️ TTL (Time To Live) configuration
    - 💾 Current cache status
    
    The cache automatically stores chat responses for 2 hours to improve performance and reduce API costs.
    """,
    response_description="Cache statistics and configuration"
)
async def get_memory_cache_stats():
    """
    Get statistics about the in-memory cache.
    
    Returns:
        dict: Cache type, backend, TTL, and status information
    """
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
    summary="Clear cache",
    description="""
    Clear all cached request/response pairs from memory.
    
    **Use cases:**
    - 🧹 Free up memory
    - 🔄 Force fresh API responses
    - 🐛 Debugging cache-related issues
    
    ⚠️ **Warning:** This will remove all cached responses. Subsequent requests will hit the AI provider APIs directly.
    """,
    response_description="Cache clear confirmation message"
)
async def clear_memory_cache():
    """
    Clear all entries from the in-memory cache.
    
    Returns:
        dict: Success message confirming cache was cleared
    """
    cache = caches.get('default')
    await cache.clear()
    
    return {
        "message": "Cache cleared successfully"
    }
