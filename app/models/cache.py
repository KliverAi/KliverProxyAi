"""Cache models for the API"""
from typing import List, Optional
from pydantic import BaseModel

from app.models.chat import TokenAiServiceUsageInfo


class CacheCreationRequest(BaseModel):
    """Request model for cache creation"""
    api_key: str
    model: str
    contents_to_cache: List[str]
    system_instruction: Optional[str] = None
    display_name: Optional[str] = None


class CacheCreationResponse(BaseModel):
    """Response model for cache creation"""
    cache_name: str
    token_usage: Optional[TokenAiServiceUsageInfo] = None
