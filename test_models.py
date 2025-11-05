import pydantic
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"

class ChatMessage(BaseModel):
    role: ChatRole
    content: str

class ChatRequest(BaseModel):
    model: str
    api_key: str
    messages: List[ChatMessage]
    context_cache_name: Optional[str] = None

# Test the model
req = ChatRequest(
    model='gemini-pro',
    api_key='test',
    messages=[],
    context_cache_name='test_cache'
)
print('context_cache_name:', req.context_cache_name)
print('Test passed!')