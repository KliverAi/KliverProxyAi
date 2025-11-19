"""Base enums and types for the application"""
from enum import Enum


class ChatRole(str, Enum):
    """Enum for chat message roles"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class AIProvider(str, Enum):
    """Enum for AI providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
