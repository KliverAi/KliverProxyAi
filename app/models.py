from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


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


class ChatMessage(BaseModel):
    """Represents a chat message with role and content"""
    role: ChatRole = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")

    @field_validator('role', mode='before')
    @classmethod
    def validate_role(cls, v):
        """Convert string role to ChatRole enum"""
        if isinstance(v, str):
            role_map = {
                "system": ChatRole.SYSTEM,
                "user": ChatRole.USER,
                "assistant": ChatRole.ASSISTANT,
                "tool": ChatRole.TOOL,
                "developer": ChatRole.DEVELOPER,
            }
            role_lower = v.lower()
            if role_lower not in role_map:
                raise ValueError(f"Invalid chat role: {v}")
            return role_map[role_lower]
        return v

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message"""
        return cls(role=ChatRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message"""
        return cls(role=ChatRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message"""
        return cls(role=ChatRole.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str) -> "ChatMessage":
        """Create a tool message"""
        return cls(role=ChatRole.TOOL, content=content)

    @classmethod
    def developer(cls, content: str) -> "ChatMessage":
        """Create a developer message"""
        return cls(role=ChatRole.DEVELOPER, content=content)


class ChatRequest(BaseModel):
    """Request model for the chat endpoint"""
    model: str = Field(..., description="The model to use (e.g., 'gpt-4', 'gemini-pro')")
    api_key: str = Field(..., description="API key (OpenAI or Google AI)")
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    provider: Optional[AIProvider] = Field(
        None,
        description="AI provider (auto-detected if not specified)"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for response generation (0.0 to 2.0). Higher values make output more random."
    )
    output_schema: Optional[dict] = Field(
        None,
        description="Schema for structured output (field_name: field_type pairs)"
    )
    run_name: Optional[str] = Field(
        None,
        description="Optional custom name for LangSmith tracing run"
    )
    context_cache_name: Optional[str] = Field(
        None,
        description="Optional name of a Gemini context cache to use for this chat"
    )

    def get_provider(self) -> AIProvider:
        """
        Detect the AI provider based on the model name if not explicitly set.

        Returns:
            AIProvider: The detected or specified provider
        """
        if self.provider:
            return self.provider

        # Auto-detect based on model name
        model_lower = self.model.lower()
        if model_lower.startswith("gemini"):
            return AIProvider.GEMINI
        elif model_lower.startswith("claude"):
            return AIProvider.CLAUDE
        elif model_lower.startswith("gpt") or "davinci" in model_lower or "turbo" in model_lower:
            return AIProvider.OPENAI
        else:
            # Default to OpenAI for backward compatibility
            return AIProvider.OPENAI

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-4",
                    "api_key": "sk-...",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello!"}
                    ],
                    "temperature": 0.7
                },
                {
                    "model": "gemini-pro",
                    "api_key": "AIza...",
                    "messages": [
                        {"role": "user", "content": "Hello!"}
                    ],
                    "provider": "gemini",
                    "temperature": 0.9
                },
                {
                    "model": "claude-3-7-sonnet-20250219",
                    "api_key": "sk-ant-...",
                    "messages": [
                        {"role": "user", "content": "Hello!"}
                    ],
                    "provider": "claude",
                    "temperature": 1.0
                },
                {
                    "model": "gpt-4",
                    "api_key": "sk-...",
                    "messages": [
                        {"role": "user", "content": "Analyze this product: iPhone 15"}
                    ],
                    "output_schema": {
                        "product_name": "string",
                        "category": "string", 
                        "price_range": "string",
                        "rating": "string"
                    },
                    "temperature": 0.7
                }
            ]
        }
    }


class TokenAiServiceUsageInfo(BaseModel):
    """Token usage information for AI service calls"""
    input_token_count: int = Field(..., description="Number of input tokens consumed")
    output_token_count: int = Field(..., description="Number of output tokens generated") 
    total_token_count: int = Field(..., description="Total tokens consumed (input + output)")
    
    def __init__(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = None, **data):
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        super().__init__(
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=total_tokens,
            **data
        )


class AiResponse(BaseModel):
    """Wrapper that contains both the AI response and token usage information"""
    response: "ChatResponse" = Field(..., description="The AI chat response")
    token_usage: Optional[TokenAiServiceUsageInfo] = Field(None, description="Token usage information (None if not available)")


class ChatResponse(BaseModel):
    """Response model for the chat endpoint"""
    content: str = Field(..., description="The response content from the AI")
    model: str = Field(..., description="The model used")
    role: ChatRole = Field(default=ChatRole.ASSISTANT, description="The role of the response")
