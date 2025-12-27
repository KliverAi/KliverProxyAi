"""Chat models for the API"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from app.models.base import ChatRole, AIProvider


class ChatMessage(BaseModel):
    """Represents a chat message with role and content"""
    role: ChatRole = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")
    file_url: Optional[str] = Field(
        None, 
        description="Optional public URL to a file (image, video, audio, document). Gemini will fetch and process it."
    )
    file_mime_type: Optional[str] = Field(
        None,
        description="MIME type of the file (e.g., 'image/jpeg', 'application/pdf'). Auto-detected if not provided."
    )

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
    azure_endpoint: Optional[str] = Field(
        None,
        description="Optional Azure OpenAI endpoint (e.g., 'https://your-resource.openai.azure.com/'). If provided, uses Azure OpenAI instead of standard OpenAI."
    )
    azure_api_version: Optional[str] = Field(
        "2024-08-01-preview",
        description="Azure OpenAI API version (default: 2024-08-01-preview)"
    )
    provider_endpoint: Optional[str] = Field(
        None,
        description="Optional custom provider endpoint/base URL (e.g., an OpenAI-compatible proxy). If provided, it overrides the default API base for supported providers."
    )
    # Alias for legacy clients that send provider_url
    provider_url: Optional[str] = Field(
        None,
        description="Alias of provider_endpoint used by some clients",
    )
    # Vertex AI (Google Cloud) support
    vertex_project: Optional[str] = Field(
        None,
        description="Google Cloud project ID for Vertex AI (used when calling Gemini via Vertex)"
    )
    vertex_location: Optional[str] = Field(
        "us-central1",
        description="Google Cloud region for Vertex AI (default: us-central1)"
    )
    oauth_token: Optional[str] = Field(
        None,
        description="OAuth access token (ya29...) for Vertex AI. If provided, will be used instead of api_key for Gemini via Vertex."
    )

    @field_validator("provider_endpoint", mode="after")
    @classmethod
    def fill_provider_endpoint_from_url(cls, v, info):
        # If provider_endpoint is not provided but provider_url is, use it
        if not v:
            provider_url = info.data.get("provider_url") if hasattr(info, "data") else None
            return provider_url or v
        return v

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
                    "temperature": 0.7,
                    "provider_endpoint": "https://api.my-proxy.example/v1"
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
    thoughts_token_count: Optional[int] = Field(None, description="Number of thinking/reasoning tokens used (Gemini 2.5 Pro models)")

    def __init__(self, input_tokens: int = 0, output_tokens: int = 0, total_tokens: int = None, thoughts_tokens: int = None, **data):
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        super().__init__(
            input_token_count=input_tokens,
            output_token_count=output_tokens,
            total_token_count=total_tokens,
            thoughts_token_count=thoughts_tokens,
            **data
        )


class ChatResponse(BaseModel):
    """Response model for the chat endpoint"""
    content: str = Field(..., description="The response content from the AI")
    model: str = Field(..., description="The model used")
    role: ChatRole = Field(default=ChatRole.ASSISTANT, description="The role of the response")


class AiResponse(BaseModel):
    """Wrapper that contains both the AI response and token usage information"""
    response: ChatResponse = Field(..., description="The AI chat response")
    token_usage: Optional[TokenAiServiceUsageInfo] = Field(None, description="Token usage information (None if not available)")
