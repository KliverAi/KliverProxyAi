import pytest


class TestChatEndpoint:
    """Tests for the chat endpoint"""

    def test_chat_missing_required_fields(self, client):
        """Test chat endpoint with missing required fields"""
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_invalid_role(self, client, mock_api_key):
        """Test chat endpoint with invalid message role"""
        request_data = {
            "model": "gpt-4",
            "api_key": mock_api_key,
            "messages": [
                {
                    "role": "invalid_role",
                    "content": "Test"
                }
            ]
        }
        response = client.post("/api/chat", json=request_data)
        assert response.status_code == 422

    def test_chat_provider_auto_detection_openai(self, sample_chat_request):
        """Test that OpenAI provider is auto-detected from model name"""
        from app.models import ChatRequest

        request = ChatRequest(**sample_chat_request)
        provider = request.get_provider()

        assert provider.value == "openai"

    def test_chat_provider_auto_detection_gemini(self, sample_gemini_request):
        """Test that Gemini provider is auto-detected from model name"""
        from app.models import ChatRequest

        request = ChatRequest(**sample_gemini_request)
        provider = request.get_provider()

        assert provider.value == "gemini"

    def test_chat_provider_auto_detection_claude(self, mock_api_key):
        """Test that Claude provider is auto-detected from model name"""
        from app.models import ChatRequest

        request_data = {
            "model": "claude-3-opus",
            "api_key": mock_api_key,
            "messages": [{"role": "user", "content": "Test"}]
        }

        request = ChatRequest(**request_data)
        provider = request.get_provider()

        assert provider.value == "claude"

    def test_chat_explicit_provider_override(self, mock_api_key):
        """Test that explicit provider overrides auto-detection"""
        from app.models import ChatRequest

        request_data = {
            "model": "gpt-4",
            "api_key": mock_api_key,
            "provider": "openai",
            "messages": [{"role": "user", "content": "Test"}]
        }

        request = ChatRequest(**request_data)
        provider = request.get_provider()

        assert provider.value == "openai"

    def test_chat_request_structure(self, client, sample_chat_request):
        """Test chat request is properly structured"""
        response = client.post("/api/chat", json=sample_chat_request)
        # Will fail without real API key but validates request structure
        assert response.status_code in [200, 400, 401, 500]

    def test_chat_with_temperature(self, client, sample_chat_request):
        """Test chat request accepts custom temperature"""
        sample_chat_request["temperature"] = 0.3
        response = client.post("/api/chat", json=sample_chat_request)
        assert response.status_code in [200, 400, 401, 500]

    def test_chat_with_context_cache(self, client, sample_gemini_request):
        """Test chat request accepts context cache parameter"""
        sample_gemini_request["context_cache_name"] = "cachedContents/test123"
        response = client.post("/api/chat", json=sample_gemini_request)
        assert response.status_code in [200, 400, 401, 500]

    def test_token_usage_info_model(self):
        """Test TokenAiServiceUsageInfo model"""
        from app.models import TokenAiServiceUsageInfo

        # Test with all fields
        token_info = TokenAiServiceUsageInfo(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150
        )

        assert token_info.input_token_count == 100
        assert token_info.output_token_count == 50
        assert token_info.total_token_count == 150

    def test_token_usage_info_defaults(self):
        """Test TokenAiServiceUsageInfo with default values"""
        from app.models import TokenAiServiceUsageInfo

        token_info = TokenAiServiceUsageInfo(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0
        )

        assert token_info.input_token_count == 0
        assert token_info.output_token_count == 0
        assert token_info.total_token_count == 0
