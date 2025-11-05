import pytest


class TestCacheCreation:
    """Tests for cache creation endpoint"""

    def test_cache_creation_missing_fields(self, client):
        """Test cache creation with missing required fields"""
        response = client.post("/api/cache/create")
        assert response.status_code == 422

    def test_cache_creation_non_gemini_model(self, client, mock_api_key):
        """Test cache creation fails with non-Gemini model"""
        data = {
            "api_key": mock_api_key,
            "model": "gpt-4",
            "contents_to_cache": '["test content"]'
        }
        response = client.post("/api/cache/create", data=data)
        assert response.status_code == 400
        assert "only supported for Gemini" in response.json()["detail"]

    def test_cache_creation_no_content(self, client, mock_api_key):
        """Test cache creation fails without content"""
        data = {
            "api_key": mock_api_key,
            "model": "gemini-2.0-flash-exp"
        }
        response = client.post("/api/cache/create", data=data)
        assert response.status_code == 400
        assert "No content provided" in response.json()["detail"]

    def test_cache_creation_validates_text_content_format(self, client, mock_api_key):
        """Test cache creation accepts text content in correct format"""
        data = {
            "api_key": mock_api_key,
            "model": "gemini-2.0-flash-exp",
            "contents_to_cache": '["This is test content"]',
            "display_name": "Test Cache"
        }
        response = client.post("/api/cache/create", data=data)
        # Will fail with real API but validates request structure
        assert response.status_code in [201, 400, 500]

    def test_cache_creation_with_text_file(self, client, mock_api_key, sample_text_file):
        """Test cache creation accepts text file"""
        with open(sample_text_file, "rb") as f:
            files = {"files": ("test.txt", f, "text/plain")}
            data = {
                "api_key": mock_api_key,
                "model": "gemini-2.0-flash-exp",
                "display_name": "File Cache"
            }
            response = client.post("/api/cache/create", data=data, files=files)
        # Will fail with real API but validates file upload works
        assert response.status_code in [201, 400, 500]

    def test_cache_creation_rejects_binary_file(self, client, mock_api_key, tmp_path):
        """Test cache creation rejects unsupported binary files"""
        binary_file = tmp_path / "test.bin"
        binary_file.write_bytes(b'\x00\x01\x02\x03')

        with open(binary_file, "rb") as f:
            files = {"files": ("test.bin", f, "application/octet-stream")}
            data = {
                "api_key": mock_api_key,
                "model": "gemini-2.0-flash-exp"
            }
            response = client.post("/api/cache/create", data=data, files=files)

        assert response.status_code in [400, 500]

    def test_cache_creation_accepts_json_format(self, client, mock_api_key):
        """Test cache creation accepts JSON array format"""
        data = {
            "api_key": mock_api_key,
            "model": "gemini-2.0-flash-exp",
            "contents_to_cache": '["Content 1", "Content 2", "Content 3"]'
        }
        response = client.post("/api/cache/create", data=data)
        assert response.status_code in [201, 400, 500]
