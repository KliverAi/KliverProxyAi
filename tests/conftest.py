import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Test client fixture"""
    return TestClient(app)


@pytest.fixture
def mock_api_key():
    """Mock API key for testing"""
    return "test_api_key_12345"


@pytest.fixture
def sample_chat_request(mock_api_key):
    """Sample chat request for testing"""
    return {
        "model": "gpt-4",
        "api_key": mock_api_key,
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "temperature": 0.7
    }


@pytest.fixture
def sample_gemini_request(mock_api_key):
    """Sample Gemini chat request for testing"""
    return {
        "model": "gemini-2.0-flash-exp",
        "api_key": mock_api_key,
        "messages": [
            {
                "role": "user",
                "content": "Test message"
            }
        ],
        "temperature": 0.7
    }


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file for testing"""
    file_path = tmp_path / "test.txt"
    content = "This is a test file content. " * 100  # Make it long enough for cache
    file_path.write_text(content)
    return file_path


@pytest.fixture
def sample_pdf_file(tmp_path):
    """Create a sample PDF file for testing"""
    from PyPDF2 import PdfWriter
    import io

    file_path = tmp_path / "test.pdf"

    # Create a simple PDF
    pdf_writer = PdfWriter()
    # Note: This creates an empty PDF, for real tests you'd need actual content

    with open(file_path, "wb") as f:
        pdf_writer.write(f)

    return file_path
