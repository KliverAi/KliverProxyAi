#!/usr/bin/env python3
"""
Test script to verify structured output works with the fixed code
"""
import json
import requests

# Test payload similar to what the user was sending
test_payload = {
    "model": "gemini-2.0-flash-exp",
    "api_key": "YOUR_API_KEY_HERE",  # Replace with actual API key
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant that responds with structured data."
        },
        {
            "role": "user", 
            "content": "Generate a simple character with name and description."
        }
    ],
    "temperature": 0.1,
    "output_schema": {
        "name": "CharacterResponse",
        "strict": True,
        "output_schema": {
            "type": "object",
            "properties": {
                "character_name": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["character_name", "description"]
        }
    }
}

def test_structured_output():
    """Test the structured output endpoint"""
    url = "http://localhost:8000/api/chat"
    
    try:
        response = requests.post(url, json=test_payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Structured output test passed!")
        else:
            print("❌ Structured output test failed!")
            
    except Exception as e:
        print(f"❌ Error testing structured output: {e}")

if __name__ == "__main__":
    print("Testing structured output fix...")
    test_structured_output()