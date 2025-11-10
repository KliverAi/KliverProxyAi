"""Test script for in-memory cache functionality"""
import asyncio
import time
from app.models import ChatRequest, ChatMessage, ChatRole, AIProvider
from app.services.chat_service import process_chat_request, clear_cache, get_cache_stats


async def test_cache():
    """Test the in-memory cache"""
    # Replace with your actual API key for testing
    api_key = "test-api-key"
    
    # Create a sample request
    request = ChatRequest(
        model="gpt-3.5-turbo",
        api_key=api_key,
        messages=[
            ChatMessage(role=ChatRole.USER, content="Hello, how are you?")
        ],
        provider=AIProvider.OPENAI,
        temperature=0.7
    )
    
    print("=" * 60)
    print("Testing In-Memory Cache")
    print("=" * 60)
    
    # Check initial cache stats
    print("\n1️⃣ Initial cache stats:")
    stats = get_cache_stats()
    print(f"   Cache size: {stats['cache_size']}")
    
    # First request (should be a cache miss)
    print("\n2️⃣ Making first request (should be cache MISS)...")
    start_time = time.time()
    try:
        response1 = await process_chat_request(request)
        duration1 = time.time() - start_time
        print(f"   ✅ Response received in {duration1:.2f}s")
        print(f"   Content: {response1.response.content[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Check cache stats after first request
    print("\n3️⃣ Cache stats after first request:")
    stats = get_cache_stats()
    print(f"   Cache size: {stats['cache_size']}")
    
    # Second request with same parameters (should be a cache hit)
    print("\n4️⃣ Making second request with same params (should be cache HIT)...")
    start_time = time.time()
    response2 = await process_chat_request(request)
    duration2 = time.time() - start_time
    print(f"   ✅ Response received in {duration2:.2f}s")
    print(f"   Content: {response2.response.content[:100]}...")
    
    # Compare responses
    print("\n5️⃣ Comparing responses:")
    print(f"   Responses are identical: {response1 == response2}")
    print(f"   Speed improvement: {((duration1 - duration2) / duration1 * 100):.1f}%")
    
    # Third request with different parameters (should be a cache miss)
    print("\n6️⃣ Making third request with different message (should be cache MISS)...")
    request3 = ChatRequest(
        model="gpt-3.5-turbo",
        api_key=api_key,
        messages=[
            ChatMessage(role=ChatRole.USER, content="What is the weather today?")
        ],
        provider=AIProvider.OPENAI,
        temperature=0.7
    )
    start_time = time.time()
    try:
        response3 = await process_chat_request(request3)
        duration3 = time.time() - start_time
        print(f"   ✅ Response received in {duration3:.2f}s")
        print(f"   Content: {response3.response.content[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Check final cache stats
    print("\n7️⃣ Final cache stats:")
    stats = get_cache_stats()
    print(f"   Cache size: {stats['cache_size']}")
    
    # Clear cache
    print("\n8️⃣ Clearing cache...")
    cleared = clear_cache()
    print(f"   ✅ Cleared {cleared} items from cache")
    
    # Check cache stats after clearing
    print("\n9️⃣ Cache stats after clearing:")
    stats = get_cache_stats()
    print(f"   Cache size: {stats['cache_size']}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    print("\n⚠️  Note: This test requires a valid API key.")
    print("⚠️  Update the 'api_key' variable in the script before running.\n")
    
    # Uncomment to run the test
    # asyncio.run(test_cache())
    print("📝 To run the test, uncomment the last line in this script.")
