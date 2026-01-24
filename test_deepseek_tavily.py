"""Test DeepSeek + Tavily integration."""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

async def main():
    # Check API keys
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")

    print("=== DeepSeek + Tavily Test ===\n")
    print(f"DeepSeek API key: {'OK' if deepseek_key else 'MISSING'}")
    print(f"Tavily API key:   {'OK' if tavily_key else 'MISSING'}")

    if not deepseek_key or not tavily_key:
        print("\nError: Missing API keys in .env")
        return

    # Test Tavily directly
    print("\n--- Testing Tavily Search ---")
    from peerrank.providers import tavily_search

    results, duration = await tavily_search("What is the capital of France?")
    if results:
        print(f"[OK] Tavily working ({duration:.2f}s)")
        print(f"  Result: {results[:100]}...")
    else:
        print("[FAIL] Tavily failed (check quota)")
        return

    # Test DeepSeek with Tavily
    print("\n--- Testing DeepSeek + Tavily ---")
    from peerrank.providers import call_llm

    prompt = "What is the current weather in Tokyo? Keep your answer brief."

    try:
        response, duration, in_tok, out_tok = await call_llm(
            "deepseek", "deepseek-chat", prompt,
            max_tokens=200, use_web_search=True
        )
        print(f"[OK] DeepSeek + Tavily working ({duration:.2f}s)")
        print(f"  Tokens: {in_tok} in / {out_tok} out")
        print(f"  Response: {response[:200]}...")
    except Exception as e:
        print(f"[FAIL] Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
