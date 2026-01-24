"""Compare answers with and without Tavily web search."""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

load_dotenv()

# Questions that NEED current web data
CURRENT_DATA_QUESTIONS = [
    "What is the current price of Bitcoin in USD?",
    "Who won the most recent Super Bowl and what was the score?",
    "What major news happened in the tech industry this week?",
]

# Questions that DON'T need current web data
STATIC_QUESTIONS = [
    "What is the square root of 144?",
    "Who wrote Romeo and Juliet?",
    "What is the chemical formula for water?",
]

async def ask_deepseek(question: str, use_web_search: bool) -> str:
    """Ask DeepSeek a question with or without web search."""
    from peerrank.providers import call_llm

    try:
        response, duration, _, _, _ = await call_llm(
            "deepseek", "deepseek-chat", question,
            max_tokens=200, use_web_search=use_web_search
        )
        return f"({duration:.1f}s) {response[:300]}"
    except Exception as e:
        return f"ERROR: {e}"

async def main():
    print("=" * 70)
    print("TAVILY COMPARISON TEST")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("QUESTIONS REQUIRING CURRENT DATA")
    print("=" * 70)

    for q in CURRENT_DATA_QUESTIONS:
        print(f"\nQ: {q}")
        print("-" * 60)

        # Without Tavily
        print("WITHOUT web search:")
        ans_no_web = await ask_deepseek(q, use_web_search=False)
        print(f"  {ans_no_web}")

        # With Tavily
        print("WITH web search (Tavily):")
        ans_web = await ask_deepseek(q, use_web_search=True)
        print(f"  {ans_web}")

    print("\n" + "=" * 70)
    print("QUESTIONS NOT REQUIRING CURRENT DATA")
    print("=" * 70)

    for q in STATIC_QUESTIONS:
        print(f"\nQ: {q}")
        print("-" * 60)

        # Without Tavily
        print("WITHOUT web search:")
        ans_no_web = await ask_deepseek(q, use_web_search=False)
        print(f"  {ans_no_web}")

        # With Tavily
        print("WITH web search (Tavily):")
        ans_web = await ask_deepseek(q, use_web_search=True)
        print(f"  {ans_web}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
