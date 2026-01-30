"""
3 LLM Comparison - Query OpenAI, Google, and Anthropic models side by side.

Requires API keys in .env:
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AIza...
    ANTHROPIC_API_KEY=sk-ant-...
"""

import os
from dotenv import load_dotenv

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic

load_dotenv()


def query_openai(prompt: str) -> str:
    """Query OpenAI GPT-5.2."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "[ERROR] OPENAI_API_KEY not set in .env"

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def query_google(prompt: str) -> str:
    """Query Google Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "[ERROR] GOOGLE_API_KEY not set in .env"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


def query_anthropic(prompt: str) -> str:
    """Query Anthropic Claude."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "[ERROR] ANTHROPIC_API_KEY not set in .env"

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def main():
    """Get user query, send to 3 LLMs, and compare results."""
    user_query = input("Enter your query: ")
    print("\n" + "=" * 60)

    print("\nQuerying OpenAI (gpt-5.2)...")
    response_openai = query_openai(user_query)

    print("Querying Google (gemini-2.0-flash)...")
    response_google = query_google(user_query)

    print("Querying Anthropic (claude-sonnet-4-5)...")
    response_anthropic = query_anthropic(user_query)

    print("\n" + "=" * 60)
    print("LLM COMPARISON RESULTS")
    print("=" * 60)

    print("\n[ OpenAI GPT-5.2 ]")
    print(response_openai)
    print("\n" + "-" * 40)

    print("\n[ Google Gemini 2.0 Flash ]")
    print(response_google)
    print("\n" + "-" * 40)

    print("\n[ Anthropic Claude Sonnet 4.5 ]")
    print(response_anthropic)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
