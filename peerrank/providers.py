"""
providers.py - LLM provider implementations for PeerRank.ai
"""

import asyncio
import os
import time

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from xai_sdk import AsyncClient as XaiAsyncClient
from xai_sdk.chat import user as xai_user
from xai_sdk.tools import web_search as xai_web_search, x_search as xai_x_search
from mistralai import Mistral

from .config import (
    MODELS,
    MAX_TOKENS_ANSWER, MAX_TOKENS_DEEPSEEK, MAX_TOKENS_GOOGLE,
    DEFAULT_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    TEMPERATURE_DEFAULT, MODEL_TEMPERATURE_OVERRIDES,
    GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_PROJECT_ID, GOOGLE_LOCATION,
    TAVILY_COST_PER_SEARCH, ANTHROPIC_WEB_SEARCH_MAX_USES, OPENAI_WEB_SEARCH_CONTEXT_SIZE,
    GOOGLE_SEARCH_THRESHOLD,
    get_api_key
)

_clients: dict = {}


def clear_clients():
    """Clear cached clients to avoid event loop issues.

    Call this before asyncio.run() if previous async operations
    used a different event loop. The Google genai client in particular
    binds to the event loop at creation time.
    """
    global _clients
    _clients.clear()


def sanitize_prompt(text: str) -> str:
    """Replace known problematic characters for API compatibility.

    Applied equally to all models for fair comparison.
    """
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Smart single quotes
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2013': '-', '\u2014': '-',  # En/em dashes
        '\u2026': '...',               # Ellipsis
        '\u00a0': ' ',                 # Non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _get_temperature(model: str, default_temp: float) -> float:
    """Get temperature for a model, applying overrides if needed."""
    return MODEL_TEMPERATURE_OVERRIDES.get(model, default_temp)


# Client factories
def _get_openai_client(api_key: str, timeout: int, base_url: str | None = None) -> AsyncOpenAI:
    cache_key = f"openai:{base_url or 'default'}:{timeout}"
    if cache_key not in _clients:
        _clients[cache_key] = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return _clients[cache_key]


def _get_anthropic_client(api_key: str, timeout: int) -> AsyncAnthropic:
    if "anthropic" not in _clients:
        _clients["anthropic"] = AsyncAnthropic(api_key=api_key, timeout=timeout)
    return _clients["anthropic"]


def _get_google_client() -> genai.Client:
    if "google" not in _clients:
        auth_method = os.getenv("GOOGLE_AUTH_METHOD", "api_key").lower()

        if auth_method == "service_account":
            if not GOOGLE_SERVICE_ACCOUNT_FILE or not GOOGLE_SERVICE_ACCOUNT_FILE.exists():
                raise ValueError(f"GOOGLE_SERVICE_ACCOUNT_FILE not set or file not found: {GOOGLE_SERVICE_ACCOUNT_FILE}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(GOOGLE_SERVICE_ACCOUNT_FILE)
            _clients["google"] = genai.Client(vertexai=True, project=GOOGLE_PROJECT_ID, location=GOOGLE_LOCATION)
        else:  # api_key (default)
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            _clients["google"] = genai.Client(api_key=api_key)
    return _clients["google"]


def _get_mistral_client(api_key: str) -> Mistral:
    if "mistral" not in _clients:
        _clients["mistral"] = Mistral(api_key=api_key)
    return _clients["mistral"]


# Provider implementations
async def _call_openai(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                       use_web_search: bool, response_format: dict | None, temperature: float) -> tuple[str, float, int, int, float]:
    client = _get_openai_client(api_key, timeout)
    start = time.time()
    effective_temp = _get_temperature(model, temperature)

    input_tokens = 0
    output_tokens = 0

    if use_web_search:
        response = await client.responses.create(
            model=model, input=prompt, tools=[{"type": "web_search_preview", "search_context_size": OPENAI_WEB_SEARCH_CONTEXT_SIZE}], max_output_tokens=max_tokens)
        content = "".join(block.text for item in response.output if item.type == "message"
                         for block in item.content if hasattr(block, 'text'))
        # Extract token usage from response if available
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
    else:
        kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": effective_temp, "max_completion_tokens": max_tokens}
        if response_format:
            kwargs["response_format"] = response_format
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        # Extract token usage from response
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)

    return (content.strip() if content else "", time.time() - start, input_tokens, output_tokens, 0.0)


async def _call_anthropic(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                          use_web_search: bool, response_format: dict | None, temperature: float) -> tuple[str, float, int, int, float]:
    client = _get_anthropic_client(api_key, timeout)
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    extra_headers = {}

    if use_web_search:
        extra_headers["anthropic-beta"] = "web-search-2025-03-05"
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": ANTHROPIC_WEB_SEARCH_MAX_USES}]
        kwargs["system"] = "Use your web_search tool once to find current information, then answer concisely."

    start = time.time()
    response = await client.messages.create(**kwargs, extra_headers=extra_headers) if extra_headers else await client.messages.create(**kwargs)
    content = "".join(block.text for block in response.content if block.type == "text")

    # Extract token usage
    input_tokens = getattr(response.usage, 'input_tokens', 0)
    output_tokens = getattr(response.usage, 'output_tokens', 0)

    return (content.strip(), time.time() - start, input_tokens, output_tokens, 0.0)


async def _call_google(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                       use_web_search: bool, response_format: dict | None, temperature: float) -> tuple[str, float, int, int, float]:
    client = _get_google_client()

    # Handle gemini-3-flash-preview variant (maps to gemini-2.5-flash)
    actual_model = model
    if model == "gemini-3-flash-preview":
        actual_model = "gemini-2.5-flash"

    # Note: thinking/reasoning mode disabled for all models
    effective_max_tokens = min(max_tokens, MAX_TOKENS_GOOGLE)
    config = {"temperature": temperature, "max_output_tokens": effective_max_tokens}

    if use_web_search:
        config["tools"] = [{
            "google_search": {
                "dynamic_retrieval_config": {
                    "mode": "MODE_DYNAMIC",
                    "dynamic_threshold": GOOGLE_SEARCH_THRESHOLD
                }
            }
        }]

    start = time.time()
    response = await client.aio.models.generate_content(model=actual_model, contents=prompt, config=config)
    duration = time.time() - start

    content = ""
    try:
        content = response.text or ""
    except Exception:
        pass

    # Fallback: extract from candidates (handles MAX_TOKENS truncation)
    candidates = getattr(response, 'candidates', None) or []
    if not content and candidates:
        for candidate in candidates:
            parts = getattr(getattr(candidate, 'content', None), 'parts', None) or []
            for part in parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text

    # Check finish reason for diagnostics
    finish_reason = None
    if candidates:
        finish_reason = getattr(candidates[0], 'finish_reason', None)

    if not content:
        reason = finish_reason or 'no_candidates'
        raise ValueError(f"Google API empty response (reason={reason})")

    # Warn if truncated but still return partial content
    if finish_reason and 'MAX_TOKENS' in str(finish_reason) and content:
        print(f"      [WARN] Google response truncated (MAX_TOKENS), using partial content", flush=True)

    # Extract token usage (include thinking tokens for models that use them)
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage_metadata'):
        input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
        output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        # Include thinking tokens in output count (they're billed as output)
        thoughts_tokens = getattr(response.usage_metadata, 'thoughts_token_count', 0)
        if thoughts_tokens:
            output_tokens += thoughts_tokens

    return (content.strip(), duration, input_tokens, output_tokens, 0.0)


# Tavily search for providers without native search
async def tavily_search(query: str, max_results: int = 5) -> tuple[str, float, bool]:
    """Returns (search_results, duration_seconds, success)"""
    import aiohttp
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "", 0.0, False

    start = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.tavily.com/search",
                json={"api_key": api_key, "query": query, "max_results": max_results,
                      "include_answer": True, "search_depth": "basic"},
                timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"      [Tavily HTTP {resp.status}: {error_text[:200]}]")
                    return "", time.time() - start, False
                data = await resp.json()

        # Format as natural prose, not citation list
        parts = []
        if data.get("answer"):
            parts.append(data['answer'])
        for r in data.get("results", [])[:max_results]:
            content = r.get('content', '')[:250]
            if content:
                parts.append(content)
        return " ".join(parts), time.time() - start, True
    except Exception as e:
        print(f"      [Tavily error: {e}]")
        return "", time.time() - start, False


def _make_openai_caller(base_url: str, use_tavily: bool = False, max_tokens_limit: int | None = None,
                        supports_response_format: bool = True):
    """Factory for OpenAI-compatible API callers."""
    async def _call(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                    use_web_search: bool, response_format: dict | None, temperature: float) -> tuple[str, float, int, int, float]:
        client = _get_openai_client(api_key, timeout, base_url)
        effective_max = min(max_tokens, max_tokens_limit) if max_tokens_limit else max_tokens
        effective_temp = _get_temperature(model, temperature)

        messages = []
        tavily_duration = 0.0
        tavily_cost = 0.0
        if use_tavily and use_web_search:
            search_results, tavily_duration, tavily_success = await tavily_search(prompt[:400])
            if tavily_success:
                tavily_cost = TAVILY_COST_PER_SEARCH
            if search_results:
                # Inject as system context - invisible to response style
                messages.append({"role": "system", "content":
                    f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{search_results}"})

        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": model, "messages": messages, "temperature": effective_temp, "max_tokens": effective_max}
        if response_format and supports_response_format:
            kwargs["response_format"] = response_format

        start = time.time()
        response = await client.chat.completions.create(**kwargs)
        llm_duration = time.time() - start

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)

        return (response.choices[0].message.content.strip(), tavily_duration + llm_duration, input_tokens, output_tokens, tavily_cost)

    return _call


# Grok implementation using xAI SDK with native web search and X search
async def _call_grok(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                     use_web_search: bool, response_format: dict | None, temperature: float) -> tuple[str, float, int, int, float]:
    client = XaiAsyncClient(api_key=api_key)
    effective_temp = _get_temperature(model, temperature)

    kwargs = {
        "model": model,
        "temperature": effective_temp,
        "max_tokens": max_tokens,
    }

    if use_web_search:
        kwargs["tools"] = [xai_web_search(), xai_x_search()]

    chat = client.chat.create(**kwargs)
    chat.append(xai_user(prompt))

    start = time.time()
    response = await chat.sample()
    duration = time.time() - start

    content = response.content or ""

    # Extract token usage from response
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage') and response.usage:
        input_tokens = getattr(response.usage, 'prompt_tokens', 0) or getattr(response.usage, 'input_tokens', 0) or 0
        output_tokens = getattr(response.usage, 'completion_tokens', 0) or getattr(response.usage, 'output_tokens', 0) or 0

    return (content.strip(), duration, input_tokens, output_tokens, 0.0)


# Mistral implementation with native web search via Agents API
async def _call_mistral(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                        use_web_search: bool, response_format: dict | None, temperature: float) -> tuple[str, float, int, int, float]:
    client = _get_mistral_client(api_key)
    effective_temp = _get_temperature(model, temperature)

    start = time.time()
    input_tokens = 0
    output_tokens = 0

    if use_web_search:
        # Use Agents API with web_search connector for grounded responses
        # Create a web search agent (or reuse cached one)
        cache_key = f"mistral_websearch_agent:{model}"
        if cache_key not in _clients:
            agent = await client.beta.agents.create_async(
                model=model,
                name="WebSearch Agent",
                description="Agent with web search capability for current information",
                instructions="You have the ability to perform web searches with web_search to find up-to-date information. Answer accurately based on search results.",
                tools=[{"type": "web_search"}],
                completion_args={
                    "temperature": effective_temp,
                    "max_tokens": max_tokens,
                }
            )
            _clients[cache_key] = agent.id

        agent_id = _clients[cache_key]

        # Start conversation with the agent
        response = await client.beta.conversations.start_async(
            agent_id=agent_id,
            inputs=prompt
        )

        # Extract content from response - handle multiple output types
        content = ""
        if hasattr(response, 'outputs') and response.outputs:
            for output in response.outputs:
                # Skip tool execution entries, only process message outputs
                if getattr(output, 'type', '') == 'tool.execution':
                    continue
                if hasattr(output, 'content'):
                    out_content = output.content
                    if isinstance(out_content, str):
                        content += out_content
                    elif isinstance(out_content, list):
                        # Content is a list of text items
                        for item in out_content:
                            if hasattr(item, 'text') and item.text:
                                content += item.text
                            elif isinstance(item, str):
                                content += item
        elif hasattr(response, 'content'):
            content = response.content
        elif hasattr(response, 'message'):
            content = response.message.content if hasattr(response.message, 'content') else str(response.message)

        # Extract token usage if available
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) or getattr(response.usage, 'input_tokens', 0) or 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) or getattr(response.usage, 'output_tokens', 0) or 0
    else:
        # Use standard chat completions without web search
        response = await client.chat.complete_async(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=effective_temp,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content if response.choices else ""

        # Extract token usage
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)

    duration = time.time() - start
    return (content.strip() if content else "", duration, input_tokens, output_tokens, 0.0)


# Provider instances
_call_deepseek = _make_openai_caller("https://api.deepseek.com", use_tavily=True, max_tokens_limit=MAX_TOKENS_DEEPSEEK)
_call_together = _make_openai_caller("https://api.together.xyz/v1", use_tavily=True)
_call_perplexity = _make_openai_caller("https://api.perplexity.ai", supports_response_format=False)
_call_kimi = _make_openai_caller("https://api.moonshot.ai/v1", use_tavily=True)

_PROVIDER_CALLS = {
    "openai": _call_openai, "anthropic": _call_anthropic, "google": _call_google,
    "grok": _call_grok, "deepseek": _call_deepseek, "together": _call_together, "perplexity": _call_perplexity,
    "kimi": _call_kimi, "mistral": _call_mistral,
}


async def call_llm(provider: str, model: str, prompt: str, max_tokens: int = MAX_TOKENS_ANSWER,
                   timeout: int = DEFAULT_TIMEOUT, use_web_search: bool = False,
                   response_format: dict | None = None, retries: int = MAX_RETRIES,
                   temperature: float = TEMPERATURE_DEFAULT) -> tuple[str, float, int, int, float]:
    """
    Call LLM API with automatic retry on transient failures.

    Returns:
        tuple: (content, duration, input_tokens, output_tokens, tavily_cost)
    """
    api_key = get_api_key(provider)
    call_fn = _PROVIDER_CALLS.get(provider)
    if not call_fn:
        raise ValueError(f"Unknown provider: {provider}")

    # Sanitize prompt for all models equally
    prompt = sanitize_prompt(prompt)

    last_error = None
    for attempt in range(retries):
        try:
            return await call_fn(model, prompt, api_key, max_tokens, timeout, use_web_search, response_format, temperature)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            retryable = any(x in error_str for x in ["timeout", "rate limit", "429", "500", "502", "503", "504",
                                                      "overloaded", "capacity", "empty response"])
            if retryable and attempt < retries - 1:
                delay = RETRY_DELAY * (2 ** attempt)
                print(f"  [Retry {attempt + 1}] {model} - {type(e).__name__}: {str(e)[:60]}, waiting {delay}s...")
                await asyncio.sleep(delay)
            else:
                raise
    raise last_error


def list_google_models() -> list[str]:
    try:
        client = _get_google_client()
        return sorted(m.name for m in client.models.list()
                     if getattr(m, 'supported_actions', None) is None or 'generateContent' in m.supported_actions)
    except Exception as e:
        print(f"  Error listing models: {e}")
        return []


def get_google_auth_method() -> str:
    """Return which auth method Google will use based on GOOGLE_AUTH_METHOD env var."""
    return os.getenv("GOOGLE_AUTH_METHOD", "api_key").lower()


def _get_reasoning_mode(provider: str, model_id: str) -> str:
    """Get the reasoning/thinking mode description for a model."""
    if provider == "google":
        if "thinking" in model_id:
            return "thinking: OFF (disabled)"
        return "standard"
    elif provider == "openai":
        if "nano" in model_id or "mini" in model_id:
            return "reasoning: default (may use internal CoT)"
        return "standard"
    elif provider == "anthropic":
        return "extended thinking: OFF"
    elif provider == "deepseek":
        return "reasoning: default (may use internal CoT)"
    else:
        return "standard"


async def health_check() -> dict:
    """Check API health for all configured providers."""
    print(f"\n{'=' * 60}\nLLM API Health Check\n{'=' * 60}\nTesting {len(MODELS)} providers...\n")

    results = {}
    completed = 0
    lock = asyncio.Lock()
    google_auth = get_google_auth_method()

    async def check(provider: str, model_id: str, name: str):
        nonlocal completed
        extra = list_google_models() if provider == "google" else []
        print(f"  [{name}] Testing...", flush=True)
        try:
            content, duration, in_tok, out_tok, _ = await call_llm(provider, model_id, "Say 'OK'.", max_tokens=1024, timeout=60, use_web_search=False)
            reasoning = _get_reasoning_mode(provider, model_id)
            result = (name, True, f"{duration:.1f}s", "", extra, in_tok, out_tok, reasoning, provider, content)
        except Exception as e:
            msg = str(e)
            if "<html>" in msg.lower():
                msg = "401 Auth error"
            result = (name, False, "", msg[:100], extra, 0, 0, "", provider, "")

        # Print result immediately
        async with lock:
            completed += 1
            is_google = provider == "google"
            auth_info = f" [{google_auth}]" if is_google else ""
            if result[1]:
                tok_str = f"in={result[5]:,}/out={result[6]:,}" if result[5] or result[6] else "tokens=N/A"
                print(f"  [{completed}/{len(MODELS)}] [OK] {name} ({result[2]}, {tok_str}){auth_info}")
                print(f"           Mode: {result[7]}")
                # Show response content (truncated)
                response_preview = result[9][:100].replace('\n', ' ') if result[9] else "(empty)"
                print(f"           Response: {response_preview}")
            else:
                print(f"  [{completed}/{len(MODELS)}] [FAIL] {name}: {result[3]}{auth_info}")
            if extra:
                print(f"           Google models: {len(extra)} available")
        return result

    responses = await asyncio.gather(*[check(p, m, n) for p, m, n in MODELS])

    working = 0
    for name, ok, dur, error, extra, in_tok, out_tok, reasoning, prov, content in responses:
        results[name] = {
            "success": ok,
            "message": error or f"OK ({dur})",
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "reasoning_mode": reasoning
        }
        working += ok

    print(f"\n{'=' * 60}\nResult: {working}/{len(MODELS)} providers OK\n{'=' * 60}")
    return results
