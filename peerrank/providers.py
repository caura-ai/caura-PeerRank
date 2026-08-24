"""
providers.py - LLM provider implementations for PeerRank.ai
"""

import asyncio
import os
import re
import time

import httpx

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from google import genai
from xai_sdk import AsyncClient as XaiAsyncClient
from xai_sdk.chat import user as xai_user
from mistralai import Mistral

from .config import (
    MODELS,
    MAX_TOKENS_ANSWER, MAX_TOKENS_DEEPSEEK,
    DEFAULT_TIMEOUT, MAX_RETRIES, RETRY_DELAY,
    TEMPERATURE_DEFAULT, MODEL_TEMPERATURE_OVERRIDES,
    MODEL_TIMEOUT_OVERRIDES,
    GOOGLE_SERVICE_ACCOUNT_FILE, GOOGLE_PROJECT_ID, GOOGLE_LOCATION,
    GOOGLE_THINKING_BUDGET,
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


async def close_clients():
    """Properly close all async clients to avoid event loop cleanup errors.

    Call this at the end of async operations, before the event loop closes.
    Prevents 'Event loop is closed' errors from httpx cleanup tasks.
    """
    global _clients
    for key, client in list(_clients.items()):
        try:
            # OpenAI/Anthropic expose close(); httpx.AsyncClient exposes aclose()
            if hasattr(client, 'aclose'):
                await client.aclose()
            elif hasattr(client, 'close'):
                await client.close()
        except Exception:
            pass  # Ignore errors during cleanup
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


def _get_httpx_client(timeout: int) -> httpx.AsyncClient:
    cache_key = f"httpx:{timeout}"
    if cache_key not in _clients:
        _clients[cache_key] = httpx.AsyncClient(timeout=timeout)
    return _clients[cache_key]


def _get_mistral_client(api_key: str) -> Mistral:
    if "mistral" not in _clients:
        _clients["mistral"] = Mistral(api_key=api_key)
    return _clients["mistral"]


# Provider implementations
async def _call_openai(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                       use_web_search: bool, response_format: dict | None, temperature: float,
                       grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    client = _get_openai_client(api_key, timeout)
    start = time.time()
    effective_temp = _get_temperature(model, temperature)

    # Build messages with optional grounding context
    messages = []
    if grounding_text:
        messages.append({"role": "system", "content":
            f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{grounding_text}"})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": model, "messages": messages,
              "temperature": effective_temp, "max_completion_tokens": max_tokens}
    if response_format:
        kwargs["response_format"] = response_format
    response = await client.chat.completions.create(**kwargs)
    content = response.choices[0].message.content

    # Extract token usage from response
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage') and response.usage:
        input_tokens = getattr(response.usage, 'prompt_tokens', 0)
        output_tokens = getattr(response.usage, 'completion_tokens', 0)

    return (content.strip() if content else "", time.time() - start, input_tokens, output_tokens, 0.0)


async def _call_anthropic(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                          use_web_search: bool, response_format: dict | None, temperature: float,
                          grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    client = _get_anthropic_client(api_key, timeout)
    effective_temp = _get_temperature(model, temperature)
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": effective_temp}

    # Inject grounding text as system context
    if grounding_text:
        kwargs["system"] = f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{grounding_text}"

    start = time.time()
    response = await client.messages.create(**kwargs)
    content = "".join(block.text for block in response.content if block.type == "text")

    # Extract token usage
    input_tokens = getattr(response.usage, 'input_tokens', 0)
    output_tokens = getattr(response.usage, 'output_tokens', 0)

    return (content.strip(), time.time() - start, input_tokens, output_tokens, 0.0)


async def _call_google(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                       use_web_search: bool, response_format: dict | None, temperature: float,
                       grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    client = _get_google_client()

    actual_model = model

    # Limit thinking budget to save tokens (default can be very high)
    effective_max_tokens = max_tokens
    config = {
        "temperature": temperature,
        "max_output_tokens": effective_max_tokens,
    }
    if GOOGLE_THINKING_BUDGET is not None:
        config["thinking_config"] = {"thinking_budget": GOOGLE_THINKING_BUDGET}

    # Inject grounding text into prompt (Google doesn't support system messages in same way)
    effective_prompt = prompt
    if grounding_text:
        effective_prompt = f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{grounding_text}\n\n---\n\n{prompt}"

    start = time.time()
    response = await client.aio.models.generate_content(model=actual_model, contents=effective_prompt, config=config)
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
        print("      [WARN] Google response truncated (MAX_TOKENS), using partial content", flush=True)

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


# Web grounding search implementations
async def tavily_search(query: str, max_results: int = 5) -> tuple[str, float, bool]:
    """Tavily search. Returns (search_results, duration_seconds, success)"""
    import aiohttp
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("      [Tavily: TAVILY_API_KEY not set]")
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


async def serpapi_search(query: str, max_results: int = 5) -> tuple[str, float, bool]:
    """SerpAPI search. Returns (search_results, duration_seconds, success)"""
    import aiohttp
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        print("      [SerpAPI: SERPAPI_KEY not set]")
        return "", 0.0, False

    start = time.time()
    try:
        params = {
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "num": max_results,
        }
        async with aiohttp.ClientSession() as session:
            async with session.get("https://serpapi.com/search",
                params=params,
                timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"      [SerpAPI HTTP {resp.status}: {error_text[:200]}]")
                    return "", time.time() - start, False
                data = await resp.json()

        # Format results as natural prose
        parts = []

        # Answer box if available
        if data.get("answer_box"):
            answer = data["answer_box"]
            if answer.get("answer"):
                parts.append(answer["answer"])
            elif answer.get("snippet"):
                parts.append(answer["snippet"])

        # Knowledge graph if available
        if data.get("knowledge_graph"):
            kg = data["knowledge_graph"]
            if kg.get("description"):
                parts.append(kg["description"])

        # Organic results
        for r in data.get("organic_results", [])[:max_results]:
            snippet = r.get('snippet', '')[:250]
            if snippet:
                parts.append(snippet)

        return " ".join(parts), time.time() - start, True
    except Exception as e:
        print(f"      [SerpAPI error: {e}]")
        return "", time.time() - start, False


async def web_grounding_search(query: str, max_results: int = 5) -> tuple[str, float, bool]:
    """Universal web grounding search using configured provider.

    Returns (search_results, duration_seconds, success)
    """
    from .config import get_web_grounding_provider
    provider = get_web_grounding_provider()

    if provider == "serpapi":
        return await serpapi_search(query, max_results)
    else:  # tavily (default)
        return await tavily_search(query, max_results)


def _make_openai_caller(base_url: str, max_tokens_limit: int | None = None,
                        supports_response_format: bool = True):
    """Factory for OpenAI-compatible API callers."""
    async def _call(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                    use_web_search: bool, response_format: dict | None, temperature: float,
                    grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
        client = _get_openai_client(api_key, timeout, base_url)
        effective_max = min(max_tokens, max_tokens_limit) if max_tokens_limit else max_tokens
        effective_temp = _get_temperature(model, temperature)

        # Build messages with optional grounding context
        messages = []
        if grounding_text:
            messages.append({"role": "system", "content":
                f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{grounding_text}"})
        messages.append({"role": "user", "content": prompt})

        kwargs = {"model": model, "messages": messages, "temperature": effective_temp, "max_tokens": effective_max}
        if response_format and supports_response_format:
            kwargs["response_format"] = response_format

        start = time.time()
        response = await client.chat.completions.create(**kwargs)
        duration = time.time() - start

        # Extract token usage
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, 'usage') and response.usage:
            input_tokens = getattr(response.usage, 'prompt_tokens', 0)
            output_tokens = getattr(response.usage, 'completion_tokens', 0)

        return (response.choices[0].message.content.strip(), duration, input_tokens, output_tokens, 0.0)

    return _call


# Grok implementation using xAI SDK (native web search removed for standardized grounding)
async def _call_grok(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                     use_web_search: bool, response_format: dict | None, temperature: float,
                     grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    client = XaiAsyncClient(api_key=api_key)
    effective_temp = _get_temperature(model, temperature)

    # Inject grounding text into prompt
    effective_prompt = prompt
    if grounding_text:
        effective_prompt = f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{grounding_text}\n\n---\n\n{prompt}"

    kwargs = {
        "model": model,
        "temperature": effective_temp,
        "max_tokens": max_tokens,
    }

    chat = client.chat.create(**kwargs)
    chat.append(xai_user(effective_prompt))

    start = time.time()
    try:
        response = await chat.sample()
    except Exception as e:
        error_str = str(e).lower()
        # Translate gRPC errors to helpful messages
        if "permission_denied" in error_str:
            if "content" in error_str:
                raise RuntimeError("Grok content policy rejection - prompt may contain restricted content") from e
            raise RuntimeError("Grok API permission denied - check GROK_API_KEY is valid") from e
        if "unauthenticated" in error_str:
            raise RuntimeError("Grok API authentication failed - check GROK_API_KEY") from e
        if "invalid_argument" in error_str:
            raise RuntimeError(f"Grok API invalid request - {str(e)[:100]}") from e
        if "resource_exhausted" in error_str:
            raise RuntimeError("Grok API rate limit exceeded - try again later") from e
        raise
    duration = time.time() - start

    content = response.content or ""

    # Extract token usage from response
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage') and response.usage:
        input_tokens = getattr(response.usage, 'prompt_tokens', 0) or getattr(response.usage, 'input_tokens', 0) or 0
        output_tokens = getattr(response.usage, 'completion_tokens', 0) or getattr(response.usage, 'output_tokens', 0) or 0

    return (content.strip(), duration, input_tokens, output_tokens, 0.0)


# Mistral implementation (native web search removed for standardized grounding)
async def _call_mistral(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                        use_web_search: bool, response_format: dict | None, temperature: float,
                        grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    client = _get_mistral_client(api_key)
    effective_temp = _get_temperature(model, temperature)

    # Build messages with optional grounding context
    messages = []
    if grounding_text:
        messages.append({"role": "system", "content":
            f"Use this current information to answer accurately. Do not mention searching or sources.\n\n{grounding_text}"})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = await client.chat.complete_async(
        model=model,
        messages=messages,
        temperature=effective_temp,
        max_tokens=max_tokens,
    )

    content = response.choices[0].message.content if response.choices else ""

    # Extract token usage
    input_tokens = 0
    output_tokens = 0
    if hasattr(response, 'usage') and response.usage:
        input_tokens = getattr(response.usage, 'prompt_tokens', 0)
        output_tokens = getattr(response.usage, 'completion_tokens', 0)

    duration = time.time() - start
    return (content.strip() if content else "", duration, input_tokens, output_tokens, 0.0)


# Provider instances (native web search removed - using standardized Tavily grounding)
_call_deepseek = _make_openai_caller("https://api.deepseek.com", max_tokens_limit=MAX_TOKENS_DEEPSEEK)
_call_together = _make_openai_caller("https://api.together.xyz/v1")
_call_kimi = _make_openai_caller("https://api.moonshot.ai/v1")
_call_minimax_raw = _make_openai_caller("https://api.minimax.io/v1")

async def _call_minimax(model, prompt, api_key, max_tokens, timeout,
                        use_web_search, response_format, temperature,
                        grounding_text=None):
    """MiniMax wrapper: strip <think> tags from response."""
    content, duration, in_tok, out_tok, reserved = await _call_minimax_raw(
        model, prompt, api_key, max_tokens, timeout,
        use_web_search, response_format, temperature, grounding_text)
    content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)
    return (content.strip(), duration, in_tok, out_tok, reserved)

# Perplexity Agent API (replaces the retired Sonar chat-completions surface).
# Presets select Perplexity's own optimised configuration; a provider-qualified
# id (e.g. "openai/gpt-5.6-sol") overrides the preset's default model.
PERPLEXITY_AGENT_URL = "https://api.perplexity.ai/v1/agent"
PERPLEXITY_PRESETS = {"fast", "low", "medium", "high", "xhigh", "wide-research"}

# Measured 2026-08-24: a preset's search is intrinsic and CANNOT be switched off -
# passing tools=[] alongside preset="medium" still returned live news with [web:N]
# citations (and cost more input tokens, 14k vs 5k). Search is only avoidable by
# passing a raw model id instead of a preset, e.g. model="openai/gpt-5.6-luna",
# which measured in=43 tokens and no web access. So the same caveat that applied to
# Sonar still applies: this participant sees more context than the others.
# None = accept the preset's default tools; a list overrides them (raw models only).
PERPLEXITY_AGENT_TOOLS: list[dict] | None = None


# Preset runs cite their sources inline as [web:3], [fetch:1] etc. No other provider in
# the set emits these, so they act as a style fingerprint that could identify this
# participant during blind evaluation - exactly what shuffle_blind exists to prevent.
# Matches an optional leading space so "... prime. [web:1]" strips cleanly.
_PPLX_CITATION_RE = re.compile(r"[ 	]*\[(?:web|fetch|finance|sandbox)[:_]\d+(?:\s*,\s*\d+)*\]")


def _extract_agent_text(payload: dict) -> str:
    """Collect output_text parts from an Agent API response, skipping reasoning/tool items."""
    parts = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "output_text":
                parts.append(block.get("text") or "")
    return "\n".join(part for part in parts if part)


async def _call_perplexity(model: str, prompt: str, api_key: str, max_tokens: int, timeout: int,
                           use_web_search: bool, response_format: dict | None, temperature: float,
                           grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    """Perplexity Agent API. `model` is either a preset name or a provider-qualified model id."""
    client = _get_httpx_client(timeout)

    body: dict = {
        "input": prompt,
        "temperature": _get_temperature(model, temperature),
        "max_output_tokens": max_tokens,
    }
    body["preset" if model in PERPLEXITY_PRESETS else "model"] = model
    if PERPLEXITY_AGENT_TOOLS is not None:
        body["tools"] = PERPLEXITY_AGENT_TOOLS
    if grounding_text:
        body["instructions"] = ("Use this current information to answer accurately. "
                                f"Do not mention searching or sources.\n\n{grounding_text}")

    start = time.time()
    try:
        response = await client.post(
            PERPLEXITY_AGENT_URL, json=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    except httpx.TimeoutException as e:
        # httpx timeouts stringify to "", which matches neither the retryable nor the
        # non-retryable list in call_llm's classifier - so they would be raised on the
        # first attempt and never retried. Name the condition explicitly.
        raise RuntimeError(f"Perplexity request timeout after {timeout}s") from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Perplexity transport error - {type(e).__name__}: {e}") from e
    duration = time.time() - start

    if response.status_code != 200:
        detail = response.text[:200]
        if response.status_code == 401:
            raise RuntimeError("Perplexity authentication failed (401) - check PERPLEXITY_API_KEY "
                               f"and that the account has credits: {detail}")
        if response.status_code == 429:
            raise RuntimeError(f"Perplexity rate limit exceeded (429): {detail}")
        raise RuntimeError(f"Perplexity API error {response.status_code}: {detail}")

    payload = response.json()
    status = payload.get("status")
    if status and status != "completed":
        raise RuntimeError(f"Perplexity agent run did not complete (status={status})")

    content = _extract_agent_text(payload)
    content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
    # Keep the original if stripping citations would empty the answer, so a citation-only
    # response is not misreported as a blank/refused one by phase 2.
    stripped = _PPLX_CITATION_RE.sub("", content)
    if stripped.strip():
        content = stripped

    usage = payload.get("usage") or {}
    return (content.strip(), duration,
            usage.get("input_tokens") or 0, usage.get("output_tokens") or 0, 0.0)


_PROVIDER_CALLS = {
    "openai": _call_openai, "anthropic": _call_anthropic, "google": _call_google,
    "grok": _call_grok, "deepseek": _call_deepseek, "together": _call_together, "perplexity": _call_perplexity,
    "kimi": _call_kimi, "mistral": _call_mistral, "minimax": _call_minimax,
}


async def call_llm(provider: str, model: str, prompt: str, max_tokens: int = MAX_TOKENS_ANSWER,
                   timeout: int = DEFAULT_TIMEOUT, use_web_search: bool = False,
                   response_format: dict | None = None, retries: int = MAX_RETRIES,
                   temperature: float = TEMPERATURE_DEFAULT,
                   grounding_text: str | None = None) -> tuple[str, float, int, int, float]:
    """
    Call LLM API with automatic retry on transient failures.

    Args:
        grounding_text: Pre-fetched web grounding text to inject as system context.
        use_web_search: Deprecated - kept for API compatibility, not used.

    Returns:
        tuple: (content, duration, input_tokens, output_tokens, reserved)
               The 5th element is reserved (always 0.0) for backwards compatibility.
    """
    api_key = get_api_key(provider)
    call_fn = _PROVIDER_CALLS.get(provider)
    if not call_fn:
        raise ValueError(f"Unknown provider: {provider}")

    # Sanitize prompt for all models equally
    prompt = sanitize_prompt(prompt)

    # Apply model-specific timeout override
    if model in MODEL_TIMEOUT_OVERRIDES:
        timeout = MODEL_TIMEOUT_OVERRIDES[model]

    last_error = None
    for attempt in range(retries):
        try:
            return await call_fn(model, prompt, api_key, max_tokens, timeout, use_web_search, response_format, temperature, grounding_text)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Don't retry errors that will always fail with the same request
            non_retryable = ["max_tokens", "permission denied", "content policy", "authentication failed",
                             "invalid request", "unauthenticated", "invalid_argument", "401", "403"]
            if any(x in error_str for x in non_retryable):
                raise
            retryable = any(x in error_str for x in ["timeout", "rate limit", "429", "500", "502", "503", "504",
                                                      "overloaded", "capacity", "empty response", "resource_exhausted"])
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
