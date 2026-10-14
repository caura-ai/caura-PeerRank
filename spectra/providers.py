"""
providers.py - the only Spectra module that imports PeerRank

An entry named after a PeerRank model ("claude-opus-5.5") is sent through
peerrank.providers.call_llm with Phase 2's settings (MAX_TOKENS_ANSWER, the default
temperature and its per-model overrides, Argon's thinking level, Grok's reasoning effort),
so a Spectra profile describes the same deployment PeerRank ranks. An entry with
litellm_model goes through LiteLLM instead, for models PeerRank does not carry.

  models:
    - claude-opus-5.5                                  # PeerRank, Phase 2 settings
    - {name: gpt-6-astra, max_tokens: 8192}            # PeerRank, one setting changed
    - {name: other, litellm_model: openai/x, params: {max_tokens: 4096}}   # LiteLLM
"""

import time
from dataclasses import asdict, dataclass

_pr = None
_litellm = None


def _peerrank():
    """Import PeerRank on first use, so LiteLLM-only entries never need its provider SDKs."""
    global _pr
    if _pr is None:
        from peerrank import config, models, providers
        _pr = (config, models, providers)
    return _pr


def _lite():
    global _litellm
    if _litellm is None:
        import litellm
        litellm.drop_params = True  # ignore params a provider doesn't support instead of failing
        _litellm = litellm
    return _litellm


@dataclass
class Reply:
    text: str
    latency_s: float
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None

    def fields(self) -> dict:
        return asdict(self)


def resolve(entry: dict) -> dict:
    """Return the entry with backend, provider and model_id filled in. Raises on an unknown name."""
    e = dict(entry)
    if e.get("litellm_model"):
        e.update(backend="litellm", provider=e["litellm_model"].split("/", 1)[0], model_id=e["litellm_model"])
        return e
    _, models, _ = _peerrank()
    match = next((m for m in models.ALL_MODELS if m["name"] == e["name"]), None)
    if match is None:
        known = ", ".join(m["name"] for m in models.ALL_MODELS)
        raise KeyError(f"{e['name']!r} is not a PeerRank model name (peerrank/models.py). Known: {known}. "
                       f"Give the entry a litellm_model to call it through LiteLLM instead.")
    e.update(backend="peerrank", provider=match["provider"], model_id=match["model_id"])
    return e


def describe(entry: dict) -> dict:
    """The settings a call will use, for run_config.json."""
    e = resolve(entry)
    if e["backend"] == "litellm":
        return {"name": e["name"], "backend": "litellm", "litellm_model": e["model_id"], "params": e.get("params") or {}}
    config, _, providers = _peerrank()
    temp = e.get("temperature", config.TEMPERATURE_DEFAULT)
    out = {"name": e["name"], "backend": "peerrank", "provider": e["provider"], "model_id": e["model_id"],
           "max_tokens": e.get("max_tokens", config.MAX_TOKENS_ANSWER),
           "temperature": config.MODEL_TEMPERATURE_OVERRIDES.get(e["model_id"], temp)}  # Argon drops it in favor of thinking_level
    try:
        out["reasoning"] = providers._get_reasoning_mode(e["provider"], e["model_id"])
    except Exception:
        pass
    return out


def max_concurrency(entry: dict, default: int) -> int:
    """Cap at PeerRank's per-provider limit (Perplexity allows 1, Mistral 2, Google 3)."""
    e = resolve(entry)
    if e["backend"] != "peerrank":
        return default
    config, _, _ = _peerrank()
    return max(1, min(default, config.PROVIDER_CONCURRENCY.get(e["provider"], default)))


async def complete(entry: dict, prompt: str, *, system_prompt: str | None = None,
                   timeout: int = 180, retries: int = 3) -> Reply:
    e = resolve(entry)
    if e["backend"] == "litellm":
        return await _complete_litellm(e, prompt, system_prompt, timeout, retries)
    if system_prompt:
        # call_llm's only system slot is grounding_text, which prepends PeerRank's grounding preface
        raise ValueError(f"{e['name']}: system_prompt is only supported on litellm_model entries")
    config, _, providers = _peerrank()
    kwargs = {"max_tokens": e.get("max_tokens", config.MAX_TOKENS_ANSWER), "timeout": timeout, "retries": retries}
    if "temperature" in e:
        kwargs["temperature"] = e["temperature"]
    text, latency, in_tok, out_tok, _ = await providers.call_llm(e["provider"], e["model_id"], prompt, **kwargs)
    # call_llm reports no finish reason and no separate thinking count; never derive one by subtraction
    return Reply(text=text, latency_s=round(latency, 3), finish_reason=None, input_tokens=in_tok,
                 output_tokens=out_tok, reasoning_tokens=None,
                 cost_usd=config.calculate_cost(e["model_id"], in_tok, out_tok))


async def _complete_litellm(e, prompt, system_prompt, timeout, retries) -> Reply:
    litellm = _lite()
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    kwargs = dict(model=e["model_id"], messages=messages, timeout=timeout, num_retries=retries)
    kwargs.update(e.get("params") or {})
    t0 = time.perf_counter()
    resp = await litellm.acompletion(**kwargs)
    latency = time.perf_counter() - t0
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    cost = None
    try:
        cost = litellm.completion_cost(completion_response=resp)
    except Exception:
        pass
    return Reply(text=choice.message.content or "", latency_s=round(latency, 3),
                 finish_reason=getattr(choice, "finish_reason", None),
                 input_tokens=getattr(usage, "prompt_tokens", None),
                 output_tokens=getattr(usage, "completion_tokens", None),
                 reasoning_tokens=getattr(details, "reasoning_tokens", None), cost_usd=cost)


async def close():
    """Close PeerRank's cached async clients before the event loop ends."""
    if _pr is not None:
        await _pr[2].close_clients()
