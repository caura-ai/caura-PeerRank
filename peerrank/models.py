"""
models.py - Model definitions and pricing for PeerRank.ai
"""

# Models with pricing - Costs verified 2026-10-07 (per 1M input, output tokens in USD)
# Sources: provider pricing pages; grok prices read from the xAI /v1/language-models API
# Standard short-context rates. Cache, batch, long-context, and peak/off-peak tiers are noted in comments and are not what calculate_cost uses.
# peerrank: whether model participates in PeerRank evaluation
ALL_MODELS = [
    {"peerrank": True, "provider": "openai", "model_id": "gpt-6-astra", "name": "gpt-6-astra", "cost": (10.00, 50.00)},  # standard short-context (developers.openai.com/api/docs/pricing, 2026-10-07). The old (4.00, 20.00) was gpt-5.6-sol's promo, copied here by mistake. Long context (>272K prompt) is (20.00, 75.00) - not modelled here
    {"peerrank": False, "provider": "openai", "model_id": "gpt-5.6-terra", "name": "gpt-5.6-terra", "cost": (2.00, 12.00)},  # cut from (2.50, 15.00) on 2026-07-30. Long context: (4.00, 18.00)
    {"peerrank": False, "provider": "openai", "model_id": "gpt-5.6-luna", "name": "gpt-5.6-luna", "cost": (0.20, 1.20)},  # cut from (1.00, 6.00) on 2026-07-30. Long context: (0.40, 1.80)
    {"peerrank": True, "provider": "anthropic", "model_id": "claude-fable-5-1", "name": "claude-fable-5.1", "cost": (10.00, 50.00)},  # API id uses hyphens; display name keeps the 5.1 label. Adaptive thinking always on, so temperature must stay at 1
    {"peerrank": True, "provider": "anthropic", "model_id": "claude-opus-5-5", "name": "claude-opus-5.5", "cost": (4.00, 20.00)},  # was (5.00, 25.00), that is Opus 5; Opus 5.5 is $4/$20 (platform.claude.com models overview, 2026-10-07). Adaptive thinking always on
    {"peerrank": False, "provider": "anthropic", "model_id": "claude-sonnet-5", "name": "claude-sonnet-5", "cost": (2.00, 10.00)},  # (2.00, 10.00) is now the STANDARD price - the 2026-09-01 rise to (3.00, 15.00) was cancelled (docs note claude-sonnet-5-introductory-pricing)
    {"peerrank": True, "provider": "google", "model_id": "gemini-3.8-flash", "name": "gemini-3.8-flash", "cost": (0.75, 3.75)},  # intro pricing thru 2026-12-31; standard is (1.50, 7.50) from 2027-01-01
    {"peerrank": False, "provider": "google", "model_id": "gemini-3.5-flash-lite", "name": "gemini-3.5-flash-lite", "cost": (0.30, 2.50)},  # was (0.15, 0.90) - wrong; verified at ai.google.dev/pricing 2026-08-27
    {"peerrank": True, "provider": "grok", "model_id": "grok-4.7", "name": "grok-4.7", "cost": (2.00, 6.00)},  # xAI /v1/language-models 2026-10-07: 20000/60000 (units of $1e-4 per 1M) = (2.00, 6.00). Long context: (4.00, 12.00)
    {"peerrank": False, "provider": "deepseek", "model_id": "deepseek-v4-pro", "name": "deepseek-v4-pro", "cost": (0.66, 1.98)},  # off-peak cache-miss rate; peak is (1.32, 3.96). Peak = 01:00-04:00 and 06:00-10:00 UTC Mon-Fri
    {"peerrank": True, "provider": "deepseek", "model_id": "deepseek-flash", "name": "deepseek-v4.1-flash", "cost": (0.15, 0.60)},  # API id is deepseek-flash (DeepSeek-V4.1-Flash). deepseek-v4.1-flash is rejected. Off-peak cache-miss (0.15, 0.60); peak (0.30, 1.20)
    {"peerrank": True, "provider": "meta", "model_id": "muse-wonder-1.0", "name": "muse-wonder-1.0", "cost": (4.00, 20.00)},  # Meta Model API standard mode (dev.meta.ai/docs/pricing-rate-limits, 2026-10-13). Preview. Cached input 0.20
    {"peerrank": True, "provider": "meta", "model_id": "muse-spark-1.3", "name": "muse-spark-1.3", "cost": (1.25, 4.25)},  # standard mode; contributor variant is (0.10, 0.20) and is not this id. Cached input 0.15
    {"peerrank": False, "provider": "together", "model_id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "name": "llama-3.3-70b", "cost": (1.04, 1.04)},  # was (0.88, 0.88) - stale; together.ai/models/llama-3-3-70b lists $1.04 in/out (2026-08-27). Llama 4 not available serverless on Together
    {"peerrank": False, "provider": "kimi", "model_id": "kimi-k3", "name": "kimi-k3", "cost": (3.00, 15.00)},  # cache-hit input is 0.30
    {"peerrank": False, "provider": "mistral", "model_id": "mistral-large-latest", "name": "mistral-large", "cost": (0.50, 1.50)},  # -latest is the GA alias, so this is Mistral Large 3 (docs.mistral.ai/inference/pricing, 2026-10-07). Large 4 is preview-only (mistral-large-4-0) at a sale price of (0.68, 2.09) and is not this alias
    {"peerrank": False, "provider": "perplexity", "model_id": "medium", "name": "pplx-agent-medium", "cost": (0.20, 1.20)},  # Agent API preset; bills at its backing model (openai/gpt-5.6-luna) rates. Excludes per-tool fees (~$0.0025/web_search, ~$0.0005/fetch_url) which are NOT tracked
    {"peerrank": False, "provider": "grok", "model_id": "grok-code-fast-1", "name": "grok-code-fast-1", "cost": (1.00, 2.00)},  # alias of grok-build-0.1; xAI API 2026-10-07 prices it at 10000/20000 = (1.00, 2.00). Long context: (2.00, 4.00)
    {"peerrank": False, "provider": "anthropic", "model_id": "claude-haiku-4-5", "name": "claude-haiku-4-5", "cost": (1.00, 5.00)},  # id resolves to claude-haiku-4-5-20251001; there is no Haiku 5 (verified via Models API 2026-08-26)
    {"peerrank": False, "provider": "google", "model_id": "gemini-3.1-pro-preview", "name": "gemini-3.1-pro-preview", "cost": (2.00, 12.00)},  # prompts <=200K; above that it is (4.00, 18.00)
    {"peerrank": False, "provider": "google", "model_id": "gemini-3.5-flash", "name": "gemini-3.5-flash", "cost": (1.50, 9.00)},  # was (0.50, 3.00) - wrong; verified at ai.google.dev/pricing 2026-08-27
    {"peerrank": False, "provider": "google", "model_id": "gemini-3.1-flash-lite", "name": "gemini-3.1-flash-lite", "cost": (0.25, 1.50)},  # was (0.10, 0.40) - wrong; verified at ai.google.dev/pricing 2026-08-27
    {"peerrank": False, "provider": "openai", "model_id": "gpt-5-nano", "name": "gpt-5-nano", "cost": (0.05, 0.40)},
    {"peerrank": False, "provider": "openai", "model_id": "gpt-5-mini", "name": "gpt-5-mini", "cost": (0.25, 2.00)},
    # {"peerrank": False, "provider": "kimi", "model_id": "kimi-k2-0905-preview", "name": "kimi-k2-0905", "cost": (0.60, 2.50)},
]


# # Models with pricing - Arxiv Article cut offFeb 2026
# # peerrank: whether model participates in PeerRank evaluation
# ALL_MODELS = [
#     {"peerrank": True, "provider": "openai", "model_id": "gpt-5.2", "name": "gpt-5.2", "cost": (1.75, 14.00)},
#     {"peerrank": True, "provider": "openai", "model_id": "gpt-5-mini", "name": "gpt-5-mini", "cost": (0.25, 2.00)},
#     {"peerrank": True, "provider": "anthropic", "model_id": "claude-opus-4-5", "name": "claude-opus-4-5", "cost": (5.00, 25.00)},
#     {"peerrank": True, "provider": "anthropic", "model_id": "claude-sonnet-4-5", "name": "claude-sonnet-4-5", "cost": (3.00, 15.00)},
#     {"peerrank": True, "provider": "google", "model_id": "gemini-3-pro-preview", "name": "gemini-3-pro-preview", "cost": (2.00, 12.00)},
#     {"peerrank": True, "provider": "google", "model_id": "gemini-3-flash-preview", "name": "gemini-3-flash-preview", "cost": (0.50, 3.00)},
#     {"peerrank": True, "provider": "grok", "model_id": "grok-4-1-fast", "name": "grok-4-1-fast", "cost": (0.60, 3.00)},
#     {"peerrank": True, "provider": "deepseek", "model_id": "deepseek-chat", "name": "deepseek-chat", "cost": (0.28, 0.42)},
#     {"peerrank": True, "provider": "together", "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "name": "llama-4-maverick", "cost": (0.27, 0.27)},
#     {"peerrank": True, "provider": "perplexity", "model_id": "sonar-pro", "name": "sonar-pro", "cost": (3.00, 15.00)},
#     {"peerrank": True, "provider": "kimi", "model_id": "kimi-k2.5", "name": "kimi-k2.5", "cost": (0.60, 3.00)},
#     {"peerrank": True, "provider": "mistral", "model_id": "mistral-large-latest", "name": "mistral-large", "cost": (2.00, 6.00)},
#     {"peerrank": False, "provider": "openai", "model_id": "gpt-5.1", "name": "gpt-5.1", "cost": (1.25, 10.00)},
#     {"peerrank": False, "provider": "google", "model_id": "gemini-2.5-pro", "name": "gemini-2.5-pro", "cost": (1.25, 10.00)},
#     {"peerrank": False, "provider": "perplexity", "model_id": "sonar-reasoning-pro", "name": "sonar-reasoning-pro", "cost": (2.00, 8.00)},
#     {"peerrank": False, "provider": "grok", "model_id": "grok-code-fast-1", "name": "grok-code-fast-1", "cost": (0.60, 3.00)},
#     {"peerrank": False, "provider": "anthropic", "model_id": "claude-haiku-4-5", "name": "claude-haiku-4-5", "cost": (0.20, 1.00)},
#     {"peerrank": False, "provider": "openai", "model_id": "gpt-5-nano", "name": "gpt-5-nano", "cost": (0.05, 0.4)},
#    # {"peerrank": False, "provider": "kimi", "model_id": "kimi-k2-0905-preview", "name": "kimi-k2-0905", "cost": (0.60, 2.50)},
# ]
