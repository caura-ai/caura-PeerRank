"""
models.py - Model definitions and pricing for PeerRank.ai
"""

# Models with pricing - Costs updated Jan 2026
# peerrank: whether model participates in PeerRank evaluation
ALL_MODELS = [
    {"peerrank": True, "provider": "openai", "model_id": "gpt-5.2", "name": "gpt-5.2", "cost": (1.75, 14.00)},
    {"peerrank": True, "provider": "openai", "model_id": "gpt-5-mini", "name": "gpt-5-mini", "cost": (0.25, 2.00)},
    {"peerrank": True, "provider": "anthropic", "model_id": "claude-opus-4-5", "name": "claude-opus-4-5", "cost": (5.00, 25.00)},
    {"peerrank": True, "provider": "anthropic", "model_id": "claude-sonnet-4-5", "name": "claude-sonnet-4-5", "cost": (3.00, 15.00)},
    {"peerrank": True, "provider": "google", "model_id": "gemini-3-pro-preview", "name": "gemini-3-pro-preview", "cost": (2.00, 12.00)},
    {"peerrank": True, "provider": "google", "model_id": "gemini-3-flash-preview", "name": "gemini-3-flash-preview", "cost": (0.50, 3.00)},
    {"peerrank": True, "provider": "grok", "model_id": "grok-4-1-fast", "name": "grok-4-1-fast", "cost": (0.60, 3.00)},
    {"peerrank": True, "provider": "deepseek", "model_id": "deepseek-chat", "name": "deepseek-chat", "cost": (0.28, 0.42)},
    {"peerrank": True, "provider": "together", "model_id": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "name": "llama-4-maverick", "cost": (0.27, 0.27)},
    {"peerrank": True, "provider": "perplexity", "model_id": "sonar-pro", "name": "sonar-pro", "cost": (3.00, 15.00)},
    {"peerrank": True, "provider": "kimi", "model_id": "kimi-k2-0905-preview", "name": "kimi-k2-0905", "cost": (0.60, 2.50)},
    {"peerrank": True, "provider": "mistral", "model_id": "mistral-large-latest", "name": "mistral-large", "cost": (2.00, 6.00)},
]
