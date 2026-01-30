"""
PeerRank - LLM Peer Evaluation System

Models generate questions, answer them with web search, cross-evaluate each other's
responses, and produce a ranked report with bias analysis.
"""

from . import config  # noqa: F401
from . import models  # noqa: F401
from . import providers  # noqa: F401

__all__ = ["config", "models", "providers"]
__version__ = "1.0.0"
