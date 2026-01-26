"""
PeerRank - LLM Peer Evaluation System

Models generate questions, answer them with web search, cross-evaluate each other's
responses, and produce a ranked report with bias analysis.
"""

from . import config
from . import models
from . import providers

__version__ = "1.0.0"
