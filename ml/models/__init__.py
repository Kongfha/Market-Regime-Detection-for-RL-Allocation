"""Models for regime detection and Q-learning."""

from .attention_qnetwork import TemporalAttentionQNetwork

try:
    from .regime_detector import GaussianHMMRegimeDetector
except ModuleNotFoundError as exc:
    if exc.name != "hmmlearn":
        raise
    GaussianHMMRegimeDetector = None

__all__ = [
    "GaussianHMMRegimeDetector",
    "TemporalAttentionQNetwork",
]
