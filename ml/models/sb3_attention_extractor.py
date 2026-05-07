"""SB3-compatible feature extractor that wraps the project's LSTM + attention encoder.

The default ``MlpPolicy`` flattens the (seq_len, state_dim) observation and
ignores the temporal structure entirely. This module bridges the project's
``TemporalAttention`` design into the SB3 training loop so that DQN/PPO/A2C
all train through the same attention architecture used in the custom
``AttentionDQNAgent`` — eliminating the dual-implementation problem.

Usage::

    from stable_baselines3 import DQN
    from ml.models.sb3_attention_extractor import (
        AttentionFeatureExtractor,
        attention_policy_kwargs,
    )

    agent = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=attention_policy_kwargs(
            seq_len=12, state_dim=env.state_dim, lstm_hidden=64
        ),
        ...
    )
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ml.models.attention_qnetwork import TemporalAttention


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """Encode (seq_len, state_dim) observations via LSTM + multi-head attention.

    Output is the attention-pooled vector at the final timestep of the sequence,
    which SB3's downstream MLP heads consume as ``features``.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 64,
        lstm_hidden: int = 64,
        attention_heads: int = 4,
        dropout: float = 0.1,
    ):
        if features_dim != lstm_hidden:
            # Keep the contract simple: features_dim == lstm_hidden
            features_dim = lstm_hidden
        super().__init__(observation_space, features_dim=features_dim)

        if len(observation_space.shape) != 2:
            raise ValueError(
                "AttentionFeatureExtractor expects a 2D observation space "
                f"(seq_len, state_dim), got shape {observation_space.shape}"
            )

        seq_len, state_dim = observation_space.shape
        self.seq_len = int(seq_len)
        self.state_dim = int(state_dim)

        self.feature_embed = nn.Linear(self.state_dim, lstm_hidden)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,  # single layer
        )
        self.pos_embedding = nn.Embedding(self.seq_len, lstm_hidden)
        self.attention = TemporalAttention(
            hidden_dim=lstm_hidden,
            num_heads=attention_heads,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(lstm_hidden)
        self.activation = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # SB3 sometimes passes flattened obs; reshape if necessary.
        if observations.dim() == 2 and observations.shape[1] == self.seq_len * self.state_dim:
            observations = observations.view(-1, self.seq_len, self.state_dim)
        elif observations.dim() != 3:
            raise ValueError(
                f"Unexpected observation shape {tuple(observations.shape)}; "
                f"expected (batch, {self.seq_len}, {self.state_dim})"
            )

        x = self.activation(self.feature_embed(observations))
        lstm_out, _ = self.lstm(x)

        positions = torch.arange(lstm_out.shape[1], device=lstm_out.device)
        lstm_out = lstm_out + self.pos_embedding(positions).unsqueeze(0)

        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        # Pool by taking the final timestep — same as the custom Q-network does
        pooled = attended[:, -1, :]
        return self.layer_norm(pooled)


def attention_policy_kwargs(
    lstm_hidden: int = 64,
    attention_heads: int = 4,
    dropout: float = 0.1,
    net_arch: list[int] | None = None,
) -> Dict[str, Any]:
    """Build SB3 ``policy_kwargs`` that wires in the attention extractor.

    Defaults yield an extractor of size ``lstm_hidden`` followed by a small
    MLP head ``[lstm_hidden]`` for the Q-values / policy logits.
    """
    if net_arch is None:
        net_arch = [lstm_hidden]
    return dict(
        features_extractor_class=AttentionFeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=lstm_hidden,
            lstm_hidden=lstm_hidden,
            attention_heads=attention_heads,
            dropout=dropout,
        ),
        net_arch=net_arch,
    )
