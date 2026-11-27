"""Agent wrapper for the maniskill_bs Actor.

Provides ``forward`` and ``inference`` with identical signatures, per UMM
"Inference Must Mirror Forward" rule.
"""

import torch
import torch.nn as nn

from umm.models.maniskill_bs.actor import Actor


class Agent(nn.Module):
    """Thin wrapper around :class:`Actor`.

    Args:
        cfg: UMM Config object. Reads ``cfg.model.state_dim`` and
            ``cfg.model.action_dim``.
    """

    def __init__(self, cfg):
        super().__init__()
        state_dim = int(cfg.model.state_dim)
        action_dim = int(cfg.model.action_dim)
        self.actor = Actor(state_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute action from observation.

        Args:
            obs: Observation tensor, shape ``(B, state_dim)`` float32.

        Returns:
            Predicted action tensor, shape ``(B, action_dim)`` float32.
        """
        return self.actor(obs)

    def inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Identical to :meth:`forward`. Kept as a distinct entry-point so
        downstream eval/viz code can call ``agent.inference(obs)`` without
        ``hasattr`` guards (per UMM rules).

        Args:
            obs: Observation tensor, shape ``(B, state_dim)`` float32.

        Returns:
            Predicted action tensor, shape ``(B, action_dim)`` float32.
        """
        return self.actor(obs)
