"""Actor MLP — exact mirror of ext/maniskill_baselines/.../bc.py:192-206.

Isolated in its own module so ``model.py`` (registry entry-point) can re-export
both :class:`Actor` and :class:`PLModel` without circular imports.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """3-layer MLP policy head: state_dim -> 256 -> ReLU -> 256 -> ReLU -> action_dim.

    Mirrors ``bc.py:192-206`` exactly. Uses PyTorch default init (Kaiming uniform
    for ``nn.Linear``). No explicit init override.

    Args:
        state_dim: Flattened observation dimension (35 for PushCube-v1).
        action_dim: Action dimension (4 for pd_ee_delta_pos: 3-DOF EE delta + 1 gripper).
    """

    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            state: Flattened observation tensor, shape ``(B, state_dim)`` float32.

        Returns:
            Action tensor, shape ``(B, action_dim)`` float32.
        """
        return self.net(state)
