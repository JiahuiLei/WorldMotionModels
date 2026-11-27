"""PyTorch Lightning module for the maniskill_bs BC policy.

Mirrors the official ManiSkill3 BC training recipe (``bc.py``):

- Loss: MSE between predicted action and ground-truth action (bc.py:323).
- Optimizer: plain ``torch.optim.Adam`` (bc.py:313); constant LR, no scheduler.
- No gradient clipping at model level (trainer sets ``gradient_clip_val: 0.0``).
"""

from typing import Any, Dict, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from umm.models.maniskill_bs.agent import Agent


class PLModel(pl.LightningModule):
    """Lightning wrapper around the BC :class:`Agent`.

    Config keys read:
        - ``cfg.model.state_dim`` (int): flattened obs dim.
        - ``cfg.model.action_dim`` (int): action dim (4 for pd_ee_delta_pos).
        - ``cfg.max_steps`` (int): used for the lr_decay_iters sanity assert.
        - ``cfg.optim.lr`` (float): Adam learning rate (3e-4 in official BC).
        - ``cfg.optim.betas`` (list[float]): Adam betas (default [0.9, 0.999]).
        - ``cfg.optim.eps`` (float): Adam eps (default 1e-8).
        - ``cfg.optim.weight_decay`` (float): Adam weight decay (default 0.0).
        - ``cfg.optim.lr_decay_iters`` (int): asserted <= ``cfg.max_steps``.

    Batch contract:
        Batch is a ``(obs, action)`` tuple of tensors with shapes
        ``(B, state_dim)`` and ``(B, action_dim)``, both float32. This mirrors
        the official BC dataloader (``bc.py:319`` ``obs, action, _ = batch`` —
        we drop ``done`` at the dataset layer since BC never uses it).
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Fail loud if lr_decay_iters > max_steps (UMM PMEM landmine: LR would
        # decay past the end-of-training, never reaching min_lr). For BC we
        # want a constant-LR schedule anyway, but the assert is kept as a
        # belt-and-suspenders against misconfiguration.
        assert cfg.optim.lr_decay_iters <= cfg.max_steps, (
            f"cfg.optim.lr_decay_iters ({cfg.optim.lr_decay_iters}) must be "
            f"<= cfg.max_steps ({cfg.max_steps}) — prevents LR-never-decays "
            f"misconfiguration."
        )

        self.agent = Agent(cfg)

        # Save hparams for TB logger / checkpoints. Only the plain-dict view of
        # cfg is saved (Config is dict-subclass, safe for Lightning).
        self.save_hyperparameters({"cfg": dict(cfg)})

    # -------------------------------------------------------------------
    # Batch unpack helper
    # -------------------------------------------------------------------
    @staticmethod
    def _unpack_batch(batch: Union[Tuple, Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract ``(obs, action)`` from either a tuple or a dict batch."""
        if isinstance(batch, (tuple, list)):
            obs, action = batch[0], batch[1]
        elif isinstance(batch, dict):
            obs, action = batch["obs"], batch["action"]
        else:
            raise TypeError(
                f"Unsupported batch type for maniskill_bs PLModel: {type(batch)}. "
                "Expected (obs, action) tuple/list or {'obs', 'action'} dict."
            )
        return obs, action

    # -------------------------------------------------------------------
    # Lightning hooks
    # -------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """MSE loss between predicted and ground-truth action.

        Args:
            batch: ``(obs, action)`` tuple with shapes ``(B, state_dim)`` and
                ``(B, action_dim)`` (both float32).
            batch_idx: Trainer-provided batch index (unused).

        Returns:
            Scalar MSE loss tensor.
        """
        obs, action = self._unpack_batch(batch)  # (B, S), (B, A)
        pred = self.agent.forward(obs)           # (B, A)
        loss = F.mse_loss(pred, action)
        # UMM convention: metric name "<metric>/train" mirrors ms_debug.yaml
        # monitor conventions ("total_loss/train") so ckpt_monitor slots in.
        self.log("total_loss/train", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Identical MSE on the val split.

        Args:
            batch: Same structure as training_step.
            batch_idx: Trainer-provided batch index (unused).

        Returns:
            Scalar MSE loss tensor.
        """
        obs, action = self._unpack_batch(batch)  # (B, S), (B, A)
        pred = self.agent.forward(obs)           # (B, A)
        loss = F.mse_loss(pred, action)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Plain Adam, constant LR. Mirrors bc.py:313.

        Returns:
            A single ``torch.optim.Adam`` optimizer (no scheduler).
        """
        o = self.cfg.optim
        betas = tuple(o.get("betas", [0.9, 0.999]))
        eps = float(o.get("eps", 1e-8))
        wd = float(o.get("weight_decay", 0.0))
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(o.lr),
            betas=betas,
            eps=eps,
            weight_decay=wd,
        )
        return optimizer
