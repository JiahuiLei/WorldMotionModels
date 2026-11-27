"""ManiskillBSAdapter — bridges UMM's maniskill_bs PLModel to the MS3 evaluator.

Wraps a live ``pl_module`` (during training, via
:class:`eval.ms3.evaluator.MS3EvalCallback`) or a saved pl_state_dict
checkpoint (the richer wrapper format that MS3EvalCallback writes — see
:class:`eval.ms3.evaluator.MS3EvalCallback` docstring "Checkpoint format")::

    {
      "pl_state_dict": ...,
      "global_step": int,
      "success_at_end": float,
      "success_once": float,
      "task": str,
    }

obs_dict keys consumed
----------------------
* ``raw_state`` — ``(state_dim,) float32``. **Required.** All other
  obs_dict keys (``robot.*``, ``objects.*``, ``goal.*``, etc.) are ignored:
  maniskill_bs is a flat-vector BC policy that only consumes the flat state.

Action shape returned
---------------------
``(num_envs, 4)`` ``float32`` numpy array (``pd_ee_delta_pos`` control mode,
action_dim=4: 3-DOF EE delta + 1 gripper). Actions are the raw output of
``pl_module.agent.inference(obs)`` — the env applies control-mode scaling
internally.

Construction patterns
---------------------
1. ``ManiskillBSAdapter(ckpt_path=None, task=..., pl_module=...)`` — live-module
   wrap. Used by :class:`MS3EvalCallback` via ``policy_factory``.
2. :pymeth:`from_pl_module` — classmethod equivalent of (1).
3. :pymeth:`from_pl_ckpt` — classmethod loading a saved pl_state_dict ckpt.
   Requires a ``cfg`` (UMM Config) to re-instantiate ``PLModel(cfg)`` before
   loading weights.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

import numpy as np
import torch

from eval.ms3.evaluator import task_action_dim
from eval.ms3.policies.base import MS3PolicyAdapter

logger = logging.getLogger(__name__)


class ManiskillBSAdapter(MS3PolicyAdapter):
    """MS3 adapter for a UMM ``maniskill_bs`` ``PLModel``.

    Args:
        ckpt_path: Optional path to a saved pl_state_dict ckpt. ``None`` when
            wrapping a live ``pl_module`` in-training.
        task: MS3 task id, e.g. ``"PushCube-v1"``.
        pl_module: The live :class:`umm.models.maniskill_bs.pl_model.PLModel`.
            Required for the in-training / live-wrap construction path.
    """

    #: Convention mapping MS3 task id → canonical config YAML path.
    #: Used by :pymeth:`from_ckpt_cli` (and the CLI evaluator path) to
    #: auto-locate the UMM cfg when the caller only has (ckpt_path, task).
    _TASK_CFG_CONVENTION: dict[str, str] = {
        "PushCube-v1": "configs/maniskill_bs/pushcube_100.yaml",
        "PickCube-v1": "configs/maniskill_bs/pickcube_100.yaml",
        "StackCube-v1": "configs/maniskill_bs/stackcube_100.yaml",
        "PegInsertionSide-v1": "configs/maniskill_bs/peginsertionside_100.yaml",
    }

    def __init__(
        self,
        ckpt_path: Optional[str],
        task: str,
        pl_module: Any = None,
        cfg_path: Optional[str] = None,
    ):
        super().__init__(ckpt_path=ckpt_path, task=task)
        # CLI-compatible path: `adapter_cls(ckpt_path=..., task=...)` with
        # no pl_module provided — auto-load from ckpt using either
        # `cfg_path` (explicit override) or the task→cfg convention above.
        # Required so the evaluator's CLI at `evaluator.py:968` can
        # construct ManiskillBSAdapter without needing to plumb a cfg arg.
        if pl_module is None and ckpt_path is not None:
            resolved_cfg_path = cfg_path or self._TASK_CFG_CONVENTION.get(task)
            if resolved_cfg_path is None:
                raise ValueError(
                    f"ManiskillBSAdapter: task '{task}' has no cfg convention "
                    f"entry. Pass cfg_path=... explicitly or extend "
                    f"_TASK_CFG_CONVENTION. Supported auto-convention tasks: "
                    f"{sorted(self._TASK_CFG_CONVENTION.keys())}"
                )
            from umm.training_utils.config_helpers import get_config
            from umm.models.maniskill_bs.pl_model import PLModel
            cfg = get_config(["--config", resolved_cfg_path])
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"ManiskillBSAdapter: checkpoint not found: {ckpt_path!r}"
                )
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "pl_state_dict" not in ckpt:
                raise KeyError(
                    f"ManiskillBSAdapter: ckpt '{ckpt_path}' missing "
                    f"'pl_state_dict' key. Expected MS3EvalCallback wrapper "
                    f"format. Found keys: {sorted(ckpt.keys())}"
                )
            pl_module = PLModel(cfg)
            pl_module.load_state_dict(ckpt["pl_state_dict"])
            pl_module.eval()
            logger.info(
                "ManiskillBSAdapter: auto-loaded ckpt=%s via cfg=%s "
                "(global_step=%s, success_at_end=%s, success_once=%s)",
                ckpt_path, resolved_cfg_path,
                ckpt.get("global_step"),
                ckpt.get("success_at_end"),
                ckpt.get("success_once"),
            )
        if pl_module is None:
            raise ValueError(
                "ManiskillBSAdapter: pl_module is required for the live-wrap "
                "path. Either pass pl_module=... (live-module), or supply "
                "ckpt_path=... (auto-load via cfg convention), or use "
                "ManiskillBSAdapter.from_pl_ckpt(ckpt_path, task, cfg)."
            )
        if not hasattr(pl_module, "agent"):
            raise AttributeError(
                "ManiskillBSAdapter: pl_module has no 'agent' attribute. "
                "Expected umm.models.maniskill_bs.pl_model.PLModel with "
                "attribute `agent` (umm.models.maniskill_bs.agent.Agent)."
            )
        self.pl_module = pl_module
        # Respect caller's device placement — do NOT auto-migrate the module.
        # MS3EvalCallback._run_once already handles eval() / no_grad() around
        # us; we just read the device for tensor placement in __call__.
        self._device: torch.device = torch.device(pl_module.device)
        logger.info(
            "ManiskillBSAdapter: wrapped pl_module on device=%s for task=%s",
            self._device,
            self.task,
        )

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------
    @classmethod
    def from_pl_module(
        cls, pl_module: Any, task: str
    ) -> "ManiskillBSAdapter":
        """Wrap a live :class:`PLModel` for in-training MS3 eval.

        Args:
            pl_module: A live ``PLModel`` instance (Lightning hands this to
                ``policy_factory(pl_module)`` inside ``MS3EvalCallback``).
            task: MS3 task id.

        Returns:
            A new :class:`ManiskillBSAdapter` sharing ``pl_module``'s weights.
        """
        return cls(ckpt_path=None, task=task, pl_module=pl_module)

    @classmethod
    def from_pl_ckpt(
        cls, ckpt_path: str, task: str, cfg: Any
    ) -> "ManiskillBSAdapter":
        """Load a saved pl_state_dict ckpt and build an adapter.

        The ckpt format is the wrapper dict written by
        :class:`eval.ms3.evaluator.MS3EvalCallback` (see class docstring
        "Checkpoint format"): the real state_dict lives under the
        ``"pl_state_dict"`` key, not at the top level.

        Args:
            ckpt_path: Path to a ``.pt`` file.
            task: MS3 task id.
            cfg: UMM Config object. Used to re-instantiate ``PLModel(cfg)``
                before loading the state_dict — weights alone cannot
                reconstruct the model.

        Returns:
            A new :class:`ManiskillBSAdapter` with weights loaded on CPU.
            (Caller is responsible for any subsequent ``.to(device)``.)
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"ManiskillBSAdapter.from_pl_ckpt: checkpoint not found: "
                f"{ckpt_path!r}"
            )
        from umm.models.maniskill_bs.pl_model import PLModel

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "pl_state_dict" not in ckpt:
            raise KeyError(
                f"ManiskillBSAdapter.from_pl_ckpt: ckpt '{ckpt_path}' is "
                f"missing the 'pl_state_dict' key. Expected format is the "
                f"MS3EvalCallback wrapper dict "
                f"{{'pl_state_dict', 'global_step', 'success_at_end', "
                f"'success_once', 'task'}}. Found keys: {sorted(ckpt.keys())}"
            )
        pl_module = PLModel(cfg)
        pl_module.load_state_dict(ckpt["pl_state_dict"])
        pl_module.eval()
        logger.info(
            "ManiskillBSAdapter.from_pl_ckpt: loaded %s (global_step=%s, "
            "success_at_end=%s, success_once=%s, task=%s)",
            ckpt_path,
            ckpt.get("global_step"),
            ckpt.get("success_at_end"),
            ckpt.get("success_once"),
            ckpt.get("task"),
        )
        return cls(ckpt_path=ckpt_path, task=task, pl_module=pl_module)

    # ------------------------------------------------------------------
    # MS3PolicyAdapter contract
    # ------------------------------------------------------------------
    def __call__(self, obs_dict_batch: List[dict]) -> np.ndarray:
        """Map a batch of per-env obs dicts to a numpy action batch.

        Args:
            obs_dict_batch: length-``num_envs`` list of observation dicts.
                Only ``raw_state`` (``(state_dim,) float32``) is consumed;
                all other keys are ignored.

        Returns:
            ``np.ndarray`` of shape ``(num_envs, 4)`` dtype ``float32``.
        """
        n = len(obs_dict_batch)
        # (n, state_dim) float32 — KeyError raised if any obs lacks raw_state
        # (per MS3PolicyAdapter contract in base.py).
        state = np.stack(
            [obs["raw_state"] for obs in obs_dict_batch], axis=0
        ).astype(np.float32)

        with torch.no_grad():
            t = torch.from_numpy(state).to(self._device)  # (n, state_dim)
            action_t = self.pl_module.agent.inference(t)  # (n, action_dim)
        action = action_t.detach().cpu().numpy().astype(np.float32)

        expected_action_dim = task_action_dim(self.task)
        if action.shape != (n, expected_action_dim):
            raise RuntimeError(
                f"ManiskillBSAdapter: agent.inference returned shape "
                f"{action.shape}, expected ({n}, {expected_action_dim}) for "
                f"task={self.task!r} (TASK_CONTROL_MODE → action_dim). Is "
                f"the PLModel trained under a different control mode?"
            )
        return action
