"""BCOfficialAdapter — MS3 adapter for vendor bc.py Actor checkpoints.

Wraps the ``Actor`` class from
``ext/maniskill_baselines/examples/baselines/bc/bc.py`` (lines 192-206) and
plugs it into :class:`eval.ms3.policies.base.MS3PolicyAdapter`.

obs_dict keys consumed
----------------------
* ``raw_state`` — ``(state_dim,) float32`` flat state vector.  **Required.**
  All other obs_dict keys are ignored by this adapter (it is a flat-vector BC
  policy and does not use rich fields such as ``robot.ee_pose``).

Action shape returned
---------------------
``(num_envs, 4)`` ``float32`` numpy array (``pd_ee_delta_pos`` control mode,
action_dim=4; matches the canary ``examples.sh:11-14`` / PushCube-v1 protocol).

Checkpoint format expected
--------------------------
bc.py ``save_ckpt`` (bc.py:208-215) saves::

    {"actor": actor.state_dict()}

Pass the ``.pt`` file path as ``ckpt_path``. The ``["actor"]`` key is
mandatory — loading a checkpoint that lacks it raises ``KeyError`` with an
explicit message pointing to bc.py:210-215.

normalize_states kwarg
----------------------
bc.py default is ``normalize_states=False`` (bc.py:60-61), so the evaluator
passes **raw** observations. If a checkpoint was trained with
``normalize_states=True``, instantiate the adapter with::

    BCOfficialAdapter(ckpt_path=..., task=...,
                      state_mean=<scalar|np.ndarray>, state_std=<scalar|np.ndarray>)

The adapter will apply ``(obs - state_mean) / state_std`` element-wise before
the forward pass. **Per vendor bc.py:179-180 ``get_state_stats``** the stats
are GLOBAL scalars (``np.mean(obs)`` / ``np.std(obs)``), NOT per-feature
vectors. Pass python floats or ``np.float32`` scalars to match the vendor
exactly. Broadcasting accepts arrays too, but the vendor-trained ckpt was
normalised with scalars; per-feature stats would produce a different
distribution than what the Actor expects.

DEVIATION note
--------------
This adapter uses ``ckpt_path=None`` to support random-init (smoke-test)
mode — bc.py itself has no equivalent; it always loads a checkpoint.
"""

from __future__ import annotations

import sys
import os
import logging
from typing import List, Optional

import numpy as np
import torch

from eval.ms3.evaluator import TASK_SPEC, task_action_dim
from eval.ms3.policies.base import MS3PolicyAdapter

logger = logging.getLogger(__name__)

# Absolute path to the vendor bc/ directory so we can extend sys.path once.
_BC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../ext/maniskill_baselines/examples/baselines/bc",
    )
)


def _import_actor():
    """Import ``Actor`` from the vendor bc.py via sys.path extension.

    Idempotent: appends ``_BC_DIR`` only if not already present.  The vendor
    module is guarded by ``if __name__ == '__main__'`` (bc.py:218), so the
    import is side-effect-free.
    """
    if _BC_DIR not in sys.path:
        sys.path.append(_BC_DIR)
    from bc import Actor  # noqa: PLC0415  (intentional delayed import)
    return Actor


def _default_device() -> torch.device:
    """Return ``cuda`` if available, else ``cpu``."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCOfficialAdapter(MS3PolicyAdapter):
    """MS3 adapter for a vendor bc.py ``Actor`` checkpoint.

    Args:
        ckpt_path: Path to a ``.pt`` file saved by bc.py ``save_ckpt``
            (format ``{"actor": state_dict}``).  Pass ``None`` to use a
            randomly-initialised Actor (smoke-test / baseline only).
        task: MS3 task id, e.g. ``"PushCube-v1"``.  Must be a key of
            :pydata:`eval.ms3.evaluator.TASK_SPEC`.
        device: Torch device string or ``torch.device``.  Defaults to
            ``cuda`` if available, else ``cpu``.
        state_mean: Optional scalar (python float / ``np.float32``) or
            ``(state_dim,)`` array.  Vendor bc.py:179-180 uses a GLOBAL scalar
            (``np.mean(obs)`` across ALL elements); matching the vendor means
            passing a scalar.  Arrays work via broadcasting but would not
            match a ckpt trained by the official trainer.
        state_std: Optional scalar or ``(state_dim,)`` array.  Same semantics
            as ``state_mean``; must be provided jointly.
    """

    def __init__(
        self,
        ckpt_path: Optional[str],
        task: str,
        device: Optional[str] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ):
        super().__init__(ckpt_path=ckpt_path, task=task)

        if task not in TASK_SPEC:
            raise KeyError(
                f"BCOfficialAdapter: unknown task '{task}'. "
                f"Supported: {sorted(TASK_SPEC.keys())}"
            )

        # Validate / resolve device
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = _default_device()
        logger.info("BCOfficialAdapter: device=%s", self.device)

        # Build Actor — state_dim from TASK_SPEC, action_dim from per-task
        # control mode (protocol-audit item #6). PushCube/Pick/Stack = 4
        # (pd_ee_delta_pos); PegInsertionSide = 7 (pd_ee_delta_pose).
        state_dim: int = TASK_SPEC[task].state_dim
        action_dim: int = task_action_dim(task)
        Actor = _import_actor()
        self.actor = Actor(state_dim, action_dim)

        # Load checkpoint if provided.
        if ckpt_path is not None:
            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(
                    f"BCOfficialAdapter: checkpoint not found: {ckpt_path!r}"
                )
            # weights_only=True is required for safety (disables arbitrary
            # pickle execution). bc.py's ckpt is a plain-tensor dict
            # (bc.py:210-215) so the flag is compatible. The umm env runs
            # torch 2.3.1 where the default is still False; pinning True
            # here is load-bearing, not cosmetic. (PyTorch announced the
            # default flip in 2.4+; pin explicitly so behaviour stays
            # predictable across torch versions.)
            ckpt = torch.load(
                ckpt_path, map_location=self.device, weights_only=True
            )
            if "actor" not in ckpt:
                raise KeyError(
                    f"BCOfficialAdapter: checkpoint '{ckpt_path}' is missing "
                    f"the 'actor' key. Expected format is "
                    f"{{'actor': state_dict}} as saved by bc.py:210-215 "
                    f"save_ckpt(). Found keys: {sorted(ckpt.keys())}"
                )
            # Pre-load sanity check: the ckpt's first and last linear layers
            # must match our Actor's state_dim and action_dim. Fail loud if
            # keys go missing (don't silently skip the check on a future
            # vendor refactor) — bc.py:196-202 is a fixed
            # ``nn.Sequential(Linear, ReLU, Linear, ReLU, Linear)``, yielding
            # `net.0` / `net.2` / `net.4`.
            actor_sd = ckpt["actor"]
            first_w = actor_sd.get("net.0.weight")
            final_w = actor_sd.get("net.4.weight")
            if first_w is None or final_w is None:
                raise KeyError(
                    f"BCOfficialAdapter: ckpt 'actor' state_dict is missing "
                    f"expected keys 'net.0.weight' / 'net.4.weight' "
                    f"(found: {sorted(actor_sd.keys())}). Did the vendor "
                    f"Actor class layout change?  bc.py:192-206 expects the "
                    f"nn.Sequential(Linear→ReLU→Linear→ReLU→Linear) form."
                )
            if first_w.shape[1] != state_dim:
                raise ValueError(
                    f"BCOfficialAdapter: ckpt state_dim mismatch — "
                    f"ckpt 'net.0.weight' has shape {tuple(first_w.shape)} "
                    f"(state_dim={first_w.shape[1]}) but task='{task}' "
                    f"expects state_dim={state_dim} "
                    f"(TASK_SPEC['{task}'].state_dim). Did you load a "
                    f"checkpoint trained on a different task?"
                )
            if final_w.shape[0] != action_dim:
                raise ValueError(
                    f"BCOfficialAdapter: ckpt action_dim mismatch — "
                    f"ckpt 'net.4.weight' has shape {tuple(final_w.shape)} "
                    f"(action_dim={final_w.shape[0]}) but this adapter is "
                    f"built for action_dim={action_dim} (pd_ee_delta_pos). "
                    f"For pd_joint_delta_pos (action_dim=8) ckpts you need "
                    f"a different adapter / different --control-mode."
                )
            self.actor.load_state_dict(ckpt["actor"])
            logger.info("BCOfficialAdapter: loaded actor from %s", ckpt_path)
        else:
            logger.info(
                "BCOfficialAdapter: ckpt_path=None — using random-init Actor "
                "(smoke-test / baseline mode)"
            )

        self.actor.to(self.device)
        self.actor.eval()

        # Optional obs normalisation (only when trained with normalize_states=True)
        if (state_mean is None) != (state_std is None):
            raise ValueError(
                "BCOfficialAdapter: state_mean and state_std must both be "
                "provided or both omitted."
            )
        self._state_mean: Optional[np.ndarray] = (
            np.asarray(state_mean, dtype=np.float32) if state_mean is not None else None
        )
        self._state_std: Optional[np.ndarray] = (
            np.asarray(state_std, dtype=np.float32) if state_std is not None else None
        )

    def __call__(self, obs_dict_batch: List[dict]) -> np.ndarray:
        """Map a batch of per-env obs dicts to a numpy action batch.

        Args:
            obs_dict_batch: length-``num_envs`` list of observation dicts.
                Only ``raw_state`` (``(state_dim,) float32``) is consumed;
                all other keys are ignored.

        Returns:
            ``np.ndarray`` of shape ``(num_envs, 4)`` dtype ``float32``.
        """
        # Stack raw_state vectors: (num_envs, state_dim)
        state = np.stack(
            [obs["raw_state"] for obs in obs_dict_batch], axis=0
        ).astype(np.float32)  # (num_envs, state_dim)

        # Optional normalisation (only when trained with normalize_states=True)
        if self._state_mean is not None:
            state = (state - self._state_mean) / (self._state_std + 1e-8)

        with torch.no_grad():
            t = torch.from_numpy(state).to(self.device)
            action_t = self.actor(t)

        # bc.py:341-342 / README §3: evaluator requires np.ndarray float32
        return action_t.cpu().numpy().astype(np.float32)  # (num_envs, 4)
