"""Panda forward-kinematics helper for the ManiSkill3 panda_v3 URDF.

Wraps ``pytorch_kinematics.build_chain_from_urdf`` with the Franka Panda URDF
shipped with ``mani_skill`` (``panda_v3.urdf``). The URDF path is resolved at
runtime via ``mani_skill.__file__`` so no absolute conda path is hard-coded.

Usage::

    import torch
    from lib_mani.utils.panda_fk import PandaFK

    fk = PandaFK()
    qpos = torch.zeros(2, 9)                   # (B, 9) Panda: 7 arm + 2 finger
    poses = fk.forward(qpos)                   # dict[link_name, (B, 4, 4)]
    palm = poses["panda_hand"]                 # (B, 4, 4)
    left = poses["panda_leftfinger"]           # (B, 4, 4)
    right = poses["panda_rightfinger"]         # (B, 4, 4)

The qpos column order matches the MS3 schema-v3 convention
(``prepare_data/ms3/schema.py::QPOS_DIM``) and the ``pytorch_kinematics`` joint
parameter order of the panda_v3 URDF — they coincide and require no permutation.
See :pyattr:`PandaFK.JOINT_ORDER` for the authoritative column labels.

This module intentionally exposes ``forward(qpos)`` ONLY — no IK, no
jacobian, no velocity FK. Any additional primitives belong in a separate file.
"""
from __future__ import annotations

import os.path as osp
from typing import Dict, List, Optional

import torch

import pytorch_kinematics as pk


__all__ = ["PandaFK"]


def _default_urdf_path() -> str:
    """Resolve ``panda_v3.urdf`` under the installed ``mani_skill`` package.

    Uses ``mani_skill.__file__`` rather than ``importlib.resources`` because
    ``mani_skill`` does not ship its asset tree under a package-data entry —
    the URDF lives next to the Python code under ``mani_skill/assets/``.

    Returns:
        Absolute filesystem path to ``panda_v3.urdf``.

    Raises:
        FileNotFoundError: if the URDF is not present at the expected location
            (e.g. ``mani_skill`` was installed without its asset tree).
    """
    import mani_skill  # local import: only needed when the default path is used
    ms_file = getattr(mani_skill, "__file__", None)
    if ms_file is None:
        raise RuntimeError(
            "mani_skill package has no __file__ attribute — cannot locate asset tree."
        )
    pkg_root = osp.dirname(osp.abspath(ms_file))
    urdf_path = osp.join(pkg_root, "assets", "robots", "panda", "panda_v3.urdf")
    if not osp.exists(urdf_path):
        raise FileNotFoundError(
            f"panda_v3.urdf not found at {urdf_path!r}. Verify that mani_skill "
            f"is installed with its asset tree (pip install mani_skill)."
        )
    return urdf_path


class PandaFK:
    """Forward-kinematics helper for the Franka Panda (MS3 panda_v3 URDF).

    The chain is built once in :py:meth:`__init__` and cached. All forward
    calls share the same underlying :class:`pytorch_kinematics.Chain`.

    Attributes:
        JOINT_ORDER: Ordered list of the 9 joint names the chain expects.
            The same order as ``qpos`` columns per MS3 schema-v3 (7 arm +
            2 gripper fingers). Consumers MUST verify their qpos columns
            match this order before calling :py:meth:`forward`.
        urdf_path: Absolute path to the URDF that was loaded.
        chain: Underlying :class:`pytorch_kinematics.Chain`.
        link_names: List of every link name the chain exposes
            (includes arm + hand + fingers + auxiliary frames).
    """

    #: Canonical joint order matching MS3 schema-v3 ``qpos[:, 0:9]``:
    #: 7 Panda arm joints + 2 prismatic gripper finger joints.
    JOINT_ORDER: List[str] = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    def __init__(self, urdf_path: Optional[str] = None) -> None:
        """Build and cache the kinematic chain.

        Args:
            urdf_path: Optional override for the URDF path. When ``None``
                (default), resolves to the ``panda_v3.urdf`` shipped with
                ``mani_skill``.

        Raises:
            FileNotFoundError: if ``urdf_path`` does not exist.
            AssertionError: if the loaded chain's joint order does not match
                :py:attr:`JOINT_ORDER` exactly (safety gate).
        """
        self.urdf_path: str = urdf_path if urdf_path is not None else _default_urdf_path()
        if not osp.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path!r}")

        with open(self.urdf_path, "rb") as f:
            urdf_bytes = f.read()
        self.chain = pk.build_chain_from_urdf(urdf_bytes)

        # Verify the chain's joint order matches our schema assumption. If this
        # ever breaks (e.g. URDF upstream reorders joints), FK'd qpos columns
        # would silently map to wrong joints.
        chain_order = list(self.chain.get_joint_parameter_names())
        assert chain_order == self.JOINT_ORDER, (
            f"PandaFK: chain joint order {chain_order!r} does NOT match "
            f"PandaFK.JOINT_ORDER {self.JOINT_ORDER!r}. Refusing to run — "
            f"qpos columns would be silently misaligned."
        )

        self.link_names: List[str] = list(self.chain.get_link_names())

    def forward(self, qpos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute per-link SE(3) matrices in robot-root frame.

        Args:
            qpos: ``(B, 9)`` float tensor with column order matching
                :py:attr:`JOINT_ORDER` (Panda 7 arm + 2 gripper fingers, rad/m).

        Returns:
            Dict mapping link name (str) to ``(B, 4, 4)`` SE(3) matrix
            (float, same dtype and device as ``qpos``). Keys always include
            ``panda_hand``, ``panda_leftfinger``, ``panda_rightfinger``,
            and ``panda_link0`` .. ``panda_link7`` (plus auxiliary frames
            like ``panda_link8``, ``panda_hand_tcp``, ``camera_link``).

        Raises:
            ValueError: if ``qpos`` is not 2-D with last-dim 9.
        """
        if qpos.ndim != 2 or qpos.shape[-1] != 9:
            raise ValueError(
                f"PandaFK.forward expects qpos shape (B, 9); got {tuple(qpos.shape)!r}"
            )

        # pytorch_kinematics Chain lives on whichever device/dtype its params
        # were built with (default: cpu/float32). Move the chain to match qpos
        # to avoid silent cross-device materialization.
        self.chain = self.chain.to(dtype=qpos.dtype, device=qpos.device)

        # forward_kinematics accepts a tensor with columns in JOINT_ORDER.
        out_frames = self.chain.forward_kinematics(qpos)   # dict[str, Transform3d]

        return {
            name: transform.get_matrix()                   # (B, 4, 4)
            for name, transform in out_frames.items()
        }
