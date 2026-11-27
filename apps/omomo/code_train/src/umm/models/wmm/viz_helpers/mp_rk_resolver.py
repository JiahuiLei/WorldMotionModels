"""Non-iterative structural RK resolver for MP manipulator robots.

Wraps `pytorch_kinematics.chain.Chain.batch_rk` (closed-form, cached
plan) to convert world-frame link poses back to joint positions for
any of the 9 MP manipulators. No iteration, no Jacobian, no gradient
descent.

Reference pattern: ``lib_twist.g1_interface.G1Interface.RK()`` at
``ext/lib_twist/src/lib_twist/g1_interface.py:429``.

Example::

    resolver = MPRkResolver()
    q, T_world_root = resolver.resolve(link_poses, 'panda', part_names)
    T_world_all = resolver.fk(q, T_world_root, 'panda', part_names)
"""

import os
import re
import importlib
import logging
from typing import Dict, List, Optional, Tuple

import torch
import pytorch_kinematics as pk

logger = logging.getLogger(__name__)

# Root of vendored URDFs for robots not available via robot_descriptions
_VENDORED_ASSETS_DIR = os.path.join(
    os.path.dirname(__file__), "assets", "manipulator_urdfs"
)

# Per-robot URDF source: ("pkg", package_name) or ("file", path)
URDF_SOURCE: Dict[str, Tuple[str, str]] = {
    "panda":  ("pkg", "panda_description"),
    "ur3":    ("pkg", "ur3_description"),
    "ur5":    ("pkg", "ur5_description"),
    "ur10":   ("pkg", "ur10_description"),
    "iiwa7":  ("pkg", "iiwa7_description"),
    "iiwa14": ("pkg", "iiwa14_description"),
    "fanuc":  ("pkg", "fanuc_m710ic_description"),
    "kinova": ("file", os.path.join(_VENDORED_ASSETS_DIR, "kinova_jaco_fixed.urdf")),
    "widowx": ("file", os.path.join(_VENDORED_ASSETS_DIR, "widowx_arm.urdf")),
}


def _strip_transmission(raw: str) -> str:
    """Strip <transmission> blocks from URDF text.

    ``pytorch_kinematics``'s strict parser rejects ``<transmission>`` blocks
    that lack ``<hardwareInterface>``. These tags are ROS hardware metadata
    and do not affect the kinematic chain, so stripping them is safe.
    """
    return re.sub(
        r"<transmission\b[^>]*>.*?</transmission>", "", raw, flags=re.DOTALL
    )


def _load_urdf_raw(robot: str) -> bytes:
    """Load raw URDF bytes for ``robot``, with transmission-tag strip."""
    if robot not in URDF_SOURCE:
        raise KeyError(
            f"Unknown MP robot {robot!r}. Known: {list(URDF_SOURCE.keys())}"
        )
    kind, v = URDF_SOURCE[robot]
    if kind == "pkg":
        m = importlib.import_module(f"robot_descriptions.{v}")
        path = m.URDF_PATH
    elif kind == "file":
        path = v
    else:
        raise ValueError(
            f"Unknown URDF source kind {kind!r} for robot {robot!r}"
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"URDF file for robot {robot!r} not found at {path}"
        )
    with open(path, "r") as f:
        raw = f.read()
    cleaned = _strip_transmission(raw)
    return cleaned.encode("utf-8")


class MPRkResolver:
    """Per-robot cached non-iterative RK solver for MP manipulators.

    Loads ``pytorch_kinematics.chain.Chain`` per robot on first use and
    caches it. :meth:`resolve` wraps ``Chain.batch_rk`` (closed-form,
    structural); :meth:`fk` wraps ``Chain.batch_fk`` (forward kinematics)
    for deploy rendering.

    Args:
        device: Torch device for cached chains. Defaults to CPU.
        dtype:  Torch dtype for cached chains. Defaults to float32.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = dtype
        self._chains: Dict[str, "pk.chain.Chain"] = {}

    def get_chain(self, robot: str) -> "pk.chain.Chain":
        """Lazy-load + cache chain for ``robot``."""
        if robot not in self._chains:
            raw = _load_urdf_raw(robot)
            chain = pk.build_chain_from_urdf(raw).to(
                dtype=self.dtype, device=self.device
            )
            self._chains[robot] = chain
            logger.info(
                f"MPRkResolver: loaded chain for {robot!r} "
                f"n_joints={chain.n_joints} "
                f"links={chain.get_link_names()[:3]}"
            )
        return self._chains[robot]

    def resolve(
        self,
        link_poses: torch.Tensor,  # (T, M, 4, 4) world-frame link poses
        robot: str,
        part_names: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """RK: world link poses -> (joint positions, world root pose).

        Uses ``Chain.batch_rk``, a closed-form non-iterative structural
        solver that walks the URDF kinematic tree once (cached) and
        extracts joint angles via rotation-trace atan2 (revolute) or
        axis-dot (prismatic). No Jacobian. No iteration.

        Args:
            link_poses: World-frame link transforms. Shape ``(T, M, 4, 4)``.
                Dtype float. Must contain all ``M`` bodies matching
                ``part_names``.
            robot: Robot name (e.g. ``'panda'``). Must be a key in
                :data:`URDF_SOURCE`.
            part_names: Ordered list of length ``M``. Index 0 is the base
                link (= world root for the robot), indices ``1..M-1`` are
                the joint-child links in the same order as the URDF
                kinematic tree.

        Returns:
            ``(q, T_world_root)``:
              - ``q``: Joint positions of shape ``(T, n_joints)``. Revolute
                in rad, prismatic in m. Each joint's angle lies in
                ``(-pi, pi]``.
              - ``T_world_root``: Root (base) world pose of shape
                ``(T, 4, 4)``.
        """
        if link_poses.ndim != 4 or link_poses.shape[-2:] != (4, 4):
            raise ValueError(
                f"link_poses must have shape (T, M, 4, 4); got "
                f"{tuple(link_poses.shape)}"
            )
        if len(part_names) != link_poses.shape[1]:
            raise ValueError(
                f"part_names has length {len(part_names)} but link_poses "
                f"has M={link_poses.shape[1]}"
            )
        chain = self.get_chain(robot)
        # Move to the chain's device/dtype without clobbering caller state
        link_poses_dt = link_poses.to(device=self.device, dtype=self.dtype)
        T_world_root = link_poses_dt[:, 0]                                   # (T, 4, 4)
        T_world_link = link_poses_dt[:, 1:]                                  # (T, M-1, 4, 4)
        T_root_world = torch.inverse(T_world_root)                           # (T, 4, 4)
        T_root_link = torch.einsum(
            "tij,tnjk->tnik", T_root_world, T_world_link
        )                                                                     # (T, M-1, 4, 4)
        joint_child_names = list(part_names[1:])
        q = chain.batch_rk(T_root_link, joint_child_names)                   # (T, n_joints)
        return (
            q.to(dtype=link_poses.dtype),
            T_world_root.to(dtype=link_poses.dtype),
        )

    def fk(
        self,
        q: torch.Tensor,             # (T, n_joints)
        T_world_root: torch.Tensor,  # (T, 4, 4)
        robot: str,
        part_names: List[str],
    ) -> torch.Tensor:
        """FK: joint positions + root pose -> world link poses for all bodies.

        Forward kinematics via ``Chain.batch_fk`` (see
        ``ext/pytorch_kinematics/src/pytorch_kinematics/chain.py:758``),
        then compose with the world-root pose to map back to world frame.
        Returns the full ``(T, M, 4, 4)`` link pose tensor matching
        ``part_names``.

        Args:
            q: Joint positions of shape ``(T, n_joints)``.
            T_world_root: Root world pose of shape ``(T, 4, 4)``.
            robot: Robot name.
            part_names: Ordered list of length ``M``; index 0 is base, rest
                are joint-child link names.

        Returns:
            ``T_world_all``: Shape ``(T, M, 4, 4)``. Index 0 is the root
            pose (= ``T_world_root``), indices ``1..M-1`` are FK-rebuilt
            joint-child link poses in ``part_names[1:]`` order.
        """
        if q.ndim != 2:
            raise ValueError(
                f"q must have shape (T, n_joints); got {tuple(q.shape)}"
            )
        if T_world_root.ndim != 3 or T_world_root.shape[-2:] != (4, 4):
            raise ValueError(
                f"T_world_root must have shape (T, 4, 4); got "
                f"{tuple(T_world_root.shape)}"
            )
        if q.shape[0] != T_world_root.shape[0]:
            raise ValueError(
                f"q batch size {q.shape[0]} != T_world_root batch size "
                f"{T_world_root.shape[0]}"
            )
        chain = self.get_chain(robot)
        q_dt = q.to(device=self.device, dtype=self.dtype)
        root_dt = T_world_root.to(device=self.device, dtype=self.dtype)
        joint_child_names = list(part_names[1:])

        # Chain.batch_fk returns (T_root_link, key_bodies) where
        # T_root_link has shape (B, K, 4, 4) in the order of key_bodies.
        T_root_link, _ = chain.batch_fk(q_dt, joint_child_names)             # (T, M-1, 4, 4)
        T_world_link = torch.einsum(
            "tij,tnjk->tnik", root_dt, T_root_link
        )                                                                     # (T, M-1, 4, 4)
        T_world_all = torch.cat(
            [root_dt.unsqueeze(1), T_world_link], dim=1
        )                                                                     # (T, M, 4, 4)
        return T_world_all.to(dtype=q.dtype)
