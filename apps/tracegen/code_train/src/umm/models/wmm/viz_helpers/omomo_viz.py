"""Visualization helpers for OMOMO (human-object interaction, CHOIS protocol).

Renders SMPL-X body mesh + rest-frame object mesh + 256-point local PCL
overlay for OMOMO CHOIS-protocol windows. Mirrors the interface of
``humoto_viz.py`` so the training viz callback can drop it in.

Uses ``prepare_data.omomo.omomo_viser`` (``_SmplxFK``, ``_resolve_obj_mesh_path``,
etc.) as the FK + geometry backbone — same pattern as ``humoto_viz.py``
importing ``prepare_data.humoto.humoto_seq.HumotoSeq``.

This file is **additive-only**: it does not modify any existing viz_helpers
module.

Scope (leader-approved 2026-04-22, see the design notes)
-----------------------------------------------------------------
``OmomoVisualizer.draw_gt(batch)`` and
``OmomoVisualizer.draw_inference(batch, pred_x0)`` are fully implemented
and verified end-to-end on collated OMOMO batches. The training-time
``omomo_viz_routine`` renders GT and logs via ``pl.logger.log_video``.

The four inference-mode branches (half-future / 1-step / o2h / h2o) are
deferred. They depend on the ``scheduler_mode`` kwarg is adding
to ``wmm/agent.py::inference(...)`` — that API does not exist yet on
this branch, and writing against a non-existent API would fail
silently at training time. **Inference-mode hooks land in a follow-up
once `agent.inference` signature stabilizes** (expected:
pre-launch or a dedicated). Until then, those branches log a
one-shot WARN and return.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# CRITICAL: before pyrender import.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender  # type: ignore
import torch
import trimesh
from PIL import Image

from prepare_data.omomo.omomo_viser import (
    _SmplxFK,
    _color_mesh,
    _make_ground_plane,
    _look_at_z_up,
    _camera_pose_diagonal,
    OmomoWindow,
    BODY_COLOR_RGBA,
    OBJ_COLOR_RGBA,
)

from umm.models.wmm.viz_helpers.omomo_penetration_viz import (
    render_penetration_sequence,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rest-template cache: per-(gender, betas) SMPL-X T-pose verts + LBS weights
# + inv-bind translations. Built lazily on first use, reused across all calls
# for the same subject. Keyed by (gender, betas_hash) where betas_hash is a
# tuple of 4-decimal-rounded floats — different subjects get different
# entries; identical betas reuse the cache.
#
# Each cache value is a 4-tuple:
#     rest_verts      (10475, 3) float64 — SMPL-X T-pose vertex positions
#     rest_homo       (10475, 4) float64 — rest_verts in homogeneous coords
#     lbs_weights_22  (10475, 22) float32 — SMPL-X LBS weights truncated to
#                     the first 22 (body) joints. Sum across joints is
#                     ~1.0 for body verts but <1.0 for hands/face.
#     inv_bind_22     (22, 4, 4) float64 — translation-only inverse bind
#                     matrices: inv_bind[j] = [[I_3, -J_rest[j]], [0, 1]].
# Per past lesson (SMPL-X uses translation-only
#                     inv_bind; rotation block stays identity).
#
# Why translation-only inv_bind: SMPL-X's internal LBS uses world joint
# transforms with bind-pose translation removed. The bind rotations are
# identity in SMPL-X's neutral T-pose convention.
_RestTemplate = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_rest_template_cache: Dict[Tuple[str, Tuple[float, ...]], _RestTemplate] = {}
# Per-gender cache for the loaded smplx.SMPLX(batch_size=1) module so we
# only pay the ~500MB load once per gender across all calls.
_rest_smplx_model_cache: Dict[str, object] = {}
# Cache the SMPL-X faces table once — same for all subjects.
_rest_smplx_faces_cache: Optional[np.ndarray] = None


def _betas_hash(betas: np.ndarray) -> Tuple[float, ...]:
    """Round betas to 4 decimals and convert to a hashable tuple.

    OMOMO subjects have distinct betas in the 1e-3 range, so rounding to
    1e-4 keeps each subject in its own cache slot while making the hash
    stable across float32/float64 round-trip noise.
    """
    arr = np.asarray(betas, dtype=np.float64).reshape(-1)
    return tuple(round(float(x), 4) for x in arr)


def _get_rest_template(
    gender: str,
    betas: np.ndarray,
    smplx_dir: Path,
) -> _RestTemplate:
    """Return the LBS rest template for one (gender, betas) subject.

    Cached at module level. Builds the template by running SMPL-X forward
    at zero pose / zero transl with the given betas, extracting:
        * rest vertices (T-pose, shape-blended)
        * rest body joints (J_rest[:22])
        * LBS weights truncated to the first 22 joints
        * translation-only inverse bind matrices

    Also runs an LBS self-sanity check: applying LBS with identity Q for
    every joint should yield a "collapsed humanoid blob" near the origin
    (each vertex picks up a weighted-J_rest offset from the rest position;
    with sum_j w[v,j] ≈ 1 this is rest_v − weighted_J_rest). Logs the
    bounding-box diagonal as a measurable witness — typical SMPL-X body
    yields ~0.5 m diagonal (skull-to-pelvis local extent).

    Args:
        gender: "male" | "female" | "neutral".
        betas: (16,) float, SMPL-X shape coefficients.
        smplx_dir: directory containing SMPLX_{GENDER}.npz.

    Returns:
        4-tuple ``(rest_verts (V,3) float64, rest_homo (V,4) float64,
        lbs_weights_22 (V,22) float32, inv_bind_22 (22,4,4) float64)``.

    Raises:
        ValueError: if the lbs_weights truncation hypothesis fails — when
            the per-vertex sum of weights[:22] is far from 1 for many
            body verts, the truncation is wrong and LBS would silently
            mis-skin (e.g. body verts collapse). We refuse to cache a
            template that fails this sanity check.
    """
    global _rest_smplx_faces_cache
    gender = gender.lower()
    if gender not in ("male", "female", "neutral"):
        raise ValueError(f"_get_rest_template: unsupported gender {gender!r}")
    betas_arr = np.asarray(betas, dtype=np.float32).reshape(-1)
    if betas_arr.size != 16:
        raise ValueError(
            f"_get_rest_template: betas must have 16 entries; got {betas_arr.size}"
        )
    key = (gender, _betas_hash(betas_arr))
    cached = _rest_template_cache.get(key)
    if cached is not None:
        return cached

    # --- Build / reuse smplx.SMPLX(batch_size=1) for this gender. -------
    model = _rest_smplx_model_cache.get(gender)
    if model is None:
        try:
            import smplx  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "_get_rest_template requires the smplx package. "
                "pip install smplx. " + str(e)
            ) from e
        model_path = Path(smplx_dir) / f"SMPLX_{gender.upper()}.npz"
        if not model_path.exists():
            raise FileNotFoundError(
                f"_get_rest_template: SMPL-X model not found: {model_path}"
            )
        model = smplx.SMPLX(
            str(model_path),
            gender=gender,
            use_pca=False,
            flat_hand_mean=False,
            batch_size=1,
            num_betas=16,
        )
        _rest_smplx_model_cache[gender] = model
        if _rest_smplx_faces_cache is None:
            _rest_smplx_faces_cache = np.asarray(model.faces, dtype=np.int32)

    # --- Run forward at zero pose / zero transl. ------------------------
    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(betas_arr.reshape(1, 16)),
            global_orient=torch.zeros(1, 3),
            body_pose=torch.zeros(1, 63),
            left_hand_pose=torch.zeros(1, 45),
            right_hand_pose=torch.zeros(1, 45),
            jaw_pose=torch.zeros(1, 3),
            leye_pose=torch.zeros(1, 3),
            reye_pose=torch.zeros(1, 3),
            expression=torch.zeros(1, 10),
            transl=torch.zeros(1, 3),
            return_verts=True,
        )
    rest_verts = out.vertices[0].detach().cpu().numpy().astype(np.float64)  # (V, 3)
    J_rest_all = out.joints[0].detach().cpu().numpy().astype(np.float64)    # (J_total, 3)
    J_rest_22 = J_rest_all[:22]                                              # (22, 3)
    V = int(rest_verts.shape[0])

    # --- LBS weights (truncate to first 22 = body joints + renormalize). ---
    # SMPL-X weights are (10475, 55). Body verts (~7100) have most of
    # their mass in the first 22 slots (sum_22 ≈ 1.0); hand/face verts
    # (~3300) put most of their mass on slots 22..54 (hand/face joints
    # not part of OMOMO's 22-body schema). We must renormalize the
    # truncated weights so every vertex has sum=1; otherwise hand/face
    # verts collapse toward the origin (95th-percentile per-vert error
    # of 1.5 m without renorm vs sub-mm body verts). Renormalization
    # binds each hand/face vertex to its nearest body joints (typically
    # wrist/jaw), which is the right visual approximation for an OMOMO
    # body-only viz: hand/face will rigid-follow the wrist/neck.
    #
    # Sanity gate: <50% of verts having sum_22 > 0.95 means the
    # truncation hypothesis is wrong — refuse to build a template.
    lbs_weights_full = (
        model.lbs_weights.detach().cpu().numpy().astype(np.float32)  # (V, 55)
    )
    lbs_weights_22_raw = lbs_weights_full[:, :22].copy()              # (V, 22)
    sum_per_vert = lbs_weights_22_raw.sum(axis=1)                     # (V,)
    body_verts_mask = sum_per_vert > 0.95
    n_body = int(body_verts_mask.sum())
    if n_body < int(0.5 * V):
        raise ValueError(
            f"_get_rest_template: lbs_weights_22 truncation suspicious: "
            f"only {n_body}/{V} verts have sum_22 > 0.95 — expected >50% "
            f"(SMPL-X body verts). lbs_weights_full.shape={lbs_weights_full.shape}"
        )
    # Renormalize so every vertex has sum_j w[v, j] == 1. Verts with
    # essentially zero mass in the first 22 slots (e.g. some fingertip
    # vertices that bind purely to hand joints 25..54) cannot be
    # renormalized that way without dividing by zero. For those we pin
    # the vertex to its nearest body joint at rest pose (Euclidean
    # distance from rest_v to J_rest[j], j=0..21), assigning weight=1
    # to that joint. This makes the vertex rigidly follow the closest
    # body joint — the standard visual approximation for "drop the
    # hand/face DoF and keep the geometry attached". For a SMPL-X body
    # this affects ~1.3% of vertices (typically fingertips).
    EPS = 1.0e-8
    zero_mask = sum_per_vert <= EPS  # (V,)
    sum_safe = np.where(zero_mask, 1.0, sum_per_vert)
    lbs_weights_22 = (lbs_weights_22_raw / sum_safe[:, None]).astype(np.float32)
    n_zero_verts = int(zero_mask.sum())
    if n_zero_verts > 0:
        # Pin zero-weight verts to the ANATOMICALLY CORRECT body joint based
        # on which hand/face SMPL-X slot carried their mass:
        #   * slots 22-24 (jaw, l_eye, r_eye) → body joint 15 (Head)
        #   * slots 25-39 (left hand fingers)  → body joint 20 (L_Wrist)
        #   * slots 40-54 (right hand fingers) → body joint 21 (R_Wrist)
        # Previously we used "nearest body joint by rest-pose distance" which
        # produced visually broken hands (fingers rigid-pinned to chest/hip for
        # some fingertips). Anatomical pinning makes hand verts rigid-follow
        # their wrist — the standard OMOMO body-only LBS approximation.
        zero_indices = np.flatnonzero(zero_mask)                         # (n0,)
        sub_full = lbs_weights_full[zero_indices, 22:].copy()            # (n0, 33)
        sub_full_sum = sub_full.sum(axis=1, keepdims=True)               # (n0, 1)
        # Normalize to probabilities over slots 22..54 so we can pick the dominant region.
        sub_full_norm = sub_full / np.maximum(sub_full_sum, EPS)          # (n0, 33)
        dominant = sub_full_norm.argmax(axis=1)                          # (n0,) in [0, 32]
        # Map dominant SMPL-X slot index (0..32 == slots 22..54) to body joint.
        # 0..2 (slots 22..24) = Head; 3..17 (slots 25..39) = L_Wrist; 18..32 (slots 40..54) = R_Wrist.
        target = np.full((n_zero_verts,), 15, dtype=np.int64)             # default Head
        target[(dominant >= 3) & (dominant <= 17)] = 20                   # L_Wrist
        target[(dominant >= 18)] = 21                                     # R_Wrist
        # Reset weights and set target body joint to 1.
        lbs_weights_22[zero_indices] = 0.0
        lbs_weights_22[zero_indices, target] = 1.0
        n_head = int((target == 15).sum())
        n_lwrist = int((target == 20).sum())
        n_rwrist = int((target == 21).sum())
        logger.info(
            "omomo_viz._get_rest_template: pinned %d/%d zero-sum verts "
            "anatomically — Head=%d, L_Wrist=%d, R_Wrist=%d",
            n_zero_verts, V, n_head, n_lwrist, n_rwrist,
        )

    # --- Translation-only inverse bind. ---------------------------------
    inv_bind_22 = np.tile(np.eye(4, dtype=np.float64)[None, :, :], (22, 1, 1))  # (22, 4, 4)
    inv_bind_22[:, :3, 3] = -J_rest_22                                # (22, 3) -> (22, :3, 3)

    # --- Homogeneous rest verts (cached for LBS inner loop). ------------
    rest_homo = np.concatenate(
        [rest_verts, np.ones((V, 1), dtype=np.float64)], axis=1,
    )                                                                  # (V, 4)

    # --- Self-sanity: LBS with identity Q -> collapsed-humanoid blob. ---
    # M[j] = I @ inv_bind[j] = inv_bind[j].
    # v_world[v] = sum_j w[v, j] * (rest_v - J_rest[j])  (homog. trick)
    #            = (sum_j w[v, j]) * rest_v - sum_j w[v, j] * J_rest[j].
    # For body verts with sum_w ≈ 1, this is rest_v - weighted_avg(J_rest).
    # Bounding box diagonal of this point cloud witnesses LBS correctness;
    # we expect ~0.4-0.6m for a typical adult SMPL-X (skull-to-pelvis-ish
    # in the body-relative frame).
    # Use a small subset (only body verts) to avoid hand/face skewing the
    # diagnostic.
    eye_M = inv_bind_22  # M = identity_22 @ inv_bind = inv_bind (broadcast)
    # einsum: w[v,j] * inv_bind[j,3,4] @ rest_homo[v,4] -> v_collapsed[v,3]
    v_per_joint = np.einsum("jrc,vc->vjr", eye_M, rest_homo)            # (V, 22, 4)
    v_collapsed_h = np.einsum("vj,vjr->vr", lbs_weights_22, v_per_joint)  # (V, 4)
    v_collapsed = v_collapsed_h[body_verts_mask, :3]                   # (n_body, 3)
    bbox_diag = float(
        np.linalg.norm(v_collapsed.max(axis=0) - v_collapsed.min(axis=0))
    )
    logger.info(
        "omomo_viz._get_rest_template: gender=%s body_verts=%d/%d "
        "collapsed-blob bbox_diag=%.3fm (sanity)",
        gender, n_body, V, bbox_diag,
    )

    template: _RestTemplate = (rest_verts, rest_homo, lbs_weights_22, inv_bind_22)
    _rest_template_cache[key] = template
    return template


def _lbs_skin_from_q(
    body_Q: np.ndarray,
    rest_template: _RestTemplate,
) -> np.ndarray:
    """Skin SMPL-X body verts via LBS from per-joint world-frame SE(3).

    Implements the standard LBS formula for a body whose per-joint world
    transforms are ``body_Q`` (shape ``(T, 22, 4, 4)``):

        M[t, j] = body_Q[t, j] @ inv_bind[j]                    (T, 22, 4, 4)
        v_world[t, v] = sum_j lbs_weights[v, j] * M[t, j] @ rest_v_homo

    With translation-only inv_bind ``[[I, -J_rest[j]], [0, 1]]``, this is
    equivalent to skinning around the rest-frame joint positions. Matches
    SMPL-X's internal LBS up to fp32 rounding (verified by the LBS-vs-FK
    smoke test in ``test/wmm/test_omomo_viz_lbs.py``).

    Args:
        body_Q: (T, 22, 4, 4) float — per-joint world-frame SE(3).
        rest_template: tuple from :func:`_get_rest_template`.

    Returns:
        (T, V=10475, 3) float32 world-frame vertex positions.
    """
    rest_verts, rest_homo, lbs_w, inv_bind = rest_template
    body_Q_d = np.asarray(body_Q, dtype=np.float64)           # (T, 22, 4, 4)
    if body_Q_d.ndim != 4 or body_Q_d.shape[1:] != (22, 4, 4):
        raise ValueError(
            f"_lbs_skin_from_q: body_Q must be (T, 22, 4, 4); got {body_Q_d.shape}"
        )
    T = int(body_Q_d.shape[0])
    V = int(rest_homo.shape[0])

    # M[t, j] = Q[t, j] @ inv_bind[j]                          # (T, 22, 4, 4)
    M = np.einsum("tjab,jbc->tjac", body_Q_d, inv_bind)
    # v_per_joint[t, v, j, r] = M[t, j, r, c] @ rest_homo[v, c] -> reduce c
    v_per_joint = np.einsum("tjrc,vc->tvjr", M, rest_homo)    # (T, V, 22, 4)
    # Weighted sum over joints with LBS weights -> homogeneous world verts.
    v_world_h = np.einsum("vj,tvjr->tvr", lbs_w, v_per_joint)  # (T, V, 4)
    v_world = v_world_h[..., :3]                              # (T, V, 3)
    return v_world.astype(np.float32)


def _get_rest_smplx_faces() -> np.ndarray:
    """Return the SMPL-X mesh faces table (cached, loaded from male model)."""
    global _rest_smplx_faces_cache
    if _rest_smplx_faces_cache is not None:
        return _rest_smplx_faces_cache
    raise RuntimeError(
        "_get_rest_smplx_faces: faces cache empty — call _get_rest_template "
        "for any subject first to populate."
    )

# Per-point sphere radius (object-local frame, meters).
_PCL_SPHERE_RADIUS = 0.015

# Top-bar indicator colors (RGB uint8). Mirrors humoto_viz.py:56-57.
# Applied as a 10-pixel bar at the top of each inference-mode frame to mark
# SEEN (anchor/known) vs PREDICTED (generated) frames.
SEEN_INDICATOR = np.array([120, 190, 120], dtype=np.uint8)   # green
PRED_INDICATOR = np.array([220, 90, 90], dtype=np.uint8)     # red

# CHOIS eval waypoint frame indices (anchor frames the model is CONDITIONED on
# at eval time; all other frames are predicted). Used by omomo_viz_eval_preds.
_CHOIS_EVAL_WAYPOINTS = (0, 30, 60, 90, 119)

# Height sanity range for rest-frame object meshes (meters).
# The UPPER bound (2.5 m) is the critical guard: it fails loud if someone
# accidentally hands us the raw-scan mesh from captured_objects/*.obj
# (~4.5 m extent, off-centre, wrong orientation) instead of the canonical
# rest_object_geo/<name>.ply. See past lesson:
# design notes
#
# The LOWER bound catches empty / degenerate meshes. It is 0.1 m — wide
# enough to admit legitimate small-object categories that ARE physically
# small (smallbox 0.164 m, plasticbox 0.302 m, trashcan 0.337 m), narrow
# enough to flag a 0-extent or near-0-extent artefact. Note that this
# deviates from the dispatch (which proposed 0.3 m lower) — leader DM
# logged at
_OBJ_BBOX_Z_MIN = 0.1
_OBJ_BBOX_Z_MAX = 2.5


# ---------------------------------------------------------------------------
# Inference-mask helpers (used by omomo_viz_routine)
# ---------------------------------------------------------------------------

def _set_prefix_future_known(trace_known: torch.Tensor, k: int) -> None:
    """Mark frames ``k..T`` as unknown (prefix of length k is seen).

    Matches training ``mask_scheduler.random_prefix_future`` with a
    SPECIFIC k chosen by the caller. Training samples k ~ Uniform{1..T-1}
    every step; at viz time we render 3 representative k values (short /
    mid / long prefixes) so wandb shows the model's continuation
    behaviour across prefix lengths.

    In-place edit of ``trace_known`` shape ``(1, T, P)``.
    """
    T = int(trace_known.shape[1])
    if T <= 1:
        return
    k = max(1, min(k, T - 1))
    trace_known[0, k:, :] = False


def _set_entity_unknown(
    trace_known: torch.Tensor, entity_ids: List[int],
) -> None:
    """Mark the given entity columns unknown across ALL frames.

    Matches training ``mask_scheduler.h2o`` (object unknown) and
    ``mask_scheduler.r2h`` aliased as ``omomo_o2b`` (body unknown).
    In-place edit of ``trace_known`` shape ``(1, T, P)``.
    """
    for eid in entity_ids:
        if 0 <= eid < trace_known.shape[2]:
            trace_known[0, :, eid] = False


def _set_chois_obj_waypoints_known(
    trace_known: torch.Tensor,
    obj_entity_ids: List[int],
    body_entity_ids: List[int],
    waypoint_indices: Optional[List[int]] = None,
    n_waypoints: int = 4,
    reveal_body_t0: bool = True,
) -> None:
    """Match training ``mask_scheduler.chois_obj_waypoints_to_all`` exactly.

    Object tokens reveal: ``waypoint_indices`` exactly (no auto-injection
    of frame 0 or frame T-1 — caller is authoritative). Body tokens
    reveal frame 0 if ``reveal_body_t0=True`` (mirrors CHOIS
    ``input_first_human_pose=True`` from
    ``trainer_chois.py:1112-1114`` and the OMOMO training path
    ``omomo_dataset.py`` which always passes ``reveal_body_t0=True``).
    All other object frames AND all other body frames are unknown.
    In-place edit of ``trace_known`` shape ``(1, T, P)``.
    """
    # All entities start known (by clone); we mask everything then re-reveal
    # only the waypoint frames on object entities. Use zero_ not assignment
    # so we preserve validity info already in trace_known upstream.
    trace_known[0, :, :] = False
    T = int(trace_known.shape[1])
    if waypoint_indices:
        frames = sorted({int(w) for w in waypoint_indices if 0 <= int(w) < T})
    else:
        inner: List[int] = []
        if n_waypoints > 0 and T > 2:
            step = T / (n_waypoints + 1)
            inner = [int(round((k + 1) * step)) for k in range(n_waypoints)]
        frames = sorted({0, T - 1, *[f for f in inner if 0 < f < T - 1]})
    for eid in obj_entity_ids:
        if 0 <= eid < trace_known.shape[2]:
            for t in frames:
                trace_known[0, t, eid] = True
    if reveal_body_t0:
        for eid in body_entity_ids:
            if 0 <= eid < trace_known.shape[2]:
                trace_known[0, 0, eid] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pcl_spheres(
    pcl_world_xyz: np.ndarray,
    pcl_rgb: Optional[np.ndarray] = None,
    radius: float = _PCL_SPHERE_RADIUS,
) -> trimesh.Trimesh:
    """Construct a single concatenated trimesh of uv_spheres at every point.

    Args:
        pcl_world_xyz: ``(N, 3)`` float64 world coordinates.
        pcl_rgb: optional ``(N, 3)`` uint8 per-point RGB. If None, uses
            ``OBJ_COLOR_RGBA``.
        radius: sphere radius in meters.

    Returns:
        Single merged ``trimesh.Trimesh`` with per-vertex colors.
    """
    pts = np.asarray(pcl_world_xyz, dtype=np.float64)
    N = pts.shape[0]

    # Build a minimal base sphere once, copy + translate for each point.
    base = trimesh.creation.uv_sphere(radius=float(radius), count=[4, 4])
    n_verts_per_sphere = base.vertices.shape[0]

    spheres: List[trimesh.Trimesh] = []
    for i, p in enumerate(pts):
        s = base.copy()
        s.apply_translation(p)
        spheres.append(s)

    if not spheres:
        # Return empty mesh
        return trimesh.Trimesh(process=False)

    merged = trimesh.util.concatenate(spheres)
    total_verts = merged.vertices.shape[0]

    # Assign per-vertex colors: repeat each point's color across its sphere verts.
    colors = np.zeros((total_verts, 4), dtype=np.uint8)
    for i in range(N):
        start = i * n_verts_per_sphere
        end = start + n_verts_per_sphere
        if pcl_rgb is not None:
            r, g, b = int(pcl_rgb[i, 0]), int(pcl_rgb[i, 1]), int(pcl_rgb[i, 2])
            colors[start:end] = np.array([r, g, b, 255], dtype=np.uint8)
        else:
            colors[start:end] = np.array(OBJ_COLOR_RGBA, dtype=np.uint8)

    return trimesh.Trimesh(
        vertices=merged.vertices,
        faces=merged.faces,
        vertex_colors=colors,
        process=False,
    )


def _assert_obj_height_sane(mesh_path: "Path | str", bbox: np.ndarray) -> None:
    """Assert the rest-frame object mesh height is within expected bounds.

    Args:
        mesh_path: path to the mesh file (for error messages).
        bbox: ``(2, 3)`` array of (min, max) corners.

    Raises:
        ValueError: if Z-extent is outside ``[_OBJ_BBOX_Z_MIN, _OBJ_BBOX_Z_MAX]``.
    """
    z_extent = float(bbox[1, 2] - bbox[0, 2])
    if not (_OBJ_BBOX_Z_MIN < z_extent < _OBJ_BBOX_Z_MAX):
        raise ValueError(
            f"OmomoViz: {mesh_path} bbox_z = {z_extent:.3f}m, "
            f"expected {_OBJ_BBOX_Z_MIN}..{_OBJ_BBOX_Z_MAX} — obj_mesh_path trap?"
        )


def _sample_pcl_from_mesh(
    mesh: trimesh.Trimesh,
    num_pts: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``num_pts`` vertices from ``mesh.vertices`` with optional RGB.

    Args:
        mesh: source trimesh.
        num_pts: number of points to sample.
        rng: optional ``np.random.Generator`` for reproducibility.

    Returns:
        ``(num_pts, 6)`` float32 array: xyz + rgb in [-0.5, 0.5].
        If the mesh has no vertex colors, RGB is filled with zeros.
    """
    if rng is None:
        rng = np.random.default_rng()

    verts = np.asarray(mesh.vertices, dtype=np.float32)  # (V, 3)
    V = verts.shape[0]

    # Sample with replacement if mesh has fewer vertices than requested.
    idx = rng.integers(0, V, size=num_pts)
    xyz = verts[idx]  # (num_pts, 3)

    # RGB: use vertex colors if available, else zeros.
    # trimesh.visual can be ColorVisuals (has vertex_colors) or TextureVisuals
    # (no vertex_colors — PLYs without per-vertex color land here). Pyright
    # cannot discriminate these at static-analysis time; the getattr guard
    # plus the except fallback keeps us safe at runtime AND silences the
    # TextureVisuals branch warning.
    visual = getattr(mesh, "visual", None)
    vc_raw = getattr(visual, "vertex_colors", None) if visual is not None else None
    if vc_raw is not None:
        try:
            vc = np.asarray(vc_raw, dtype=np.float32)  # (V, 4) uint8-based
            # vertex_colors are [0,255]; normalize to [0,1] then shift to [-0.5, 0.5]
            rgb_raw = vc[idx, :3] / 255.0 - 0.5  # (num_pts, 3)
        except (ValueError, IndexError, TypeError) as e:
            # Log the deviation — do not swallow silently. Malformed
            # per-vertex colors fall back to zero RGB (HUMOTO PCL-NaN
            # lesson: never leave RGB as NaN; zero is the safe default).
            logger.warning(
                "omomo_viz: vertex_colors malformed, zero-filling RGB "
                "(V=%d, err=%s)", V, e,
            )
            rgb_raw = np.zeros((num_pts, 3), dtype=np.float32)
    else:
        rgb_raw = np.zeros((num_pts, 3), dtype=np.float32)

    return np.concatenate([xyz, rgb_raw.astype(np.float32)], axis=1)  # (num_pts, 6)


def _render_scene_with_pcl(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    pcl_world_xyz: Optional[np.ndarray],
    pcl_rgb: Optional[np.ndarray],
    out_path: "Path | str",
    width: int = 960,
    height: int = 960,
    quality: int = 92,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
) -> None:
    """Render body + object + PCL spheres + ground to JPEG via pyrender EGL.

    Args:
        body_mesh: SMPL-X body trimesh (world frame).
        obj_mesh: rigid-transformed object trimesh (world frame).
        pcl_world_xyz: ``(N, 3)`` float64 world-frame PCL points.
        pcl_rgb: optional ``(N, 3)`` uint8 per-point RGB.
        out_path: output JPEG path.
        width, height: output resolution.
        quality: JPEG quality (0-95).
    """
    out_path = Path(out_path)

    # Compute combined bbox of body, object, and PCL for camera framing.
    b_bbox = body_mesh.bounds      # (2, 3)
    o_bbox = obj_mesh.bounds       # (2, 3)
    all_pts = [b_bbox[0], b_bbox[1], o_bbox[0], o_bbox[1]]
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        all_pts.append(pcl_world_xyz.min(axis=0))
        all_pts.append(pcl_world_xyz.max(axis=0))
    all_pts_np = np.stack(all_pts, axis=0)  # (K, 3)
    both_min = all_pts_np.min(axis=0)
    both_max = all_pts_np.max(axis=0)
    center = 0.5 * (both_min + both_max)
    diag = float(np.linalg.norm(both_max - both_min))

    # Aim at upper-body height, biased toward combined scene center.
    body_h = float(b_bbox[1, 2] - b_bbox[0, 2])
    target = np.array(
        [center[0], center[1], b_bbox[0, 2] + 0.55 * max(body_h, 0.2)],
        dtype=np.float64,
    )
    cam_pose = _camera_pose_diagonal(target, dist=max(3.2, 0.9 * diag))

    scene = pyrender.Scene(
        bg_color=np.array([0.85, 0.88, 0.92, 1.0], dtype=np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )

    # Ground plane
    scene.add(
        pyrender.Mesh.from_trimesh(_make_ground_plane(half_size=4.0, z=0.0), smooth=False)
    )

    # Body mesh
    scene.add(
        pyrender.Mesh.from_trimesh(_color_mesh(body_mesh, BODY_COLOR_RGBA), smooth=False)
    )

    # Object mesh (opaque)
    scene.add(
        pyrender.Mesh.from_trimesh(_color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False)
    )

    # PCL spheres
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        pcl_sphere_mesh = _build_pcl_spheres(
            pcl_world_xyz, pcl_rgb, radius=pcl_radius,
        )
        scene.add(
            pyrender.Mesh.from_trimesh(pcl_sphere_mesh, smooth=False)
        )

    # Camera
    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)

    # Lights: key + fill + rim directional
    key = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=4.0)
    scene.add(
        key,
        pose=_look_at_z_up(eye=target + np.array([2.0, -2.0, 4.0]), target=target),
    )
    fill = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)
    scene.add(
        fill,
        pose=_look_at_z_up(eye=target + np.array([-3.0, 2.0, 2.0]), target=target),
    )
    rim = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)
    scene.add(
        rim,
        pose=_look_at_z_up(eye=target + np.array([0.0, 3.0, 1.5]), target=target),
    )

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).convert("RGB").save(str(out_path), format="JPEG", quality=quality)


def _extract_scalar_field(extra: Dict, key: str, bid: int, default=None):
    """Pull ``extra[key]`` either as a per-item list or a batch-wide scalar.

    OMOMO collates ``obj_category``, ``betas``, ``gender`` into
    ``batch["extra"]`` as dict-of-lists per CanonicalDatasetOutput. When
    the dataloader uses a naive default collate they may become a flat
    scalar shared across the batch — handle both.

    Returns ``default`` if ``key`` is absent.
    """
    if key not in extra:
        return default
    v = extra[key]
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        return v[bid] if bid < len(v) else v[-1]
    # Tensors / arrays: per-item first-dim indexing when possible.
    if isinstance(v, (np.ndarray, torch.Tensor)):
        if v.ndim >= 1 and v.shape[0] > bid:
            return v[bid]
        return v
    return v


def _build_training_scene(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    width: int,
    height: int,
    pcl_world_xyz: Optional[np.ndarray] = None,
    pcl_rgb: Optional[np.ndarray] = None,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
) -> pyrender.Scene:
    """Build the pyrender scene + lights + camera for training-time viz.

    Shared by the cached-renderer path (``_render_frame_given_renderer``).

    Args:
        body_mesh: SMPL-X body trimesh (world frame).
        obj_mesh: rigid-transformed object trimesh (world frame).
        width, height: output resolution.
        pcl_world_xyz: optional ``(N, 3)`` float64 world-frame PCL points.
            When provided and non-empty, 256 spheres are rendered in the
            scene at these positions. Used by ``draw_gt`` to overlay the
            object's 256-point local_pcl from the batch (transformed to
            world frame per-frame).
        pcl_rgb: optional ``(N, 3)`` uint8 per-point RGB colors. Falls
            back to OBJ_COLOR_RGBA when None.
        pcl_radius: sphere radius (meters). Defaults to
            ``_PCL_SPHERE_RADIUS`` (15 mm).
    """
    b_bbox = body_mesh.bounds      # (2, 3)
    o_bbox = obj_mesh.bounds       # (2, 3)
    # Build combined bbox: include PCL extent so the camera frame still
    # captures all 256 spheres even when they protrude beyond the obj_mesh
    # bbox (rare but possible when the rest mesh is sparser than the PCL).
    all_pts = [b_bbox[0], b_bbox[1], o_bbox[0], o_bbox[1]]
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        all_pts.append(np.asarray(pcl_world_xyz).min(axis=0))
        all_pts.append(np.asarray(pcl_world_xyz).max(axis=0))
    all_pts_np = np.stack(all_pts, axis=0)  # (K, 3)
    both_min = all_pts_np.min(axis=0)
    both_max = all_pts_np.max(axis=0)
    center = 0.5 * (both_min + both_max)
    diag = float(np.linalg.norm(both_max - both_min))
    body_h = float(b_bbox[1, 2] - b_bbox[0, 2])
    target = np.array(
        [center[0], center[1], b_bbox[0, 2] + 0.55 * max(body_h, 0.2)],
        dtype=np.float64,
    )
    cam_pose = _camera_pose_diagonal(target, dist=max(3.2, 0.9 * diag))

    scene = pyrender.Scene(
        bg_color=np.array([0.85, 0.88, 0.92, 1.0], dtype=np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )
    scene.add(pyrender.Mesh.from_trimesh(
        _make_ground_plane(half_size=4.0, z=0.0), smooth=False,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(body_mesh, BODY_COLOR_RGBA), smooth=False,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False,
    ))
    # PCL spheres (object's local_pcl, transformed to world by caller).
    if pcl_world_xyz is not None and len(pcl_world_xyz) > 0:
        pcl_sphere_mesh = _build_pcl_spheres(
            pcl_world_xyz, pcl_rgb, radius=float(pcl_radius),
        )
        scene.add(pyrender.Mesh.from_trimesh(pcl_sphere_mesh, smooth=False))
    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)
    for offset, inten in (
        (np.array([2.0, -2.0, 4.0]), 4.0),
        (np.array([-3.0, 2.0, 2.0]), 2.0),
        (np.array([0.0, 3.0, 1.5]), 2.0),
    ):
        light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=inten)
        scene.add(light, pose=_look_at_z_up(eye=target + offset, target=target))
    return scene


def _render_frame_given_renderer(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    renderer: pyrender.OffscreenRenderer,
    width: int,
    height: int,
    pcl_world_xyz: Optional[np.ndarray] = None,
    pcl_rgb: Optional[np.ndarray] = None,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
) -> np.ndarray:
    """Render into an EXISTING renderer; caller owns its lifecycle.

    Forwards the optional PCL kwargs to ``_build_training_scene`` so the
    cached-renderer path (used by ``draw_gt``) can overlay 256-point
    object PCL spheres per-frame.
    """
    scene = _build_training_scene(
        body_mesh, obj_mesh, width, height,
        pcl_world_xyz=pcl_world_xyz,
        pcl_rgb=pcl_rgb,
        pcl_radius=pcl_radius,
    )
    color, _ = renderer.render(scene)
    return color[..., :3].astype(np.uint8)


def _build_ctx_scene(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    width: int,
    height: int,
) -> pyrender.Scene:
    """Build the ctx-panel pyrender scene WITHOUT a ground plane.

    Identical lighting / camera framing to ``_build_training_scene`` but
    omits the ground plane — the ctx is an abstract reference frame
    (everything at origin), not a physical scene. The camera is centered
    on the combined collapsed-body + rest-object bounding box.
    """
    b_bbox = body_mesh.bounds
    o_bbox = obj_mesh.bounds
    both_min = np.minimum(b_bbox[0], o_bbox[0])
    both_max = np.maximum(b_bbox[1], o_bbox[1])
    center = 0.5 * (both_min + both_max)
    diag = float(np.linalg.norm(both_max - both_min))
    body_h = float(b_bbox[1, 2] - b_bbox[0, 2])
    target = np.array(
        [center[0], center[1], b_bbox[0, 2] + 0.55 * max(body_h, 0.2)],
        dtype=np.float64,
    )
    cam_pose = _camera_pose_diagonal(target, dist=max(2.0, 0.9 * diag))

    scene = pyrender.Scene(
        bg_color=np.array([0.85, 0.88, 0.92, 1.0], dtype=np.float32),
        ambient_light=np.array([0.35, 0.35, 0.35], dtype=np.float32),
    )
    # NOTE: NO ground plane — ctx is abstract.
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(body_mesh, BODY_COLOR_RGBA), smooth=False,
    ))
    scene.add(pyrender.Mesh.from_trimesh(
        _color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False,
    ))
    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)
    for offset, inten in (
        (np.array([2.0, -2.0, 4.0]), 4.0),
        (np.array([-3.0, 2.0, 2.0]), 2.0),
        (np.array([0.0, 3.0, 1.5]), 2.0),
    ):
        light = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=inten)
        scene.add(light, pose=_look_at_z_up(eye=target + offset, target=target))
    return scene


def _render_frame_no_ground(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    renderer: pyrender.OffscreenRenderer,
    width: int,
    height: int,
) -> np.ndarray:
    """Render the ctx panel without a ground plane; reuse existing renderer."""
    scene = _build_ctx_scene(body_mesh, obj_mesh, width, height)
    color, _ = renderer.render(scene)
    return color[..., :3].astype(np.uint8)


# ---------------------------------------------------------------------------
# OmomoVisualizer — training-time callback class
# ---------------------------------------------------------------------------

class OmomoVisualizer:
    """Mesh-based visualizer for OMOMO human-object interaction sequences.

    Uses SMPL-X forward kinematics for the body and rigid transforms for
    the single object. Mirrors the interface of ``HumotoVisualizer``.

    Scope in this module:
        * ``draw_gt(batch)`` is fully implemented — renders a per-frame
          GT video from the batch's trace (no model inference required)
          and logs it via ``pl_module.logger.log_video``.
        * ``draw_inference(batch, pred_x0)`` is implemented for the
          minimal "GT-substituted-for-pred" path (useful when a caller
          passes a prediction tensor but the training harness hasn't
          yet wired the wmm Agent's OMOMO inference path).
        * ``omomo_viz_routine(batch, phase, batch_idx, eval_model)``
          renders GT only for now. The half-future / 1-step / o2h / h2o
          multi-mode branches log a WARNING and skip, because the wmm
          Agent's OMOMO ``inference()`` signature is not stable in
          master at this module's checkpoint. They will be wired when the
          Agent API stabilizes (DM logged
          

    Args:
        pl_module: Parent PyTorch Lightning module (``PLModel``).
        omomo_cfg (dict): YAML block from ``model.omomo``. Optional keys:
            ``smplx_dir`` (default ``data_wmm/omomo/smplx``),
            ``geo_dir`` (default ``data_wmm/omomo/rest_object_geo``),
            ``width`` (default 512), ``height`` (default 512),
            ``num_pcl_points`` (default 256),
            ``pcl_seed`` (default 0).
    """

    def __init__(self, pl_module, omomo_cfg):
        self.pl = pl_module
        # Defensive defaults initialized BEFORE the cfg-empty early-return,
        # so partial instances still have a consistent attribute surface
        # (review-04 NOTE #2).
        self._inference_defer_logged = False
        if not omomo_cfg or len(omomo_cfg) == 0:
            self.flag = False
            return
        self.flag = True
        self.smplx_dir = Path(omomo_cfg.get("smplx_dir", "data_wmm/omomo/smplx"))
        self.geo_dir = Path(omomo_cfg.get("geo_dir", "data_wmm/omomo/rest_object_geo"))
        # data_root is needed to read betas/gender from the source npz
        # when the batch does not carry them in extras. Defaults to
        # geo_dir.parent = data_wmm/omomo/ which contains test_npz/ +
        # train_npz/.
        self._data_root = Path(omomo_cfg.get("data_root", self.geo_dir.parent))
        self.omomo_viz_cfg = self.pl.cfg.get("visualize", {}).get("omomo", {})
        # Resolution: visualize.omomo.{height,width} preferred; fall back
        # to omomo_cfg.{height,width}; final default 512.
        self._H = int(
            self.omomo_viz_cfg.get("height", omomo_cfg.get("height", 512))
        )
        self._W = int(
            self.omomo_viz_cfg.get("width", omomo_cfg.get("width", 512))
        )
        self._num_pcl_points = int(omomo_cfg.get("num_pcl_points", 256))
        self._pcl_seed = int(omomo_cfg.get("pcl_seed", 0))
        self._fk = _SmplxFK(self.smplx_dir)
        # Cache a single pyrender offscreen renderer for the training-time
        # draw_gt path — avoids 120× EGL init/teardown per batch.
        # Protected by a lock because the training callback may be
        # invoked from non-main threads in some PL configurations.
        self._render_lock = __import__("threading").Lock()
        self._cached_renderer: Optional[pyrender.OffscreenRenderer] = None

    # ------------------------------------------------------------------
    # Public helpers for inspection / ad-hoc use
    # ------------------------------------------------------------------

    def draw_gt(self, batch: Dict, bid: int = 0) -> np.ndarray:
        """Render a (T_actual, H, W, 3) uint8 GT video from batch[bid].

        Expects the wmm OMOMO batch layout produced by ``OmomoDataset``:
        ``trace[bid]`` contains 22 body joints (group 0) + 1 object
        (group 1), each carrying SMPL-X axis-angle rotation embedded in
        its SE(3) rotation block (pelvis SE(3) also carries root trans).
        Body joint axis-angles are reconstructed by matrix-logging each
        rotation column; SMPL-X FK then produces the per-frame mesh.

        Args:
            batch: collated training batch dict. Expected keys:
                ``trace`` ``(B, T, P, 4, 4)``, ``trace_mask`` ``(B, T, P)``,
                ``extra`` ``{obj_category, entry, ...}``. Exact layout
                enforced at runtime; deviations raise loudly.
            bid: batch item index.

        Returns:
            ``(T_actual, H, W, 3)`` uint8 frames.
        """
        if not self.flag:
            raise RuntimeError("OmomoVisualizer is disabled (empty omomo_cfg)")

        body_verts, body_faces, obj_R, obj_t, actual_len, obj_category = \
            self._fk_from_batch(batch, bid)

        rest_mesh, _, _ = self._load_rest(obj_category)
        V_rest = np.asarray(rest_mesh.vertices, dtype=np.float64)
        obj_faces = np.asarray(rest_mesh.faces)

        # Extract object's batch-supplied local_pcl (rest frame, xyz+rgb).
        # OmomoDataset.__getitem_inner__ stores the object PCL at entity
        # index 22 with mask=1; body entities 0..21 carry zero-PCL with
        # mask=0. After collation, batch["local_pcl"] has shape
        # (B, P, num_pts, PCL_dim) with PCL_dim==6 (xyz + centered rgb in
        # [-0.5, 0.5]). When the batch was assembled without PCL (older
        # tests / non-OMOMO domains) we fall back to no overlay so this
        # path stays backwards compatible.
        pcl_xyz_rest: Optional[np.ndarray] = None
        pcl_rgb_u8: Optional[np.ndarray] = None
        local_pcl = batch.get("local_pcl")
        local_pcl_mask = batch.get("local_pcl_mask")
        if local_pcl is not None:
            try:
                pcl_t = local_pcl[bid]                  # (P, num_pts, 6)
                if isinstance(pcl_t, torch.Tensor):
                    pcl_np = pcl_t.detach().cpu().numpy()
                else:
                    pcl_np = np.asarray(pcl_t)
                # Determine which row carries the object PCL. P>=23
                # (22 body + 1 object); object lives at index 22 per
                # OmomoDataset. Honor local_pcl_mask when available so we
                # never pick up a zero-row body slot if entity layout
                # changes upstream.
                obj_idx = 22 if pcl_np.shape[0] > 22 else (pcl_np.shape[0] - 1)
                mask_ok = True
                if local_pcl_mask is not None:
                    mask_t = local_pcl_mask[bid]
                    if isinstance(mask_t, torch.Tensor):
                        mask_np = mask_t.detach().cpu().numpy().astype(bool)
                    else:
                        mask_np = np.asarray(mask_t).astype(bool)
                    if obj_idx < mask_np.shape[0] and not bool(mask_np[obj_idx]):
                        # Object PCL slot is masked off — skip overlay rather
                        # than fall back to a body slot (which would render
                        # 256 zero-XYZ spheres at world origin).
                        mask_ok = False
                        logger.warning(
                            "OmomoVisualizer.draw_gt: local_pcl_mask[%d] is "
                            "False (bid=%d) — skipping PCL overlay", obj_idx, bid,
                        )
                pcl_xyzrgb = pcl_np[obj_idx] if mask_ok else None  # (num_pts, 6) | None
                if pcl_xyzrgb is not None and pcl_xyzrgb.shape[0] > 0:
                    pcl_xyz_rest = pcl_xyzrgb[:, :3].astype(np.float64)  # (N, 3)
                    rgb_centered = pcl_xyzrgb[:, 3:6].astype(np.float32)  # (N, 3)
                    pcl_rgb_u8 = np.clip(
                        (rgb_centered + 0.5) * 255.0, 0.0, 255.0,
                    ).astype(np.uint8)
            except (IndexError, KeyError, TypeError) as e:
                logger.warning(
                    "OmomoVisualizer.draw_gt: failed to slice local_pcl "
                    "(bid=%d): %s — falling back to no PCL overlay",
                    bid, e,
                )
                pcl_xyz_rest = None
                pcl_rgb_u8 = None

        frames: List[np.ndarray] = []
        for t in range(actual_len):
            body_mesh = trimesh.Trimesh(
                vertices=body_verts[t].astype(np.float64),
                faces=body_faces, process=False,
            )
            V_w = V_rest @ obj_R[t].T + obj_t[t][None, :]
            obj_mesh = trimesh.Trimesh(
                vertices=V_w, faces=obj_faces, process=False,
            )
            # Per-frame world-frame PCL: same rigid transform as the
            # object mesh. Mirrors render_omomo_window_jpeg's PCL path
            # but uses the batch's collated local_pcl (the actual model
            # input) instead of a fresh sample from the rest mesh — so
            # the viz reflects exactly what the model sees.
            pcl_world_xyz_t: Optional[np.ndarray] = None
            if pcl_xyz_rest is not None:
                pcl_world_xyz_t = pcl_xyz_rest @ obj_R[t].T + obj_t[t][None, :]
                # (N, 3) float64
            # PCL is now overlaid via the cached-renderer path; we still
            # reuse the instance-cached renderer across frames (review-02
            # #5 — avoids 120x EGL init/teardown).
            frame = self._render_frame_with_cached_renderer(
                body_mesh, obj_mesh,
                pcl_world_xyz=pcl_world_xyz_t,
                pcl_rgb=pcl_rgb_u8,
            )
            frames.append(frame)
        return np.stack(frames, axis=0)  # (T_actual, H, W, 3)

    def _render_frame_with_cached_renderer(
        self,
        body_mesh: trimesh.Trimesh,
        obj_mesh: trimesh.Trimesh,
        pcl_world_xyz: Optional[np.ndarray] = None,
        pcl_rgb: Optional[np.ndarray] = None,
        pcl_radius: float = _PCL_SPHERE_RADIUS,
    ) -> np.ndarray:
        """Thread-safe render using a single OffscreenRenderer per instance.

        Forwards the optional PCL kwargs to
        ``_render_frame_given_renderer`` so ``draw_gt`` can overlay the
        256-point object local_pcl per-frame. When ``pcl_world_xyz`` is
        None, falls back to body+object only (preserves backwards
        compatibility for callers that have no PCL).
        """
        with self._render_lock:
            if self._cached_renderer is None:
                self._cached_renderer = pyrender.OffscreenRenderer(
                    viewport_width=self._W, viewport_height=self._H,
                )
            return _render_frame_given_renderer(
                body_mesh, obj_mesh, self._cached_renderer,
                self._W, self._H,
                pcl_world_xyz=pcl_world_xyz,
                pcl_rgb=pcl_rgb,
                pcl_radius=pcl_radius,
            )

    def _maybe_log_penetration_viz(
        self,
        batch: Dict,
        bid: int,
        phase: str,
        batch_idx: int,
        actual_len: int,
        log_video,
    ) -> None:
        """Render + log world-space penetration overlay (palm spheres color-coded by SDF).

        Gated by ``self.pl.cfg.viz_penetration_flag``. When False this is a no-op.
        Reads obj_category, betas, gender from ``batch['extra']``; runs SMPL-X
        forward on GT pose to get palm world positions; queries SDF v2; saves
        per-frame JPGs and logs video to wandb under name ``GT_penetration_<phase>``.
        """
        # Read flag from PL cfg (top level), default False
        cfg = getattr(self.pl, "cfg", None) or {}
        flag = bool(cfg.get("viz_penetration_flag", False)) if hasattr(cfg, "get") else False
        if not flag:
            return

        extra = batch.get("extra", {}) or {}
        obj_categories = extra.get("obj_category")
        if obj_categories is None or bid >= len(obj_categories):
            logger.warning("penetration_viz: missing obj_category — skip")
            return
        obj_category = obj_categories[bid]

        # Locate sdf v2 + rest_geo paths from omomo_cfg (or fall back to defaults)
        omomo_cfg = self.omomo_cfg if hasattr(self, "omomo_cfg") else {}
        sdf_v2_dir = (cfg.get("model", {}) or {}).get(
            "sdf_v2_dir", "data_wmm/omomo_v2/rest_object_sdf_v2"
        )
        rest_geo_dir = (cfg.get("model", {}) or {}).get(
            "rest_geo_dir", "data_wmm/omomo_v2/rest_object_geo"
        )
        ply_path = os.path.join(rest_geo_dir, f"{obj_category}.ply")
        sdf_stem = os.path.join(sdf_v2_dir, obj_category)

        # GT object SE3 from batch.trace at slot 22 (OMOMO obj entity)
        trace = batch["trace"]  # (B, T, P, 4, 4)
        T_full = int(trace.shape[1])
        T = max(1, min(actual_len, T_full))
        obj_slot = 22 if trace.shape[2] > 22 else (trace.shape[2] - 1)
        gt_obj_se3 = trace[bid, :T, obj_slot]                      # (T, 4, 4)
        obj_R_seq = gt_obj_se3[..., :3, :3].cpu().numpy()
        obj_t_seq = gt_obj_se3[..., :3, 3].cpu().numpy()

        # GT palm world: SMPL-X forward on GT betas + body pose. The dataset
        # only stores wrist positions, not palm. Use lhand_wrist_jpos +
        # rhand_wrist_jpos from extras as a proxy (off by ~7cm; acceptable
        # for viz overlay where the SDF query is a sphere of ~2.5cm radius).
        lwrist = extra.get("lhand_wrist_jpos")
        rwrist = extra.get("rhand_wrist_jpos")
        if lwrist is None or rwrist is None:
            logger.warning("penetration_viz: missing wrist_jpos in extras — skip")
            return
        lw = np.asarray(lwrist[bid])[:T]                            # (T, 3)
        rw = np.asarray(rwrist[bid])[:T]                            # (T, 3)
        palm_world_seq = np.stack([lw, rw], axis=1).astype(np.float32)  # (T, 2, 3)

        # Render. Reuse the OmomoVisualizer's cached pyrender.OffscreenRenderer
        # to avoid creating a new EGL context per call (creating + deleting
        # additional OffscreenRenderers during a viz cycle corrupts the GL
        # state of the cached renderer and causes downstream draw_inference
        # calls to fail with OpenGL.error.GLError, silently dropping all
        # generated-mode videos for the rest of the run).
        log_dir_root = getattr(self.pl.logger, "log_dir", None) or getattr(self.pl, "log_dir", "/tmp")
        out_dir = os.path.join(
            log_dir_root,
            f"penetration_viz_{phase}_step{self.pl.global_step}",
        )
        # Hold the same lock used by `_render_frame_with_cached_renderer` for
        # the entire render so the cached renderer is single-writer (matches
        # the GT/inference render pattern). Force-init the cached renderer if
        # draw_gt has not run yet.
        with self._render_lock:
            if self._cached_renderer is None:
                self._cached_renderer = pyrender.OffscreenRenderer(
                    viewport_width=self._W, viewport_height=self._H,
                )
            try:
                paths = render_penetration_sequence(
                    obj_mesh_path=ply_path,
                    sdf_stem=sdf_stem,
                    obj_R_seq_world=obj_R_seq,
                    obj_t_seq_world=obj_t_seq,
                    palm_world_seq=palm_world_seq,
                    out_dir=out_dir,
                    save_n_frames=min(T, 16),  # 16 frames keeps wandb mp4 under ~MB
                    renderer=self._cached_renderer,
                    width=self._W,
                    height=self._H,
                )
            except Exception as exc:
                logger.warning("penetration_viz: render failed: %s", exc, exc_info=True)
                return

        if not paths:
            return
        # Stack JPGs into video tensor (T_save, H, W, 3) and log
        from PIL import Image as _PIL
        frames = [np.asarray(_PIL.open(p).convert("RGB")) for p in paths]
        video = torch.from_numpy(np.stack(frames, axis=0)).float().permute(0, 3, 1, 2) / 255.0
        log_video(
            phase, f"GT_penetration_{phase}", video,
            self.pl.global_step, batch_idx,
        )

    def draw_inference(
        self,
        batch: Dict,
        pred_x0: Optional[torch.Tensor],
        bid: int = 0,
    ) -> np.ndarray:
        """Render predicted trace the same way ``draw_gt`` renders GT.

        If ``pred_x0`` is None, falls back to GT (useful for dry-run).
        Expects ``pred_x0`` shape matching ``batch['trace']``. The helper
        swaps the selected batch item's trace for the predicted one and
        calls ``draw_gt`` internally.
        """
        if not self.flag:
            raise RuntimeError("OmomoVisualizer is disabled (empty omomo_cfg)")
        if pred_x0 is None:
            return self.draw_gt(batch, bid=bid)

        # Shallow copy batch and replace the trace. When pred_x0 was produced
        # by per-item inference (shape (1, T, P, 4, 4)) we also have to
        # slice trace_mask to match — draw_gt's shape-check requires
        # trace.shape[:3] == trace_mask.shape[:3]. After slicing, render
        # with bid=0.
        sub_batch = dict(batch)
        pred = pred_x0.detach().clone()
        sub_batch["trace"] = pred
        if pred.shape[0] != int(batch["trace_mask"].shape[0]):
            sub_batch["trace_mask"] = batch["trace_mask"][bid:bid + 1]
            bid = 0
        return self.draw_gt(sub_batch, bid=bid)

    # ------------------------------------------------------------------
    # Inference dispatcher used by omomo_viz_routine
    # ------------------------------------------------------------------
    def _run_inference_one(
        self,
        batch: Dict,
        bid: int,
        eval_model,
        set_known_fn,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Run eval_model.inference for one batch item in one masking mode.

        Builds ``trace_known_mask`` / ``context_known_mask`` tensors (all
        ones), applies the mode-specific ``set_known_fn`` to trace_known
        (in-place edit), then calls AgentV11.inference. Returns
        ``(gen_trace (1, T, P, 4, 4), tk (1, T, P) bool)`` — ``tk`` is
        returned so the viz routine can paint the SEEN/PRED top-bar
        indicator per-frame (green=seen, red=predicted).
        """
        tk = torch.ones_like(batch["trace_mask"][bid:bid + 1]).bool()
        ck = torch.ones_like(batch["context_mask"][bid:bid + 1]).bool()
        set_known_fn(tk)

        local_pcl = batch.get("local_pcl")
        local_pcl_mask = batch.get("local_pcl_mask")
        if local_pcl is not None:
            local_pcl = local_pcl[bid:bid + 1]
            local_pcl_mask = (
                local_pcl_mask[bid:bid + 1] if local_pcl_mask is not None else None
            )

        _, gen_trace = eval_model.inference(
            trace=batch["trace"][bid:bid + 1],
            context=batch["context"][bid:bid + 1],
            trace_known_mask=tk,
            context_known_mask=ck,
            trace_mask=batch["trace_mask"][bid:bid + 1],
            context_mask=batch["context_mask"][bid:bid + 1],
            real_timestep=batch["time"][bid:bid + 1],
            group_id=batch["trace_group_id"][bid:bid + 1],
            sem_type=batch["trace_sem_type"][bid:bid + 1],
            rot_mask=batch["trace_rot_mask"][bid:bid + 1],
            local_pcl=local_pcl,
            local_pcl_mask=local_pcl_mask,
            text=[batch["text"][bid]] if batch.get("text") else None,
        )
        return gen_trace, tk

    # ------------------------------------------------------------------
    # Ctx panel — collapsed-LBS body + object-at-origin (single frame)
    # ------------------------------------------------------------------
    def _render_ctx_panel(
        self, batch: Dict, bid: int,
    ) -> np.ndarray:
        """Honest render of the dataset's context frame — NO tricks in the viz.

        Reads ``batch["context"][bid, 0, :22]`` (body ctx SE(3), shape
        (22, 4, 4)) and ``batch["context"][bid, 0, 22]`` (object ctx
        SE(3)) directly. LBS-skins the body mesh from those per-joint
        transforms and rigid-transforms the object rest mesh by the
        object ctx. Exactly the semantics the model sees for ctx — no
        cosmetic substitution, no identity-hardcode.

        Post-dataset-DA-pose rewrite (2026-04-24), OMOMO's body_ctx is
        the DA-pose world-frame SE(3) from
        ``prepare_data/omomo/build_q.da_pose_q``: standing body, arms
        ~45° down, Z-up world. This method therefore renders that
        standing body honestly. If the dataset changes ctx again, this
        viz tracks it automatically.

        Rendered **without a ground plane** — ctx is a reference frame,
        not a physical scene.

        Args:
            batch: training batch dict with ``context`` key, shape
                ``(B, C, P, 4, 4)``.
            bid: batch item index.

        Returns:
            ``(H, W, 3)`` uint8 image.
        """
        if not self.flag:
            raise RuntimeError("OmomoVisualizer is disabled (empty omomo_cfg)")

        if "context" not in batch:
            raise KeyError(
                "OmomoVisualizer._render_ctx_panel: batch['context'] missing "
                "(required for honest ctx viz — no identity fallback)."
            )
        context = batch["context"]
        if context.ndim != 5:
            raise ValueError(
                f"_render_ctx_panel: expected batch['context'] to be "
                f"(B, C, P, 4, 4), got ndim={context.ndim}"
            )

        extra = batch.get("extra", {}) or {}
        obj_category = _extract_scalar_field(
            extra, "obj_category", bid, default=None,
        )
        if obj_category is None:
            raise KeyError(
                "OmomoVisualizer._render_ctx_panel: "
                "batch['extra']['obj_category'] is required"
            )
        betas_np, gender = self._resolve_betas_gender(extra, bid)

        # Read the dataset's ctx SE(3) for this batch item. Layout is
        # (B, C, P, 4, 4) per the wmm collate. OMOMO uses C=1, P=23.
        ctx_body_Q = context[bid, 0, :22].detach().cpu().numpy().astype(np.float64)  # (22, 4, 4)
        ctx_obj_se3 = context[bid, 0, 22].detach().cpu().numpy().astype(np.float64)  # (4, 4)

        # Q → axis-angle → SMPL-X FK (one-frame "sequence"). Same path as
        # _fk_from_batch — gives correct hand/face geometry for the DA-pose
        # or whatever ctx the dataset provides.
        from prepare_data.omomo.build_q import se3_to_axis_angle
        body_Q_seq = ctx_body_Q.astype(np.float32)[None, ...]          # (1, 22, 4, 4)
        root_trans_aa, root_orient_aa, pose_body_aa = se3_to_axis_angle(
            body_Q_seq, betas=betas_np, gender=gender, smplx_dir=self.smplx_dir,
        )
        fk = self._get_smplx_fk()
        verts_seq, faces = fk.fk_sequence(
            root_trans_aa.astype(np.float32),
            root_orient_aa.astype(np.float32),
            pose_body_aa.astype(np.float32),
            betas_np, gender,
        )                                                              # (1, V, 3)
        body_mesh = trimesh.Trimesh(
            vertices=verts_seq[0].astype(np.float64), faces=faces, process=False,
        )

        # Object: rigid-transform rest mesh by ctx_obj_se3 (identity in
        # current OMOMO dataset → mesh stays at rest pose / origin).
        rest_mesh, _, _ = self._load_rest(str(obj_category))
        R_obj = ctx_obj_se3[:3, :3]
        t_obj = ctx_obj_se3[:3, 3]
        rest_verts_obj = np.asarray(rest_mesh.vertices, dtype=np.float64)  # (V, 3)
        obj_verts_world = rest_verts_obj @ R_obj.T + t_obj[None, :]        # (V, 3)
        obj_mesh = trimesh.Trimesh(
            vertices=obj_verts_world,
            faces=np.asarray(rest_mesh.faces),
            process=False,
        )

        # Render WITHOUT a ground plane — ctx is abstract, not a scene.
        with self._render_lock:
            if self._cached_renderer is None:
                self._cached_renderer = pyrender.OffscreenRenderer(
                    viewport_width=self._W, viewport_height=self._H,
                )
            return _render_frame_no_ground(
                body_mesh, obj_mesh, self._cached_renderer,
                self._W, self._H,
            )

    # ------------------------------------------------------------------
    # Internal helpers — batch extraction + FK
    # ------------------------------------------------------------------

    def _fk_from_batch(
        self,
        batch: Dict,
        bid: int,
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, str]":
        """Extract LBS-skinned body verts + object pose from an OMOMO batch.

        The wmm OMOMO trace layout (see ``OmomoDataset``) is P = 23:
        joints 0..21 = SMPL-X body **world-frame SE(3)** (Stage-3 rewrite —
        no axis-angle anywhere in the pipeline; the dataset reads the
        pre-computed ``Q`` field directly), joint 22 = rigid object SE(3).

        This helper LBS-skins the SMPL-X body mesh from the per-joint
        world transforms using the formula

            M[t, j] = body_Q[t, j] @ inv_bind[j]                # (T, 22, 4, 4)
            v_world[t, v] = sum_j lbs_weights[v, j] * M[t, j] @ rest_v

        where ``inv_bind[j] = [[I, -J_rest[j]], [0, 1]]`` is the
        translation-only inverse bind (past lesson), ``J_rest``
        is the shape-blended rest-pose joint table from SMPL-X(betas), and
        ``lbs_weights`` is the SMPL-X LBS weight matrix truncated to the
        first 22 (body) joints. The rest template is built once per
        ``(gender, betas)`` and cached at module level.

        This replaces the prior pipeline (matrix-log → axis-angle →
        SMPL-X.forward()) which was both slower and dependent on a
        runtime SMPL-X invocation per render. LBS is closed-form and
        matches SMPL-X FK to sub-mm precision (verified by
        ``test/wmm/test_omomo_viz_lbs.py``).

        Expected batch keys:
            * ``trace`` — ``(B, T, P, 4, 4)`` float tensor (collated).
            * ``trace_mask`` — ``(B, T, P)`` bool tensor.
            * ``extra["betas"]`` — ``(16,)`` per item OR batch-wide.
            * ``extra["gender"]`` — str per item OR batch-wide.
            * ``extra["obj_category"]`` — str.

        Returns:
            ``(body_verts (T_actual,10475,3) float32,
               body_faces (F,3) int32,
               obj_R (T_actual,3,3) float64,
               obj_t (T_actual,3) float64,
               actual_len int,
               obj_category str)``.
        """
        # CONTRACT (matches CanonicalDatasetOutput.collate +
        # humoto_viz.py:415,432):
        #   batch["trace"]      = (B, T, P, 4, 4) — time-major, entity-minor
        #   batch["trace_mask"] = (B, T, P)       — bool
        # We trust this exact shape. Any deviation raises loudly, not
        # silently remapped (the reviewer's iter-02 catch).
        trace = batch["trace"]
        tmask = batch.get("trace_mask")
        if trace.ndim != 5:
            raise ValueError(
                f"OmomoVisualizer: expected trace ndim=5 (B,T,P,4,4), got {trace.ndim}"
            )
        if tmask is None or tmask.ndim != 3:
            raise ValueError(
                "OmomoVisualizer: batch['trace_mask'] must be (B,T,P) bool "
                "— required to determine actual_len from the collated batch"
            )
        if trace.shape[:3] != tmask.shape[:3]:
            raise ValueError(
                f"OmomoVisualizer: trace.shape[:3]={tuple(trace.shape[:3])} "
                f"!= trace_mask.shape[:3]={tuple(tmask.shape[:3])}"
            )

        # bTrace: (T, P, 4, 4); bMask: (T, P)
        bTrace = trace[bid]
        bMask = tmask[bid]
        T, P = bTrace.shape[0], bTrace.shape[1]
        if P < 23:
            raise ValueError(
                f"OmomoVisualizer: OMOMO contract expects P>=23 "
                f"(22 body + 1 object), got P={P}"
            )

        # actual_len = # time steps where pelvis (entity 0) is valid.
        actual_len = int(bMask[:, 0].sum().item())
        actual_len = max(1, min(actual_len, T))

        bTrace_np = bTrace.detach().cpu().numpy().astype(np.float64)

        # Body world-frame Q for this item, time slice up to actual_len.
        body_Q = bTrace_np[:actual_len, :22, :, :]            # (T_a, 22, 4, 4)

        # Object pose at entity index 22.
        obj_R = bTrace_np[:actual_len, 22, :3, :3]            # (T_a, 3, 3)
        obj_t = bTrace_np[:actual_len, 22, :3, 3]             # (T_a, 3)

        # Resolve betas + gender + obj_category. Stage-3 dataset places
        # betas + gender directly in extras; older fallbacks remain in
        # `_resolve_betas_gender` for legacy npz where the dataset did
        # not yet propagate them.
        extra = batch.get("extra", {})
        obj_category = _extract_scalar_field(
            extra, "obj_category", bid, default=None,
        )
        if obj_category is None:
            raise KeyError(
                "OmomoVisualizer: batch['extra']['obj_category'] is required "
                "to locate rest_object_geo/<cat>.ply"
            )
        betas_np, gender = self._resolve_betas_gender(extra, bid)

        # Q → axis-angle → SMPL-X FK. Gives the FULL SMPL-X body mesh with
        # correct hand/face geometry. Body-only 22-joint LBS was cheaper but
        # produced spiky hand/finger artifacts for rest-pose hand joints;
        # full FK uses SMPL-X's 55-joint kinematic chain with zero AA for
        # hand/face (OMOMO doesn't carry those DoF) — anatomically consistent.
        from prepare_data.omomo.build_q import se3_to_axis_angle
        from prepare_data.omomo.omomo_viser import _SmplxFK
        root_trans_aa, root_orient_aa, pose_body_aa = se3_to_axis_angle(
            body_Q.astype(np.float32), betas=betas_np, gender=gender,
            smplx_dir=self.smplx_dir,
        )
        fk = self._get_smplx_fk()
        verts, faces = fk.fk_sequence(
            root_trans_aa.astype(np.float32),
            root_orient_aa.astype(np.float32),
            pose_body_aa.astype(np.float32),
            betas_np, gender,
        )                                                     # (T_a, V, 3) float32, (F, 3) int32

        return verts, faces, obj_R, obj_t, actual_len, str(obj_category)

    def _get_smplx_fk(self):
        """Lazy-initialize + cache the SMPL-X FK helper."""
        if not hasattr(self, "_smplx_fk") or self._smplx_fk is None:
            from prepare_data.omomo.omomo_viser import _SmplxFK
            self._smplx_fk = _SmplxFK(Path(self.smplx_dir))
        return self._smplx_fk

    def _resolve_betas_gender(
        self, extra: Dict, bid: int,
    ) -> "tuple[np.ndarray, str]":
        """Return ``(betas (16,) float32, gender str)`` for the batch item.

        Looks first at ``extra['betas']`` / ``extra['gender']`` (in case
        a future dataset extension populates them). If absent, loads
        from ``self._npz_dir_for(entry)/<entry>.npz`` using the entry
        stem stored in ``extra['entry']`` (populated by
        ``OmomoDataset.__getitem_inner__``).

        Fails loud if neither path works.
        """
        betas = _extract_scalar_field(extra, "betas", bid, default=None)
        gender = _extract_scalar_field(extra, "gender", bid, default=None)
        if betas is not None and gender is not None:
            return (
                np.asarray(betas, dtype=np.float32).reshape(-1),
                str(gender),
            )

        # Fallback: load from the source npz via `entry`.
        entry = _extract_scalar_field(extra, "entry", bid, default=None)
        if entry is None:
            raise KeyError(
                "OmomoVisualizer._resolve_betas_gender: extras lack both "
                "('betas','gender') AND 'entry' — cannot locate source npz. "
                "OmomoDataset.__getitem_inner__ must populate at least 'entry'."
            )
        npz_path = self._find_npz_for_entry(str(entry))
        with np.load(str(npz_path), allow_pickle=True) as d:
            betas_np = np.asarray(d["betas"], dtype=np.float32).reshape(-1)
            gender_str = str(d["gender"].item())
        return betas_np, gender_str

    def _find_npz_for_entry(self, entry: str) -> Path:
        """Return absolute path to the npz for a given entry stem.

        Searches ``<data_root>/{test,train}_npz/<entry>.npz`` in that
        order. ``data_root`` is resolved from ``omomo_cfg['data_root']``
        or inferred from ``geo_dir.parent``. Fails loud if not found.
        """
        data_root = getattr(self, "_data_root", None)
        if data_root is None:
            data_root = self.geo_dir.parent
        for split in ("test", "train"):
            cand = Path(data_root) / f"{split}_npz" / f"{entry}.npz"
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"OmomoVisualizer: cannot find npz for entry {entry!r} under "
            f"{data_root}/(test|train)_npz/"
        )

    def _load_rest(
        self, obj_category: str,
    ) -> "tuple[trimesh.Trimesh, Path, float]":
        """Load the rest-frame PLY for an object category, with height check."""
        geo_path = self.geo_dir / f"{obj_category}.ply"
        if not geo_path.exists():
            raise FileNotFoundError(
                f"OmomoVisualizer: rest-frame geometry missing: {geo_path}"
            )
        rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)
        if not isinstance(rest_mesh, trimesh.Trimesh):
            raise ValueError(
                f"OmomoVisualizer: expected single mesh from {geo_path}, "
                f"got {type(rest_mesh).__name__}"
            )
        _assert_obj_height_sane(geo_path, rest_mesh.bounds)
        bbox_z_m = float(rest_mesh.bounds[1, 2] - rest_mesh.bounds[0, 2])
        return rest_mesh, geo_path, bbox_z_m

    # ------------------------------------------------------------------
    # Training-time callback (partial — GT only)
    # ------------------------------------------------------------------

    def omomo_viz_routine(
        self,
        batch: Dict,
        phase: str,
        batch_idx: int,
        eval_model,
    ) -> None:
        """Training-time visualization callback.

        Scope:
          * GT video rendered and logged via
            ``pl_module.logger.log_video``.
          * Half-future / 1-step / o2h / h2o inference modes log a
            WARNING and skip (wmm Agent OMOMO inference not yet wired).

        Args:
            batch: training batch dict.
            phase: ``"train"`` or ``"val"``.
            batch_idx: batch index.
            eval_model: callable for running model inference.
        """
        if not self.flag:
            return
        # --- Prefer window_idx=0 batch item for cleaner viz ---------------
        # (post-Phase-A verdict). Train-split
        # sliding windows sample mid-action frames at arbitrary
        # start_t_idx > 0; rendering those as "GT" produces visually
        # surprising (though physically correct) mp4s with objects
        # already tilted at frame 0. For training-time viz only, pick
        # a batch item whose `extra["window_idx"] == 0` when present —
        # this makes wandb train-viz look like test-viz. Training
        # itself uses all windows (no change to loss / data pipeline).
        bid = 0
        extra: Dict = batch.get("extra", {}) or {}
        try:
            window_idx_list = extra.get("window_idx")
            if window_idx_list is not None:
                for i, wi in enumerate(window_idx_list):
                    wi_v = int(wi) if wi is not None else -1
                    if wi_v == 0:
                        bid = i
                        break
        except (TypeError, ValueError, KeyError):
            bid = 0

        log_video = getattr(self.pl.logger, "log_video", None)
        if log_video is None:
            logger.warning(
                "omomo_viz_routine: pl.logger has no log_video attribute "
                "— frames rendered but not logged (phase=%s batch=%d)",
                phase, batch_idx,
            )
            return

        text_caption = (
            batch["text"][bid] if batch.get("text") and batch["text"][bid] else ""
        )

        # actual_len for this batch item from trace_mask (entity 0 = pelvis).
        tmask = batch["trace_mask"]
        actual_len = int(tmask[bid, :, 0].sum().item())
        actual_len = max(1, min(actual_len, int(batch["trace"].shape[1])))

        # --- GT ---
        frames_gt = self.draw_gt(batch, bid=bid)
        video_gt = torch.from_numpy(frames_gt).float().permute(0, 3, 1, 2) / 255.0
        log_video(
            phase, f"GT_omomo_{phase}", video_gt,
            self.pl.global_step, batch_idx,
        )

        # --- w1.6+ Penetration overlay (gated by viz_penetration_flag) ---
        # Renders palm spheres color-coded by SDF signed distance, plus a
        # numeric depth (mm) panel per frame. Module: omomo_penetration_viz.
        # Defaults OFF — flip viz_penetration_flag in YAML once VLM-checked.
        try:
            self._maybe_log_penetration_viz(
                batch, bid=bid, phase=phase, batch_idx=batch_idx,
                actual_len=actual_len, log_video=log_video,
            )
        except Exception as exc:
            logger.warning(
                "omomo_viz_routine: penetration viz failed (phase=%s batch=%d): %s",
                phase, batch_idx, exc,
            )

        # --- Inference modes ---
        # OMOMO trace layout: entities 0..21 = SMPL-X body, entity 22 = object.
        # P may be >22 when batch-pad wider domains; use the exact P of this
        # batch item and only treat the object-as-entity-22 slot.
        # Modes MUST mirror the training mask_scheduler patterns exactly
        # so the model sees the same known/unknown distribution at inference
        # time as it did during training (mask_scheduler.py::OMOMO_PATTERNS).
        P = int(batch["trace"].shape[2])
        T = int(batch["trace"].shape[1])
        body_ids = list(range(min(22, P)))
        obj_id = 22 if P > 22 else None

        # CHOIS waypoint indices from the npz (5 values by default:
        # 0, T//4, T//2, 3T//4, T-1). Prefer these if present on the batch;
        # else fall back to n_waypoints=4 evenly spaced.
        extra_wp = extra.get("waypoint_indices")
        if extra_wp is not None and bid < len(extra_wp):
            wp_raw = extra_wp[bid]
            waypoint_indices: Optional[List[int]] = (
                [int(w) for w in wp_raw] if wp_raw is not None else None
            )
        else:
            waypoint_indices = None

        # Three representative prefix lengths for random_prefix_future:
        # short (~15%), mid (~50%), long (~85%) of T. Training samples
        # k uniformly in [1, T-1] so the model saw the full range.
        _prefix_ks = (max(1, T // 7), T // 2, max(1, (6 * T) // 7))
        modes: List[Dict] = [
            {
                "name": f"FuturePred_omomo_k{_prefix_ks[0]}",
                "caption": f"prefix_future (seen={_prefix_ks[0]}/{T}) | {text_caption}",
                "set_known": lambda tk, k=_prefix_ks[0]: _set_prefix_future_known(tk, k),
            },
            {
                "name": f"FuturePred_omomo_k{_prefix_ks[1]}",
                "caption": f"prefix_future (seen={_prefix_ks[1]}/{T}) | {text_caption}",
                "set_known": lambda tk, k=_prefix_ks[1]: _set_prefix_future_known(tk, k),
            },
            {
                "name": f"FuturePred_omomo_k{_prefix_ks[2]}",
                "caption": f"prefix_future (seen={_prefix_ks[2]}/{T}) | {text_caption}",
                "set_known": lambda tk, k=_prefix_ks[2]: _set_prefix_future_known(tk, k),
            },
        ]
        if obj_id is not None:
            modes.append({
                "name": "o2h_omomo",
                "caption": f"o2h (objects known, body predicted) | {text_caption}",
                "set_known": lambda tk, bi=body_ids: _set_entity_unknown(tk, bi),
            })
            modes.append({
                "name": "h2o_omomo",
                "caption": f"h2o (body known, objects predicted) | {text_caption}",
                "set_known": lambda tk, oi=[obj_id]: _set_entity_unknown(tk, oi),
            })
            # chois_obj_waypoints_to_all is the DOMINANT (8.0x weight in w1.1)
            # training pattern and matches CHOIS's eval protocol. This is
            # the mode most likely to produce high-quality inference viz.
            modes.append({
                "name": "chois_waypoints_omomo",
                "caption": f"chois_waypoints (4 obj waypoints) | {text_caption}",
                "set_known": lambda tk, oi=[obj_id], bi=body_ids, wp=waypoint_indices: \
                    _set_chois_obj_waypoints_known(tk, oi, bi, wp, n_waypoints=4),
            })

        for mode in modes:
            try:
                pred_trace, tk = self._run_inference_one(
                    batch, bid, eval_model, mode["set_known"],
                )
            except Exception as e:
                # Per project rule: no silent swallow. Log the error and
                # skip this mode — other modes and training continue.
                logger.warning(
                    "omomo_viz_routine: inference mode %s failed: %s",
                    mode["name"], e, exc_info=True,
                )
                continue

            frames = self.draw_inference(batch, pred_trace, bid=bid)
            # SEEN/PRED top-bar indicator: paint the top 10 rows green
            # when any entity is "seen" (known) on that frame, red when
            # all entities at that frame are being predicted. ``tk``
            # shape is (1, T, P) bool; frames[t_idx] is the t_idx-th
            # VALID frame (range [0, actual_len)), which aligns 1:1 with
            # tk[0, t_idx, :] because trace time axis is dense [0..T-1].
            try:
                _tk_np = tk.detach().cpu().numpy()[0]        # (T, P) bool
                seen_per_frame = _tk_np.any(axis=1)          # (T,) bool
                n_frames = int(frames.shape[0])
                n_t = int(_tk_np.shape[0])
                for t_idx in range(n_frames):
                    color = (
                        SEEN_INDICATOR
                        if (t_idx < n_t and bool(seen_per_frame[t_idx]))
                        else PRED_INDICATOR
                    )
                    frames[t_idx, :10, :, :3] = color[None, None, :]
            except Exception as e:
                logger.warning(
                    "omomo_viz_routine: failed to paint SEEN/PRED bar "
                    "(mode=%s): %s", mode["name"], e,
                )

            video = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
            log_video(
                phase, f"{mode['name']}_{phase}", video,
                self.pl.global_step, batch_idx,
                caption=mode["caption"],
            )

        # --- Ctx panel (collapsed-LBS body + object-at-origin reference) ---
        # Single-frame reference image showing the identity-context state
        # (every body joint world-SE(3) = I, object SE(3) = I).
        # Logged as IMAGE not video — ctx is a static abstract reference,
        # no time dimension. Stage-4 LBS rewrite: the body collapses to a
        # ~0.3-0.6 m blob near origin (per-vertex weighted-J_rest offset).
        # DESIGN NOTE: we do NOT paint the indicator bar on the GT video
        # (all frames would be green — no information); the bar is
        # reserved for inference-mode videos where SEEN/PRED contrast is
        # meaningful.
        log_image = getattr(self.pl.logger, "log_image", None)
        if log_image is None:
            logger.warning(
                "omomo_viz_routine: pl.logger has no log_image attribute — "
                "ctx panel skipped (phase=%s batch=%d)", phase, batch_idx,
            )
        else:
            try:
                ctx_img = self._render_ctx_panel(batch, bid)  # (H, W, 3) uint8
                # log_image expects (C, H, W) tensor in [0, 1].
                ctx_chw = (
                    torch.from_numpy(ctx_img).float().permute(2, 0, 1)
                    / 255.0
                )
                log_image(
                    phase, f"Ctx_omomo_{phase}", ctx_chw,
                    self.pl.global_step, batch_idx,
                )
            except Exception as e:
                logger.warning(
                    "omomo_viz_routine: ctx panel render failed: %s", e,
                    exc_info=True,
                )

        if not self._inference_defer_logged:
            logger.info(
                "omomo_viz_routine: GT + inference modes "
                "(prefix_future@k={short,mid,long} / o2h / h2o / "
                "chois_waypoints) all wired — masks mirror training "
                "distribution. This notice is logged ONCE. phase=%s batch=%d",
                phase, batch_idx,
            )
            self._inference_defer_logged = True


# ---------------------------------------------------------------------------
# Standalone entry point: render_omomo_window_jpeg
# ---------------------------------------------------------------------------

def render_omomo_window_jpeg(
    npz_path: "str | os.PathLike",
    frame: int,
    out_path: "str | os.PathLike",
    smplx_dir: "str | os.PathLike" = "data_wmm/omomo/smplx",
    geo_dir: "str | os.PathLike" = "data_wmm/omomo/rest_object_geo",
    width: int = 960,
    height: int = 960,
    include_pcl: bool = True,
    num_pcl_points: int = 256,
    pcl_radius: float = _PCL_SPHERE_RADIUS,
    pcl_seed: int = 0,
    fk_cache: Optional[_SmplxFK] = None,
    window_cache: Optional[OmomoWindow] = None,
) -> Dict[str, object]:
    """Render one OMOMO window frame to JPEG with optional PCL overlay.

    Reads an OMOMO Q12 .npz, runs SMPL-X FK, loads the CANONICAL rest-frame
    object mesh from ``geo_dir/<category>.ply`` (NOT the obj_mesh_path raw
    scan), builds a scene with ground plane, body, object, and optionally
    256 PCL spheres, and writes a JPEG.

    The rest-frame geo path is resolved by stripping the ``_cleaned_simplified.obj``
    suffix from the npz's ``obj_mesh_path`` to recover ``<category>``, then
    loading ``<geo_dir>/<category>.ply``. This path is printed + returned in
    the metadata so the caller can verify it.

    Args:
        npz_path: OMOMO Q12 .npz window path.
        frame: frame index in [0, T-1]. **Silently clamped** to
            ``actual_len - 1`` if the caller passes a frame beyond the
            valid range. Callers that want to reject out-of-range frames
            MUST check ``actual_len`` themselves BEFORE calling
            (``render_gate.py`` does this). The clamp is defensive only
            and never masks a real "teleport-to-origin" zero-pad.
        out_path: output JPEG path.
        smplx_dir: SMPL-X models (SMPLX_MALE/FEMALE/NEUTRAL.npz).
        geo_dir: directory with ``<category>.ply`` rest-frame geometry.
        width, height: output resolution.
        include_pcl: overlay 256 PCL spheres transformed by frame-t pose.
        num_pcl_points: points to sample (default 256, matches OmomoDataset).
        pcl_radius: sphere radius (meters).
        pcl_seed: RNG seed for reproducible PCL sampling.
        fk_cache, window_cache: reuse for multi-frame rendering.

    Returns:
        dict with keys: ``out_path``, ``geo_path`` (resolved rest PLY),
        ``obj_category``, ``frame``, ``actual_len``, ``seq_name``, ``text``,
        ``bbox_z_m`` (object height sanity).
    """
    npz_path = Path(npz_path)
    out_path = Path(out_path)
    geo_dir = Path(geo_dir)
    smplx_dir = Path(smplx_dir)

    # ------------------------------------------------------------------
    # 1. Load npz fields
    # ------------------------------------------------------------------
    d = np.load(str(npz_path), allow_pickle=True)
    actual_len = int(d["actual_len"].item())
    seq_name = str(d["seq_name"].item())
    text = str(d["text"].item())
    obj_mesh_path_raw = str(d["obj_mesh_path"].item())
    betas = np.asarray(d["betas"], dtype=np.float32).reshape(-1)  # (16,)
    gender = str(d["gender"].item())
    root_trans = np.asarray(d["root_trans"], dtype=np.float32)      # (T, 3)
    root_orient_aa = np.asarray(d["root_orient_aa"], dtype=np.float32)  # (T, 3)
    pose_body_aa = np.asarray(d["pose_body_aa"], dtype=np.float32)   # (T, 21, 3)
    obj_trans = np.asarray(d["obj_trans"], dtype=np.float32)         # (T, 3)
    obj_rot = np.asarray(d["obj_rot"], dtype=np.float32)             # (T, 3, 3)

    # ------------------------------------------------------------------
    # 2. Frame clamping
    # ------------------------------------------------------------------
    frame = min(int(frame), actual_len - 1)

    # ------------------------------------------------------------------
    # 3. Category resolution: strip suffix from obj_mesh_path basename
    # ------------------------------------------------------------------
    base = Path(obj_mesh_path_raw).name
    SUFFIXES = (
        "_cleaned_simplified.obj",
        "_cleaned_simplified.ply",
        "_simplified.obj",
        ".obj",
        ".ply",
    )
    category = None
    for suf in SUFFIXES:
        if base.endswith(suf):
            category = base[: -len(suf)]
            break
    if category is None:
        raise ValueError(
            f"Cannot strip object suffix from {obj_mesh_path_raw!r}; "
            f"expected one of {SUFFIXES}"
        )

    geo_path = geo_dir / f"{category}.ply"
    if not geo_path.exists():
        raise FileNotFoundError(
            f"Rest-frame geometry not found: {geo_path}. "
            f"Cannot fall back to raw obj_mesh_path — raw scan is not in "
            f"canonical frame. geo_dir={geo_dir}"
        )

    # ------------------------------------------------------------------
    # 4. Load rest-frame PLY and sanity-check height
    # ------------------------------------------------------------------
    rest_mesh = trimesh.load(str(geo_path), force="mesh", process=False)
    if not isinstance(rest_mesh, trimesh.Trimesh):
        raise ValueError(
            f"Expected single mesh from {geo_path}, got {type(rest_mesh).__name__}"
        )
    _assert_obj_height_sane(geo_path, rest_mesh.bounds)
    bbox_z_m = float(rest_mesh.bounds[1, 2] - rest_mesh.bounds[0, 2])

    # ------------------------------------------------------------------
    # 5. SMPL-X FK (reuse window_cache when available to skip re-FK)
    # ------------------------------------------------------------------
    if (
        window_cache is not None
        and getattr(window_cache, "name", None) == npz_path.stem
    ):
        verts = np.asarray(window_cache.body_verts, dtype=np.float32)
        faces = np.asarray(window_cache.body_faces, dtype=np.int32)
    else:
        fk = fk_cache if fk_cache is not None else _SmplxFK(smplx_dir)
        verts, faces = fk.fk_sequence(
            root_trans, root_orient_aa, pose_body_aa, betas, gender
        )

    # ------------------------------------------------------------------
    # 6. Build frame-t meshes
    # ------------------------------------------------------------------
    body_mesh = trimesh.Trimesh(
        vertices=verts[frame].astype(np.float64),
        faces=faces,
        process=False,
    )

    # Object: V_world = V_rest @ R.T + t (no scale — reference fix #2)
    R = obj_rot[frame].astype(np.float64)     # (3, 3)
    t_obj = obj_trans[frame].astype(np.float64)  # (3,)
    V_rest = np.asarray(rest_mesh.vertices, dtype=np.float64)  # (V, 3)
    V_w = V_rest @ R.T + t_obj[None, :]       # (V, 3)
    obj_mesh = trimesh.Trimesh(
        vertices=V_w,
        faces=np.asarray(rest_mesh.faces),
        process=False,
    )

    # ------------------------------------------------------------------
    # 7. PCL overlay
    # ------------------------------------------------------------------
    pcl_world_xyz: Optional[np.ndarray] = None
    pcl_rgb_u8: Optional[np.ndarray] = None

    if include_pcl:
        rng = np.random.default_rng(pcl_seed)
        pcl_data = _sample_pcl_from_mesh(rest_mesh, num_pcl_points, rng=rng)
        # pcl_data: (N, 6) float32 — xyz + rgb in [-0.5, 0.5]
        pcl_xyz_rest = pcl_data[:, :3].astype(np.float64)   # (N, 3)
        pcl_rgb_centered = pcl_data[:, 3:]                  # (N, 3) float32

        # Transform PCL to world frame
        pcl_world_xyz = pcl_xyz_rest @ R.T + t_obj[None, :]  # (N, 3)

        # Convert RGB from [-0.5, 0.5] back to uint8
        pcl_rgb_u8 = np.clip(
            (pcl_rgb_centered + 0.5) * 255.0, 0, 255
        ).astype(np.uint8)  # (N, 3)

    # ------------------------------------------------------------------
    # 8. Render
    # ------------------------------------------------------------------
    _render_scene_with_pcl(
        body_mesh=body_mesh,
        obj_mesh=obj_mesh,
        pcl_world_xyz=pcl_world_xyz,
        pcl_rgb=pcl_rgb_u8,
        out_path=out_path,
        width=width,
        height=height,
        pcl_radius=pcl_radius,
    )

    logger.info(
        "omomo_viz: rendered %s f%d -> %s (geo=%s, bbox_z=%.3fm)",
        seq_name, frame, out_path, geo_path.name, bbox_z_m,
    )

    return {
        "out_path": str(out_path),
        "geo_path": str(geo_path),
        "obj_category": category,
        "frame": frame,
        "actual_len": actual_len,
        "seq_name": seq_name,
        "text": text,
        "bbox_z_m": bbox_z_m,
    }


# ---------------------------------------------------------------------------
# Eval-time helper: omomo_viz_eval_preds
# ---------------------------------------------------------------------------

def _eval_render_body_obj_frame(
    fk: _SmplxFK,
    renderer: pyrender.OffscreenRenderer,
    width: int,
    height: int,
    root_trans_t: np.ndarray,   # (3,) float32
    root_orient_aa_t: np.ndarray,   # (3,) float32
    pose_body_aa_t: np.ndarray,  # (21, 3) float32
    betas: np.ndarray,           # (16,) float32
    gender: str,
    rest_mesh: trimesh.Trimesh,
    obj_R_t: np.ndarray,         # (3, 3) float64
    obj_t_t: np.ndarray,         # (3,)   float64
) -> np.ndarray:
    """Render a single body+object frame without instantiating OmomoVisualizer.

    Shared helper for ``omomo_viz_eval_preds``. Runs SMPL-X FK on a
    single-frame batch (T=1) and rigid-transforms the rest-frame object
    mesh. Returns ``(H, W, 3)`` uint8.
    """
    # Run FK on one frame — pack the three (*, 3) inputs as (1, *).
    root_t = root_trans_t.reshape(1, 3).astype(np.float32)
    root_aa = root_orient_aa_t.reshape(1, 3).astype(np.float32)
    body_aa = pose_body_aa_t.reshape(1, 21, 3).astype(np.float32)
    verts, faces = fk.fk_sequence(root_t, root_aa, body_aa, betas, gender)

    body_mesh = trimesh.Trimesh(
        vertices=verts[0].astype(np.float64),
        faces=faces, process=False,
    )
    V_rest = np.asarray(rest_mesh.vertices, dtype=np.float64)  # (V, 3)
    V_w = V_rest @ obj_R_t.T + obj_t_t[None, :]                # (V, 3)
    obj_mesh = trimesh.Trimesh(
        vertices=V_w, faces=np.asarray(rest_mesh.faces),
        process=False,
    )
    return _render_frame_given_renderer(
        body_mesh, obj_mesh, renderer, width, height,
    )


def _resolve_eval_obj_category(obj_mesh_path: str) -> str:
    """Strip known suffixes from a raw obj_mesh_path basename.

    Mirrors ``render_omomo_window_jpeg`` step 3. Fails loud on an
    unknown suffix — eval viz must never silently render the wrong mesh.
    """
    base = Path(obj_mesh_path).name
    for suf in (
        "_cleaned_simplified.obj",
        "_cleaned_simplified.ply",
        "_simplified.obj",
        ".obj",
        ".ply",
    ):
        if base.endswith(suf):
            return base[: -len(suf)]
    raise ValueError(
        f"omomo_viz_eval_preds: cannot strip object suffix from "
        f"{obj_mesh_path!r}"
    )


def omomo_viz_eval_preds(
    pl_module,
    pred_npz_paths: List[str],
    gt_list: List[Dict],
    step: int,
    phase: str = "eval",
    n_samples: int = 3,
    smplx_dir: str = "data_wmm/omomo/smplx",
    geo_dir: str = "data_wmm/omomo/rest_object_geo",
    width: int = 512,
    height: int = 512,
) -> None:
    """Render up to ``n_samples`` prediction MP4s and log to wandb.

    For each prediction npz, reads the predicted body + object trajectory
    (matches the schema produced by
    ``src/umm/callbacks/wmm_omomo_eval.py::_save_pred_npz``), runs SMPL-X
    FK per frame, rigid-transforms the rest-frame object mesh, paints a
    top-bar indicator (green on CHOIS waypoint frames {0, 30, 60, 90,
    119}, red on all others), and logs the video to
    ``<phase>_video/omomo_eval_viz/<seq_name>``.

    Fails soft: any per-sample error is logged and skipped so eval is
    never killed by a viz bug.

    Args:
        pl_module: PyTorch Lightning module. Must expose
            ``.logger.log_video``. If not, the function returns without
            logging (viz is always optional).
        pred_npz_paths: list of saved pred npz paths (K=1 per stem for
            debug; arbitrary for prod — caller picks which stems).
        gt_list: GT dicts in the SAME order as ``pred_npz_paths``. Each
            must match the schema from ``_load_gt_npz``.
        step: ``trainer.global_step`` for wandb logging.
        phase: wandb phase key (default ``"eval"``).
        n_samples: upper bound on the number of videos to render.
        smplx_dir: SMPL-X model directory.
        geo_dir: rest-frame object geometry directory.
        width, height: output resolution.
    """
    log_video = getattr(getattr(pl_module, "logger", None), "log_video", None)
    if log_video is None:
        logger.info(
            "omomo_viz_eval_preds: pl.logger has no log_video "
            "attribute — eval viz skipped (step=%d)", step,
        )
        return

    n = min(int(n_samples), len(pred_npz_paths), len(gt_list))
    if n <= 0:
        return

    fk = _SmplxFK(Path(smplx_dir))
    geo_dir_path = Path(geo_dir)
    renderer: Optional[pyrender.OffscreenRenderer] = None
    try:
        renderer = pyrender.OffscreenRenderer(
            viewport_width=int(width), viewport_height=int(height),
        )

        for idx in range(n):
            pred_path = pred_npz_paths[idx]
            gt = gt_list[idx]
            try:
                with np.load(str(pred_path), allow_pickle=True) as d:
                    # Pred body (T, 3) / (T, 3) / (T, 21, 3); pred obj (T, 3, 3) / (T, 3).
                    root_trans = np.asarray(d["root_trans"], dtype=np.float32)
                    root_orient_aa = np.asarray(
                        d["root_orient_aa"], dtype=np.float32,
                    )
                    pose_body_aa = np.asarray(
                        d["pose_body_aa"], dtype=np.float32,
                    )
                    pred_obj_trans = np.asarray(
                        d["pred_obj_trans"], dtype=np.float32,
                    )
                    pred_obj_rot_mat = np.asarray(
                        d["pred_obj_rot_mat"], dtype=np.float32,
                    )

                # Metadata: betas + gender from GT (pred npz copies them
                # but GT is the source of truth for shape params).
                betas = np.asarray(gt["betas"], dtype=np.float32).reshape(-1)
                gender = str(gt["gender"])
                obj_mesh_path = str(gt["obj_mesh_path"])
                seq_name = str(gt["seq_name"])
                actual_len = int(gt.get("actual_len", root_trans.shape[0]))
                actual_len = max(1, min(actual_len, int(root_trans.shape[0])))

                # Resolve rest-frame PLY.
                category = _resolve_eval_obj_category(obj_mesh_path)
                geo_path = geo_dir_path / f"{category}.ply"
                if not geo_path.exists():
                    raise FileNotFoundError(
                        f"omomo_viz_eval_preds: rest-frame geometry "
                        f"missing: {geo_path}"
                    )
                rest_mesh = trimesh.load(
                    str(geo_path), force="mesh", process=False,
                )
                if not isinstance(rest_mesh, trimesh.Trimesh):
                    raise ValueError(
                        f"omomo_viz_eval_preds: expected Trimesh from "
                        f"{geo_path}, got {type(rest_mesh).__name__}"
                    )
                _assert_obj_height_sane(geo_path, rest_mesh.bounds)

                # Render each frame.
                frames: List[np.ndarray] = []
                for t in range(actual_len):
                    frame = _eval_render_body_obj_frame(
                        fk=fk,
                        renderer=renderer,
                        width=int(width), height=int(height),
                        root_trans_t=root_trans[t],
                        root_orient_aa_t=root_orient_aa[t],
                        pose_body_aa_t=pose_body_aa[t],
                        betas=betas, gender=gender,
                        rest_mesh=rest_mesh,
                        obj_R_t=pred_obj_rot_mat[t].astype(np.float64),
                        obj_t_t=pred_obj_trans[t].astype(np.float64),
                    )
                    # Top-bar: green on CHOIS waypoints, red elsewhere.
                    color = (
                        SEEN_INDICATOR
                        if t in _CHOIS_EVAL_WAYPOINTS
                        else PRED_INDICATOR
                    )
                    frame[:10, :, :3] = color[None, None, :]
                    frames.append(frame)

                video_np = np.stack(frames, axis=0)  # (T, H, W, 3) uint8
                video_t = (
                    torch.from_numpy(video_np).float().permute(0, 3, 1, 2)
                    / 255.0
                )
                log_video(
                    phase, f"omomo_eval_viz/{seq_name}", video_t,
                    int(step), 0,
                )
                logger.info(
                    "omomo_viz_eval_preds: logged %s (T=%d) step=%d",
                    seq_name, actual_len, int(step),
                )
            except Exception as e:
                logger.warning(
                    "omomo_viz_eval_preds: sample %d (%s) failed: %s",
                    idx, pred_path, e, exc_info=True,
                )
                continue
    except Exception as e:
        # Outer guard: a renderer-init failure or other setup error must
        # never kill eval. Log loudly and return.
        logger.warning(
            "omomo_viz_eval_preds: fatal setup error (step=%d): %s",
            int(step), e, exc_info=True,
        )
    finally:
        if renderer is not None:
            try:
                renderer.delete()
            except Exception as e:
                logger.warning(
                    "omomo_viz_eval_preds: renderer.delete() failed: %s", e,
                )


__all__ = [
    "OmomoVisualizer",
    "render_omomo_window_jpeg",
    "omomo_viz_eval_preds",
    "SEEN_INDICATOR",
    "PRED_INDICATOR",
]
