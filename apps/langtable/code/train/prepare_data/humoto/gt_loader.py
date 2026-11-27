"""Independent ground-truth loader for HUMOTO glb files.

This module is intentionally INDEPENDENT of `glb_parse.py` and `humoto_seq.py`.
It uses `pygltflib` (a different binary reader and accessor interface) to parse
the same glb, then reimplements the minimum FK + LBS arithmetic needed to skin
the main body sub-mesh at one frame.

Coordinate frame: the returned vertices are in the RAW glTF coordinate system
(Y-up, no R_WORLD rotation, no unit conversion). The caller is responsible for
applying the same coordinate transform on its side before comparing.

Public API:
    skin_body_at(glb_path, frame) -> (V, 3) float64 ndarray

Everything else is private (`_` prefix).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pygltflib  # pyright: ignore[reportMissingImports]
from pygltflib import GLTF2  # pyright: ignore[reportMissingImports]

# glTF componentType -> numpy dtype
_COMP_DTYPE: Dict[int, np.dtype] = {
    5120: np.dtype(np.int8),
    5121: np.dtype(np.uint8),
    5122: np.dtype(np.int16),
    5123: np.dtype(np.uint16),
    5125: np.dtype(np.uint32),
    5126: np.dtype(np.float32),
}

# glTF type -> component count
_TYPE_NCOMP: Dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _read_accessor(gltf: GLTF2, blob: bytes, acc_idx: int) -> np.ndarray:
    """Decode accessor `acc_idx` from `blob`, fully independent of glb_parse."""
    acc = gltf.accessors[acc_idx]
    if acc.bufferView is None:
        raise ValueError(f"Accessor {acc_idx} has no bufferView (sparse not supported)")
    bv = gltf.bufferViews[acc.bufferView]
    comp = acc.componentType
    if comp not in _COMP_DTYPE:
        raise ValueError(f"Unsupported componentType {comp}")
    dtype = _COMP_DTYPE[comp]
    ncomp = _TYPE_NCOMP[acc.type]
    count = acc.count
    elem_size = dtype.itemsize * ncomp

    base = (bv.byteOffset or 0) + (acc.byteOffset or 0)
    stride = bv.byteStride or elem_size

    if stride == elem_size:
        raw = np.frombuffer(
            blob, dtype=dtype, count=count * ncomp, offset=base
        ).reshape(count, ncomp).copy()
    else:
        raw = np.empty((count, ncomp), dtype=dtype)
        for i in range(count):
            off = base + i * stride
            raw[i] = np.frombuffer(blob, dtype=dtype, count=ncomp, offset=off)

    if acc.type == "SCALAR":
        return raw.reshape(count)
    if acc.type == "MAT4":
        # glTF stores MAT4 column-major. Convert to row-major (math) form.
        return raw.reshape(count, 4, 4).transpose(0, 2, 1).astype(np.float64)
    return raw  # (count, ncomp)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix. Different formula style
    than glb_parse._quat_to_R to make this an independent implementation."""
    x = float(q[0]); y = float(q[1]); z = float(q[2]); w = float(q[3])
    n2 = x * x + y * y + z * z + w * w
    if n2 < 1e-20:
        return np.eye(3, dtype=np.float64)
    # Normalize first (alt path): produces same result up to fp noise.
    inv = 1.0 / np.sqrt(n2)
    x *= inv; y *= inv; z *= inv; w *= inv
    xx = x * x; yy = y * y; zz = z * z
    xy = x * y; xz = x * z; yz = y * z
    wx = w * x; wy = w * y; wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
            [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _slerp(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    """Shortest-arc slerp between two unit quaternions (x, y, z, w)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + a * (q1 - q0)
        out = out / (np.linalg.norm(out) + 1e-20)
        return out
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    s = float(np.sin(theta))
    s0 = float(np.sin((1.0 - a) * theta)) / s
    s1 = float(np.sin(a * theta)) / s
    return s0 * q0 + s1 * q1


def _trs_to_mat(
    t: np.ndarray,
    r: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """TRS -> 4x4 row-major matrix. Independent reimplementation."""
    M = np.eye(4, dtype=np.float64)
    R = _quat_to_mat(r)
    # Multiply scale into rotation columns (glTF: M = T * R * S).
    M[0, 0] = R[0, 0] * s[0]; M[0, 1] = R[0, 1] * s[1]; M[0, 2] = R[0, 2] * s[2]
    M[1, 0] = R[1, 0] * s[0]; M[1, 1] = R[1, 1] * s[1]; M[1, 2] = R[1, 2] * s[2]
    M[2, 0] = R[2, 0] * s[0]; M[2, 1] = R[2, 1] * s[1]; M[2, 2] = R[2, 2] * s[2]
    M[0, 3] = float(t[0]); M[1, 3] = float(t[1]); M[2, 3] = float(t[2])
    return M


def _node_rest_local(node: pygltflib.Node) -> np.ndarray:
    """Rest local matrix of a node, from `matrix` or TRS fields."""
    if node.matrix is not None:
        # glTF matrix is column-major flat list
        m = np.asarray(node.matrix, dtype=np.float64).reshape(4, 4).T
        return m
    t = np.asarray(node.translation if node.translation is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    r = np.asarray(node.rotation if node.rotation is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    s = np.asarray(node.scale if node.scale is not None else [1.0, 1.0, 1.0], dtype=np.float64)
    return _trs_to_mat(t, r, s)


def _build_parents(gltf: GLTF2) -> np.ndarray:
    n = len(gltf.nodes)
    parent = np.full(n, -1, dtype=np.int64)
    for i, nd in enumerate(gltf.nodes):
        for c in (nd.children or []):
            ci = int(c)
            if parent[ci] != -1:
                raise ValueError(f"Node {ci} has multiple parents")
            parent[ci] = i
    return parent


def _eval_sampler_at_frame(
    gltf: GLTF2,
    blob: bytes,
    sampler: pygltflib.AnimationSampler,
    canonical_time: float,
    cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
    sampler_idx: int,
    is_rotation: bool,
) -> np.ndarray:
    """Evaluate one sampler at `canonical_time`. Returns the (dim,) output value.

    Handles LINEAR + STEP. For rotation channels uses slerp on LINEAR.
    """
    interp = sampler.interpolation or "LINEAR"
    if interp not in ("LINEAR", "STEP"):
        raise ValueError(f"Unsupported sampler interpolation {interp}")

    if sampler_idx in cache:
        in_times, out_vals = cache[sampler_idx]
    else:
        in_times = _read_accessor(gltf, blob, sampler.input).astype(np.float64).reshape(-1)
        out_raw = _read_accessor(gltf, blob, sampler.output).astype(np.float64)
        if out_raw.ndim == 1:
            out_vals = out_raw.reshape(-1, 1)
        else:
            out_vals = out_raw
        cache[sampler_idx] = (in_times, out_vals)

    n = in_times.shape[0]
    if canonical_time <= in_times[0]:
        return out_vals[0].copy()
    if canonical_time >= in_times[-1]:
        return out_vals[-1].copy()
    idx = int(np.searchsorted(in_times, canonical_time, side="right") - 1)
    idx = max(0, min(n - 2, idx))
    if interp == "STEP":
        return out_vals[idx].copy()
    t0 = float(in_times[idx]); t1 = float(in_times[idx + 1])
    a = (canonical_time - t0) / (t1 - t0) if t1 > t0 else 0.0
    if is_rotation:
        return _slerp(out_vals[idx], out_vals[idx + 1], a)
    return (1.0 - a) * out_vals[idx] + a * out_vals[idx + 1]


def _node_local_at_time(
    gltf: GLTF2,
    blob: bytes,
    node_idx: int,
    canonical_time: float,
    node_path_sampler: Dict[int, Dict[str, int]],
    sampler_cache: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Local 4x4 of node `node_idx` at `canonical_time` from animation sampler 0
    (the skeleton animation). Falls back to the node's rest TRS for any path
    that has no channel.
    """
    nd = gltf.nodes[node_idx]
    paths = node_path_sampler.get(node_idx, {})

    if nd.matrix is not None and not paths:
        # Pure matrix-only static node (no animation).
        return _node_rest_local(nd)

    rest_t = np.asarray(nd.translation if nd.translation is not None else [0.0, 0.0, 0.0], dtype=np.float64)
    rest_r = np.asarray(nd.rotation if nd.rotation is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    rest_s = np.asarray(nd.scale if nd.scale is not None else [1.0, 1.0, 1.0], dtype=np.float64)

    anim = gltf.animations[0]

    if "translation" in paths:
        t_val = _eval_sampler_at_frame(
            gltf, blob, anim.samplers[paths["translation"]], canonical_time,
            sampler_cache, paths["translation"], is_rotation=False,
        )
    else:
        t_val = rest_t
    if "rotation" in paths:
        r_val = _eval_sampler_at_frame(
            gltf, blob, anim.samplers[paths["rotation"]], canonical_time,
            sampler_cache, paths["rotation"], is_rotation=True,
        )
    else:
        r_val = rest_r
    if "scale" in paths:
        s_val = _eval_sampler_at_frame(
            gltf, blob, anim.samplers[paths["scale"]], canonical_time,
            sampler_cache, paths["scale"], is_rotation=False,
        )
    else:
        s_val = rest_s

    return _trs_to_mat(t_val, r_val, s_val)


def _topo_order(parent: np.ndarray) -> List[int]:
    n = parent.shape[0]
    children: List[List[int]] = [[] for _ in range(n)]
    roots: List[int] = []
    for i in range(n):
        p = int(parent[i])
        if p == -1:
            roots.append(i)
        else:
            children[p].append(i)
    order: List[int] = []
    stack = list(roots)
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i])
    if len(order) != n:
        raise ValueError(f"Topo failed: {len(order)} of {n}")
    return order


def _canonical_time_array(
    gltf: GLTF2, blob: bytes
) -> np.ndarray:
    """Pick the longest input sampler in animations[0] as the canonical timeline."""
    anim = gltf.animations[0]
    best: Optional[np.ndarray] = None
    best_n = -1
    for samp in anim.samplers:
        in_acc = gltf.accessors[samp.input]
        if in_acc.count > best_n:
            best_n = in_acc.count
            best = _read_accessor(gltf, blob, samp.input).astype(np.float64).reshape(-1)
    if best is None:
        raise ValueError("Animation 0 has no samplers")
    return best


def skin_body_at(glb_path: str, frame: int) -> np.ndarray:
    """Independent LBS of the main body sub-mesh at frame `frame`.

    Pipeline: pygltflib parse -> build parent array -> evaluate per-node local
    TRS at frame `frame` from animation samplers -> FK to joint globals (raw
    glTF coordinate frame, NO Z-up rotation, NO unit conversion) -> LBS on the
    LARGEST skinned sub-mesh.

    Returns: (V, 3) float64 world vertex positions in raw glTF coordinates.
    """
    gltf = GLTF2.load_binary(glb_path)
    if gltf is None:
        raise ValueError(f"Failed to load glb {glb_path}")
    blob = gltf.binary_blob()
    if blob is None or len(blob) == 0:
        raise ValueError(f"Empty binary blob in {glb_path}")

    n_nodes = len(gltf.nodes)
    parent = _build_parents(gltf)
    order = _topo_order(parent)

    if len(gltf.skins) != 1:
        raise ValueError(f"Expected exactly 1 skin, got {len(gltf.skins)}")
    skin = gltf.skins[0]
    joints: List[int] = list(skin.joints)
    if skin.inverseBindMatrices is None:
        raise ValueError("Skin has no inverseBindMatrices")
    inv_bind = _read_accessor(gltf, blob, skin.inverseBindMatrices)  # (J,4,4)
    if inv_bind.shape[0] != len(joints):
        raise ValueError(f"inv_bind {inv_bind.shape} mismatch joints {len(joints)}")

    if not gltf.animations:
        raise ValueError("No animations")
    anim = gltf.animations[0]

    # node -> {path -> sampler_idx}
    node_path_sampler: Dict[int, Dict[str, int]] = {}
    for ch in anim.channels:
        node_idx = ch.target.node
        if node_idx is None:
            continue
        node_path_sampler.setdefault(node_idx, {})[ch.target.path] = ch.sampler

    # Canonical timeline (longest sampler)
    times = _canonical_time_array(gltf, blob)
    if frame < 0 or frame >= times.shape[0]:
        raise ValueError(f"Frame {frame} out of range [0, {times.shape[0]})")
    canon_t = float(times[frame])

    # Evaluate every node's local at `canon_t`
    sampler_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    locals_T = np.empty((n_nodes, 4, 4), dtype=np.float64)
    for i in range(n_nodes):
        locals_T[i] = _node_local_at_time(
            gltf, blob, i, canon_t, node_path_sampler, sampler_cache
        )

    # FK
    globals_T = np.empty((n_nodes, 4, 4), dtype=np.float64)
    for i in order:
        p = int(parent[i])
        if p == -1:
            globals_T[i] = locals_T[i]
        else:
            globals_T[i] = globals_T[p] @ locals_T[i]

    # Joint matrices in skin order
    n_joints = len(joints)
    joint_world = np.empty((n_joints, 4, 4), dtype=np.float64)
    for jidx, jn in enumerate(joints):
        joint_world[jidx] = globals_T[jn]
    joint_M = joint_world @ inv_bind  # (J, 4, 4)

    # Find the largest skinned mesh by vertex count.
    best_mesh_idx = -1
    best_v = -1
    for mi, m in enumerate(gltf.meshes):
        prim = m.primitives[0]
        if prim.attributes.JOINTS_0 is None or prim.attributes.WEIGHTS_0 is None:
            continue
        v_acc = gltf.accessors[prim.attributes.POSITION]
        if v_acc.count > best_v:
            best_v = v_acc.count
            best_mesh_idx = mi
    if best_mesh_idx < 0:
        raise ValueError("No skinned mesh found")

    prim = gltf.meshes[best_mesh_idx].primitives[0]
    verts = _read_accessor(gltf, blob, prim.attributes.POSITION).astype(np.float64)  # (V,3)
    j_idx = _read_accessor(gltf, blob, prim.attributes.JOINTS_0).astype(np.int64)    # (V,4)
    w_val = _read_accessor(gltf, blob, prim.attributes.WEIGHTS_0).astype(np.float64) # (V,4)
    V = verts.shape[0]
    if j_idx.shape != (V, 4) or w_val.shape != (V, 4):
        raise ValueError(f"JOINTS_0/WEIGHTS_0 shape mismatch: {j_idx.shape}/{w_val.shape}")

    # Normalize weight rows (defensive — same as glb_parse)
    row_sum = w_val.sum(axis=1, keepdims=True)
    row_sum_min = float(row_sum.min())
    if row_sum_min < 1e-6:
        raise ValueError(f"Degenerate weight row (min sum {row_sum_min:.2e})")
    w_val = w_val / row_sum

    # LBS, vertex-by-vertex (independent style: explicit loop over the 4
    # influences rather than glb_parse's einsum).
    v_h = np.concatenate([verts, np.ones((V, 1), dtype=np.float64)], axis=1)  # (V,4)
    out = np.zeros((V, 3), dtype=np.float64)
    for k in range(4):
        Jk = j_idx[:, k]  # (V,)
        Wk = w_val[:, k]  # (V,)
        Mk = joint_M[Jk]  # (V, 4, 4)
        # transformed = Mk @ v_h ; do per-vertex
        # einsum is fine here, this is a different decomposition than glb_parse
        v_t = np.einsum("vij,vj->vi", Mk, v_h)[:, :3]  # (V, 3)
        out += Wk[:, None] * v_t

    return out
