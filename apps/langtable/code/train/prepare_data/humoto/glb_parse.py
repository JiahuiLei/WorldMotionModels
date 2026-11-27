"""Pure-Python GLB parser for HUMOTO dataset.

No external glb library — we read the header + JSON chunk + BIN blob directly.
Returns a GlbData dataclass with everything needed to rebuild the animated skinned
mesh: joint hierarchy, inverse-bind matrices, sub-mesh vertex/face/weights,
per-frame joint global SE(3), per-object per-frame world SE(3).
"""

import json
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Y-up (glTF) -> Z-up (UMM) conversion. Applied ONCE on the LEFT to every final
# per-frame world SE(3). NOT applied to bind verts, inv_bind, or local TRS.
# y_axis_src -> z_axis_dst, z_axis_src -> -y_axis_dst.  (Rx(+90))
# -----------------------------------------------------------------------------
R_WORLD: np.ndarray = np.eye(4, dtype=np.float64)  # (4, 4)
R_WORLD[:3, :3] = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, -1.0],
     [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class SubMesh:
    name: str
    verts: np.ndarray  # (V, 3) float64, bind pose (meters, rig-local)
    faces: np.ndarray  # (F, 3) uint32
    weights: Optional[np.ndarray]  # (V, 65) float64 row-stochastic; None for rigid objects


@dataclass
class GlbData:
    joint_names: List[str]                       # len 65
    parent_idx: np.ndarray                       # (65,) int32, -1 for joint-space root
    inv_bind: np.ndarray                         # (65, 4, 4) float64
    body_sub_meshes: List[SubMesh]               # [main_body, eyes]
    body_T_w: np.ndarray                         # (T, 65, 4, 4) float64, Z-up world
    object_meshes: Dict[str, SubMesh]            # obj_name -> SubMesh (rigid)
    object_T_w: Dict[str, np.ndarray]            # obj_name -> (T, 4, 4) float64, Z-up world
    fps: int
    T: int


# -----------------------------------------------------------------------------
# GLB binary chunk reader
# -----------------------------------------------------------------------------
_GLB_MAGIC = 0x46546C67   # 'glTF'
_CHUNK_JSON = 0x4E4F534A  # 'JSON'
_CHUNK_BIN = 0x004E4942   # 'BIN\0'


def _read_glb(path: str) -> Tuple[dict, bytes]:
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 12:
        raise ValueError(f"GLB file too short: {path}")
    magic, version, total_len = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"Bad GLB magic 0x{magic:08X} in {path}")
    if version != 2:
        raise ValueError(f"Unsupported GLB version {version} in {path}")
    if total_len != len(data):
        raise ValueError(
            f"GLB length mismatch: header says {total_len}, file is {len(data)}"
        )

    offset = 12
    json_chunk: Optional[bytes] = None
    bin_chunk: Optional[bytes] = None
    while offset < total_len:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk_data = data[offset:offset + chunk_len]
        offset += chunk_len
        if chunk_type == _CHUNK_JSON:
            json_chunk = chunk_data
        elif chunk_type == _CHUNK_BIN:
            bin_chunk = chunk_data
        else:
            # Skip unknown chunks (per spec)
            continue

    if json_chunk is None:
        raise ValueError(f"GLB missing JSON chunk: {path}")
    if bin_chunk is None:
        raise ValueError(f"GLB missing BIN chunk: {path}")
    j = json.loads(json_chunk.decode("utf-8"))
    return j, bin_chunk


# -----------------------------------------------------------------------------
# Accessor decoder
# -----------------------------------------------------------------------------
_COMPONENT_DTYPE: Dict[int, np.dtype] = {
    5120: np.dtype(np.int8),
    5121: np.dtype(np.uint8),
    5122: np.dtype(np.int16),
    5123: np.dtype(np.uint16),
    5125: np.dtype(np.uint32),
    5126: np.dtype(np.float32),
}

_TYPE_NCOMP: Dict[str, int] = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


def _read_accessor(j: dict, bin_blob: bytes, accessor_idx: int) -> np.ndarray:
    acc = j["accessors"][accessor_idx]
    bv = j["bufferViews"][acc["bufferView"]]
    comp_type = acc["componentType"]
    if comp_type not in _COMPONENT_DTYPE:
        raise ValueError(f"Unsupported componentType {comp_type}")
    dtype = _COMPONENT_DTYPE[comp_type]
    ncomp = _TYPE_NCOMP[acc["type"]]
    count = acc["count"]
    elem_size = dtype.itemsize * ncomp

    base_offset = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = bv.get("byteStride", 0) or elem_size

    if stride == elem_size:
        raw = np.frombuffer(
            bin_blob,
            dtype=dtype,
            count=count * ncomp,
            offset=base_offset,
        ).reshape(count, ncomp)
    else:
        # Strided: take bytes per element separately
        out = np.empty((count, ncomp), dtype=dtype)
        for i in range(count):
            off = base_offset + i * stride
            out[i] = np.frombuffer(
                bin_blob, dtype=dtype, count=ncomp, offset=off
            )
        raw = out

    if acc["type"] == "SCALAR":
        return raw.reshape(count)
    if acc["type"] == "MAT4":
        # glTF stores MAT4 column-major; convert to row-major (math) form
        return raw.reshape(count, 4, 4).transpose(0, 2, 1).astype(np.float64)
    if acc["type"] in ("VEC2", "VEC3", "VEC4"):
        return raw  # (count, ncomp)
    return raw


# -----------------------------------------------------------------------------
# TRS / quaternion -> matrix
# -----------------------------------------------------------------------------
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    # q = (x, y, z, w)
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = x * x + y * y + z * z + w * w
    if n < 1e-20:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [[1.0 - (yy + zz), xy - wz, xz + wy],
         [xy + wz, 1.0 - (xx + zz), yz - wx],
         [xz - wy, yz + wx, 1.0 - (xx + yy)]],
        dtype=np.float64,
    )


def _slerp_at(
    in_times: np.ndarray,  # (n,)
    quats: np.ndarray,     # (n, 4)
    t: float,
) -> np.ndarray:
    n = in_times.shape[0]
    if t <= in_times[0]:
        return quats[0].copy()
    if t >= in_times[-1]:
        return quats[-1].copy()
    idx = int(np.searchsorted(in_times, t, side="right") - 1)
    idx = max(0, min(n - 2, idx))
    t0, t1 = float(in_times[idx]), float(in_times[idx + 1])
    a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    q0 = quats[idx]
    q1 = quats[idx + 1]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + a * (q1 - q0)
        out = out / (np.linalg.norm(out) + 1e-20)
        return out
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s0 = np.sin((1.0 - a) * theta) / np.sin(theta)
    s1 = np.sin(a * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def _trs_to_matrix(
    t: np.ndarray,  # (3,)
    r: np.ndarray,  # (4,) quat (x,y,z,w)
    s: np.ndarray,  # (3,)
) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    R = _quat_to_R(r)
    M[:3, :3] = R * s[None, :]  # scale columns of R
    M[:3, 3] = t
    return M


def _node_local_static(node: dict) -> np.ndarray:
    """Compute the 'rest' local matrix of a node from its TRS or matrix field."""
    if "matrix" in node:
        m = np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # col-major -> row
        return m
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    r = np.array(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    return _trs_to_matrix(t, r, s)


# -----------------------------------------------------------------------------
# Main parser
# -----------------------------------------------------------------------------
def parse_glb(path: str) -> GlbData:  # noqa: C901 (single big function ok)
    j, bin_blob = _read_glb(path)

    nodes = j["nodes"]
    n_nodes = len(nodes)

    # Build parent[] for every node
    parent_node = np.full(n_nodes, -1, dtype=np.int32)  # (n_nodes,)
    for i, node in enumerate(nodes):
        for c_raw in node.get("children", []):
            c = int(c_raw)
            if int(parent_node[c]) != -1:
                raise ValueError(f"Node {c} has multiple parents")
            parent_node[c] = i

    # Static (rest) local matrix per node
    rest_local = np.stack(
        [_node_local_static(nd) for nd in nodes], axis=0
    )  # (n_nodes, 4, 4)

    # Skin: should be exactly 1
    if "skins" not in j or len(j["skins"]) != 1:
        raise ValueError(f"Expected exactly 1 skin in {path}, got {len(j.get('skins', []))}")
    skin = j["skins"][0]
    joint_nodes: List[int] = list(skin["joints"])  # node idx per joint
    n_joints = len(joint_nodes)
    if n_joints != 65:
        raise ValueError(f"Expected 65 joints, got {n_joints} in {path}")
    inv_bind = _read_accessor(j, bin_blob, skin["inverseBindMatrices"])  # (65,4,4)
    if inv_bind.shape != (65, 4, 4):
        raise ValueError(f"inverseBindMatrices wrong shape {inv_bind.shape}")

    joint_names = [nodes[jn].get("name", f"joint_{idx}") for idx, jn in enumerate(joint_nodes)]
    node_to_joint: Dict[int, int] = {jn: idx for idx, jn in enumerate(joint_nodes)}

    # Joint-space parent (parent in joint index, walking up the node parent chain)
    parent_idx = np.full(n_joints, -1, dtype=np.int32)
    for jidx, jn in enumerate(joint_nodes):
        p = int(parent_node[jn])
        while p != -1 and p not in node_to_joint:
            p = int(parent_node[p])
        parent_idx[jidx] = node_to_joint[p] if p != -1 else -1

    # ---------- Bind-pose unit-scale assertion ----------
    # HUMOTO glbs bake the cm->m correction as a static scale=[0.01, 0.01, 0.01]
    # on the Armature node. Find the immediate non-joint ancestor of the root
    # joint (Hips) and verify its scale. Any other value means either the
    # source shipped a different unit convention (verify_roundtrip would still
    # pass because the FK walk is self-consistent, but downstream consumers
    # assuming meters would silently get wrong units) or this file is
    # structurally unlike the validated HUMOTO format.
    root_joint_node = joint_nodes[int(np.argmax(parent_idx == -1))]
    armature_node_idx = int(parent_node[root_joint_node])
    if armature_node_idx >= 0:
        armature_scale = nodes[armature_node_idx].get("scale")
        if armature_scale is not None:
            s = np.asarray(armature_scale, dtype=np.float64)
            if not np.allclose(s, [0.01, 0.01, 0.01], atol=1e-6):
                raise ValueError(
                    f"Expected Armature node scale [0.01, 0.01, 0.01] (HUMOTO "
                    f"cm->m correction), got {s.tolist()} in {path}. The "
                    f"converter assumes meters in world coordinates after "
                    f"this scale is applied."
                )

    # ---------- Animation 0: skeleton ----------
    if "animations" not in j or len(j["animations"]) < 1:
        raise ValueError(f"No animations in {path}")
    skel_anim = j["animations"][0]

    # For each node, collect (path -> sampler_idx) for animated joints.
    # Also validate: animation[0] must only target NODES IN THE JOINT SET —
    # any non-joint node with channels here means we're mis-identifying the
    # skeleton animation vs object animations.
    node_path_sampler: Dict[int, Dict[str, int]] = {}
    non_joint_targets: List[int] = []
    for ch in skel_anim["channels"]:
        tgt = ch["target"]
        node_idx = tgt["node"]
        if node_idx not in node_to_joint:
            non_joint_targets.append(node_idx)
        node_path_sampler.setdefault(node_idx, {})[tgt["path"]] = ch["sampler"]
    if non_joint_targets:
        # Allow channels targeting the Armature node itself (some exporters
        # put root motion there); reject anything else.
        unexpected = [
            n for n in non_joint_targets if n != armature_node_idx
        ]
        if unexpected:
            raise ValueError(
                f"animations[0] in {path} targets non-joint nodes {unexpected} "
                f"(expected only joint nodes or the Armature parent). This "
                f"glb does not match the HUMOTO convention where anim[0] is "
                f"purely the skeleton."
            )

    # Determine the dominant frame count (the longest sampler in skeleton anim).
    # HUMOTO glbs typically mix per-frame (T=181) samplers and 2-sample constant
    # samplers (joints that don't move). Use T = max, interpolate shorter ones.
    sampler_counts: List[int] = []
    for samp in skel_anim["samplers"]:
        if samp.get("interpolation", "LINEAR") not in ("LINEAR", "STEP"):
            raise ValueError(
                f"Unsupported sampler interpolation {samp.get('interpolation')} in {path}"
            )
        sampler_counts.append(j["accessors"][samp["input"]]["count"])
    if not sampler_counts:
        raise ValueError(f"Animation 0 has no samplers in {path}")
    T_count = int(max(sampler_counts))
    if T_count < 2:
        raise ValueError(f"Need >=2 frames, got {T_count}")

    # Find one sampler that has T_count samples and use its time array as the
    # canonical frame timing.
    canonical_input_acc: Optional[int] = None
    for samp in skel_anim["samplers"]:
        in_acc = samp["input"]
        if j["accessors"][in_acc]["count"] == T_count:
            canonical_input_acc = in_acc
            break
    if canonical_input_acc is None:
        raise ValueError(f"No sampler has count {T_count} in {path}")
    times = _read_accessor(j, bin_blob, canonical_input_acc).astype(np.float64)  # (T,)
    if times.shape != (T_count,):
        raise ValueError(f"Times shape {times.shape} != ({T_count},)")
    dt = (times[-1] - times[0]) / (T_count - 1)
    fps_f = 1.0 / float(dt)
    if not (29.5 < fps_f < 30.5):
        raise ValueError(f"Expected 30 fps, got {fps_f:.4f} in {path}")
    fps = 30

    # Decode per-sampler arrays once. Shorter samplers (e.g. 2-sample
    # constants) are resampled to the canonical (T,) time grid via linear
    # interpolation (slerp for quats).
    sampler_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _resample(
        anim: dict, sidx: int, kind: str, dim: int
    ) -> np.ndarray:
        key = (sidx, kind)
        if key in sampler_cache:
            return sampler_cache[key]
        samp = anim["samplers"][sidx]
        in_times = _read_accessor(j, bin_blob, samp["input"]).astype(np.float64)  # (n,)
        out_vals = _read_accessor(j, bin_blob, samp["output"]).astype(np.float64)  # (n, dim)
        if out_vals.ndim == 1:
            out_vals = out_vals.reshape(-1, dim)
        if out_vals.shape[1] != dim:
            raise ValueError(
                f"Sampler {sidx} {kind}: dim {out_vals.shape[1]} != expected {dim}"
            )
        n_in = in_times.shape[0]
        if n_in == T_count and np.array_equal(in_times, times):
            arr = out_vals
        else:
            interp_kind = samp.get("interpolation", "LINEAR")
            arr = np.empty((T_count, dim), dtype=np.float64)
            if interp_kind == "STEP":
                # nearest-floor lookup — applies to TRS and rotation alike
                for t_i in range(T_count):
                    idx = int(np.searchsorted(in_times, times[t_i], side="right") - 1)
                    idx = max(0, min(n_in - 1, idx))
                    arr[t_i] = out_vals[idx]
            elif kind == "rotation":
                for t_i in range(T_count):
                    arr[t_i] = _slerp_at(in_times, out_vals, times[t_i])
            else:
                for d in range(dim):
                    arr[:, d] = np.interp(times, in_times, out_vals[:, d])
        sampler_cache[key] = arr
        return arr

    def _get_translations(sidx: int) -> np.ndarray:
        return _resample(skel_anim, sidx, "translation", 3)

    def _get_rotations(sidx: int) -> np.ndarray:
        return _resample(skel_anim, sidx, "rotation", 4)

    def _get_scales(sidx: int) -> np.ndarray:
        return _resample(skel_anim, sidx, "scale", 3)

    # Pre-extract rest TRS for each node (used as fallback when a path is not animated)
    rest_t = np.zeros((n_nodes, 3), dtype=np.float64)
    rest_r = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_nodes, 1))
    rest_s = np.ones((n_nodes, 3), dtype=np.float64)
    for i, nd in enumerate(nodes):
        if "matrix" in nd:
            # matrix-only nodes: we'll fall back to rest_local directly when not animated
            continue
        rest_t[i] = nd.get("translation", [0.0, 0.0, 0.0])
        rest_r[i] = nd.get("rotation", [0.0, 0.0, 0.0, 1.0])
        rest_s[i] = nd.get("scale", [1.0, 1.0, 1.0])

    # Per-frame node-local matrices for animated nodes only; static nodes use rest_local
    animated_nodes = sorted(node_path_sampler.keys())
    # Storage: (T, n_nodes, 4, 4) is too big? n_nodes ~70, T<2k → 70*2000*16*8 = ~18 MB ok
    # We'll only fill animated nodes; static nodes broadcast rest_local.
    locals_T = np.broadcast_to(rest_local, (T_count, n_nodes, 4, 4)).copy()

    for node_idx in animated_nodes:
        paths = node_path_sampler[node_idx]
        # T arrays of shape (3,), (4,), (3,)
        if "translation" in paths:
            t_arr = _get_translations(paths["translation"])
        else:
            t_arr = np.broadcast_to(rest_t[node_idx], (T_count, 3))
        if "rotation" in paths:
            r_arr = _get_rotations(paths["rotation"])
        else:
            r_arr = np.broadcast_to(rest_r[node_idx], (T_count, 4))
        if "scale" in paths:
            s_arr = _get_scales(paths["scale"])
        else:
            s_arr = np.broadcast_to(rest_s[node_idx], (T_count, 3))

        for t in range(T_count):
            locals_T[t, node_idx] = _trs_to_matrix(t_arr[t], r_arr[t], s_arr[t])

    # FK across whole scene per frame, in topological order (parents before children)
    topo_order = _topo_sort(parent_node)  # list of node indices, parents first

    globals_T = np.empty((T_count, n_nodes, 4, 4), dtype=np.float64)
    for t in range(T_count):
        for i in topo_order:
            p = parent_node[i]
            if p == -1:
                globals_T[t, i] = locals_T[t, i]
            else:
                globals_T[t, i] = globals_T[t, p] @ locals_T[t, i]

    # Bind-pose sanity check: at REST (using rest_local globals), bind_global @ inv_bind == I
    rest_globals = np.empty((n_nodes, 4, 4), dtype=np.float64)
    for i in topo_order:
        p = parent_node[i]
        if p == -1:
            rest_globals[i] = rest_local[i]
        else:
            rest_globals[i] = rest_globals[p] @ rest_local[i]

    eye4 = np.eye(4, dtype=np.float64)
    for jidx, jn in enumerate(joint_nodes):
        check = rest_globals[jn] @ inv_bind[jidx]
        err = float(np.abs(check - eye4).max())
        if err > 1e-5:
            raise ValueError(
                f"Bind sanity failed for joint '{joint_names[jidx]}' "
                f"(node {jn}): max abs err {err:.3e}"
            )

    # Final body_T_w: pick out joint nodes, apply R_WORLD on the left
    body_T_w = np.empty((T_count, n_joints, 4, 4), dtype=np.float64)
    for jidx, jn in enumerate(joint_nodes):
        body_T_w[:, jidx] = R_WORLD @ globals_T[:, jn]

    # ---------- Sub-meshes (body) ----------
    if "meshes" not in j or len(j["meshes"]) < 2:
        raise ValueError(f"Expected >=2 meshes in {path}")

    body_sub_meshes: List[SubMesh] = []
    for mi in (0, 1):
        m = j["meshes"][mi]
        prim = m["primitives"][0]
        attr = prim["attributes"]
        verts = _read_accessor(j, bin_blob, attr["POSITION"]).astype(np.float64)
        joints_v = _read_accessor(j, bin_blob, attr["JOINTS_0"]).astype(np.int64)
        weights_v = _read_accessor(j, bin_blob, attr["WEIGHTS_0"]).astype(np.float64)
        idx_v = _read_accessor(j, bin_blob, prim["indices"]).astype(np.uint32)
        faces = idx_v.reshape(-1, 3)

        V = verts.shape[0]
        if joints_v.shape != (V, 4) or weights_v.shape != (V, 4):
            raise ValueError(
                f"mesh {mi}: joints/weights shape mismatch "
                f"{joints_v.shape}/{weights_v.shape}"
            )

        # Build dense (V, 65) weights
        wd = np.zeros((V, n_joints), dtype=np.float64)
        for k in range(4):
            wd[np.arange(V), joints_v[:, k]] += weights_v[:, k]
        # Normalize rows (some glTFs have rows that don't quite sum to 1)
        row_sums = wd.sum(axis=1)
        max_dev = float(np.max(np.abs(row_sums - 1.0)))
        if max_dev > 1e-3:
            raise ValueError(
                f"mesh {mi}: skin weight row sum dev {max_dev:.3e} > 1e-3"
            )
        wd = wd / row_sums[:, None]

        body_sub_meshes.append(
            SubMesh(
                name=m.get("name", f"mesh_{mi}"),
                verts=verts,
                faces=faces,
                weights=wd,
            )
        )

    # Order: main_body first, eyes second (sort by vertex count desc).
    # Assert body >> eyes so the two cannot be confused (observed ratio in HUMOTO: ~13.8k vs 1.2k).
    body_sub_meshes.sort(key=lambda sm: -sm.verts.shape[0])
    if len(body_sub_meshes) < 2:
        raise ValueError(f"Expected 2 body sub-meshes in {path}, got {len(body_sub_meshes)}")
    v0, v1 = body_sub_meshes[0].verts.shape[0], body_sub_meshes[1].verts.shape[0]
    if v0 < 4 * v1:
        raise ValueError(
            f"Body sub-mesh ordering ambiguous in {path}: main={v0} verts, eyes={v1} verts "
            f"(expected main >> eyes, ratio >=4)"
        )

    # ---------- Object animations + meshes ----------
    object_T_w: Dict[str, np.ndarray] = {}
    object_meshes: Dict[str, SubMesh] = {}

    for ai in range(1, len(j["animations"])):
        oa = j["animations"][ai]
        # All channels should target a single object node
        target_nodes = {ch["target"]["node"] for ch in oa["channels"]}
        if len(target_nodes) != 1:
            raise ValueError(
                f"Object anim {ai} targets multiple nodes {target_nodes} in {path}"
            )
        tgt_node = next(iter(target_nodes))
        node = nodes[tgt_node]
        obj_name = node.get("name", f"obj_{tgt_node}")
        # Strip any rig prefix; keep as-is otherwise
        obj_key = obj_name.split(":")[-1] if ":" in obj_name else obj_name

        # Validate interpolation mode
        for samp in oa["samplers"]:
            if samp.get("interpolation", "LINEAR") not in ("LINEAR", "STEP"):
                raise ValueError(
                    f"Unsupported interpolation in object anim {ai}: "
                    f"{samp.get('interpolation')}"
                )

        # Collect path -> sampler idx for this anim
        path_to_samp: Dict[str, int] = {}
        for ch in oa["channels"]:
            path_to_samp[ch["target"]["path"]] = ch["sampler"]

        def _resample_obj(sidx: int, kind: str, dim: int) -> np.ndarray:
            samp = oa["samplers"][sidx]
            in_times = _read_accessor(j, bin_blob, samp["input"]).astype(np.float64)
            out_vals = _read_accessor(j, bin_blob, samp["output"]).astype(np.float64)
            if out_vals.ndim == 1:
                out_vals = out_vals.reshape(-1, dim)
            if out_vals.shape[1] != dim:
                raise ValueError(
                    f"Object sampler {sidx} {kind} dim {out_vals.shape[1]} != {dim}"
                )
            n_in = in_times.shape[0]
            if n_in == T_count and np.array_equal(in_times, times):
                return out_vals
            interp_kind = samp.get("interpolation", "LINEAR")
            arr = np.empty((T_count, dim), dtype=np.float64)
            if interp_kind == "STEP":
                for t_i in range(T_count):
                    idx = int(np.searchsorted(in_times, times[t_i], side="right") - 1)
                    idx = max(0, min(n_in - 1, idx))
                    arr[t_i] = out_vals[idx]
            elif kind == "rotation":
                for t_i in range(T_count):
                    arr[t_i] = _slerp_at(in_times, out_vals, times[t_i])
            else:
                for d in range(dim):
                    arr[:, d] = np.interp(times, in_times, out_vals[:, d])
            return arr

        if "translation" in path_to_samp:
            obj_t = _resample_obj(path_to_samp["translation"], "translation", 3)
        else:
            obj_t = np.broadcast_to(rest_t[tgt_node], (T_count, 3))
        if "rotation" in path_to_samp:
            obj_r = _resample_obj(path_to_samp["rotation"], "rotation", 4)
        else:
            obj_r = np.broadcast_to(rest_r[tgt_node], (T_count, 4))
        if "scale" in path_to_samp:
            obj_s = _resample_obj(path_to_samp["scale"], "scale", 3)
        else:
            obj_s = np.broadcast_to(rest_s[tgt_node], (T_count, 3))

        # Build per-frame local matrix of the object node
        obj_local_T = np.empty((T_count, 4, 4), dtype=np.float64)
        for t in range(T_count):
            obj_local_T[t] = _trs_to_matrix(obj_t[t], obj_r[t], obj_s[t])

        # Walk ancestors of tgt_node (rest globals) to compose true world.
        # NOTE: object nodes in HUMOTO are top-level siblings of Armature (no parent),
        # but we handle the general case for safety.
        ancestor_chain: List[int] = []
        p = int(parent_node[tgt_node])
        while p != -1:
            ancestor_chain.append(p)
            p = int(parent_node[p])
        ancestor_chain.reverse()  # root .. immediate parent

        if ancestor_chain:
            anc_global = np.eye(4, dtype=np.float64)
            for a in ancestor_chain:
                anc_global = anc_global @ rest_local[a]
        else:
            anc_global = np.eye(4, dtype=np.float64)

        obj_world_T = np.empty((T_count, 4, 4), dtype=np.float64)
        for t in range(T_count):
            obj_world_T[t] = R_WORLD @ (anc_global @ obj_local_T[t])
        object_T_w[obj_key] = obj_world_T

        # Object mesh (rigid)
        if "mesh" in node:
            mesh_idx = node["mesh"]
            m_obj = j["meshes"][mesh_idx]
            prim = m_obj["primitives"][0]
            attr = prim["attributes"]
            verts = _read_accessor(j, bin_blob, attr["POSITION"]).astype(np.float64)
            idx_v = _read_accessor(j, bin_blob, prim["indices"]).astype(np.uint32)
            faces = idx_v.reshape(-1, 3)
            object_meshes[obj_key] = SubMesh(
                name=obj_key,
                verts=verts,
                faces=faces,
                weights=None,
            )

    return GlbData(
        joint_names=joint_names,
        parent_idx=parent_idx,
        inv_bind=inv_bind,
        body_sub_meshes=body_sub_meshes,
        body_T_w=body_T_w,
        object_meshes=object_meshes,
        object_T_w=object_T_w,
        fps=fps,
        T=T_count,
    )


# -----------------------------------------------------------------------------
# Topological sort over node parent array
# -----------------------------------------------------------------------------
def _topo_sort(parent: np.ndarray) -> List[int]:
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
        # extend in reverse so children come out in original order (irrelevant for FK)
        stack.extend(children[i])
    if len(order) != n:
        raise ValueError(f"Topo sort produced {len(order)} of {n} nodes (cycle?)")
    return order
