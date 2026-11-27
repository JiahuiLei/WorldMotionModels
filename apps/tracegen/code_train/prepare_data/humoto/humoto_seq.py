"""HumotoSeq — loader + LBS skinner + round-trip verifier for one processed sequence.

Storage layout (under processed_dir/):
    sequences/<seq_name>.npz     # body_T_w, obj_<name>_T_w..., object_names, fps, T,
                                 # rig_hash, schema_version
    sequences/<seq_name>.yaml    # text_short, text_segments, scene, declared_objects,
                                 # start/end, fps, T, rig_hash
    objects/<name>/mesh.obj      # glb-internal mesh, deduped by sha256 across seqs
    template/<rig_hash>/template.npz   # per-rig template (13 variants in HUMOTO)

Schema version is bumped whenever the on-disk format changes incompatibly.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import yaml

from .glb_parse import R_WORLD, parse_glb

# Bump whenever the on-disk format changes incompatibly. Loader refuses to
# read mismatched versions.
HUMOTO_SCHEMA_VERSION = 3


@dataclass
class HumotoSeq:
    seq_name: str
    body_T_w: np.ndarray              # (T, 65, 4, 4) float64, Z-up world
    objects: Dict[str, np.ndarray]    # name -> (T, 4, 4)
    text_short: str
    text_segments: List[Tuple[int, int, str]]
    fps: int
    T: int
    scene: str
    declared_objects: List[str]
    start_frame: int
    end_frame: int
    rig_hash: str                     # key into template/<rig_hash>/

    _processed_dir: Path = field(default_factory=lambda: Path("."))
    _template: Optional[dict] = None
    _object_mesh_cache: Dict[str, trimesh.Trimesh] = field(default_factory=dict)
    # True only when constructed via load() from a real on-disk processed dir.
    # False when built via from_glb(). verify_roundtrip gates on this so a
    # caller cannot accidentally roundtrip-verify an in-memory seq (which would
    # bypass the npz save/load path that we actually want to stress-test).
    _loaded_from_disk: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, processed_dir: str, seq_name: str) -> "HumotoSeq":
        """Load a sequence from disk. Reads sequences/<seq>.npz +
        sequences/<seq>.yaml. Template is loaded lazily on first skin call.
        """
        pd = Path(processed_dir)
        npz_path = pd / "sequences" / f"{seq_name}.npz"
        meta_path = pd / "sequences" / f"{seq_name}.yaml"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")

        data = np.load(npz_path, allow_pickle=True)
        # Schema version check — fail loud on mismatch
        if "schema_version" not in data.files:
            raise KeyError(
                f"Missing 'schema_version' field in {npz_path}. This processed "
                f"dir was built by a pre-v{HUMOTO_SCHEMA_VERSION} converter — "
                f"regenerate via `python -m prepare_data.humoto.convert ...`."
            )
        found_v = int(data["schema_version"])
        if found_v != HUMOTO_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch in {npz_path}: file={found_v} "
                f"loader={HUMOTO_SCHEMA_VERSION}. Regenerate the processed "
                f"dir with the current converter, then run the object refinement "
                f"step to add per-object pcl.npz + mesh_viz.obj:\n"
                f"  python -m prepare_data.humoto.convert --root <src> --out <processed_dir>\n"
                f"  python -m prepare_data.humoto.refine_objects --processed_dir <processed_dir>"
            )
        body_T_w = data["body_T_w"].astype(np.float64)  # (T, 65, 4, 4)
        T = int(body_T_w.shape[0])
        fps = int(data["fps"])
        if "rig_hash" not in data.files:
            raise KeyError(
                f"Missing 'rig_hash' field in {npz_path}. Regenerate."
            )
        rig_h = data["rig_hash"].item() if hasattr(data["rig_hash"], "item") else str(data["rig_hash"])

        objects: Dict[str, np.ndarray] = {}
        if "object_names" in data.files:
            for name in data["object_names"]:
                key = f"obj_{name}_T_w"
                if key not in data.files:
                    raise KeyError(f"Object array {key} missing in {npz_path}")
                objects[str(name)] = data[key].astype(np.float64)

        with open(meta_path, "r") as f:
            meta = yaml.safe_load(f)

        text_segments: List[Tuple[int, int, str]] = [
            (int(s["start_frame"]), int(s["end_frame"]), str(s["script"]))
            for s in meta.get("text_segments", []) or []
        ]

        return cls(
            seq_name=seq_name,
            body_T_w=body_T_w,
            objects=objects,
            text_short=str(meta.get("text_short", "")),
            text_segments=text_segments,
            fps=fps,
            T=T,
            scene=str(meta.get("scene", "")),
            declared_objects=list(meta.get("declared_objects", []) or []),
            start_frame=int(meta.get("start_frame", 1)),
            end_frame=int(meta.get("end_frame", T)),
            rig_hash=rig_h,
            _processed_dir=pd,
            _loaded_from_disk=True,
        )

    @classmethod
    def from_glb(
        cls,
        glb_path: str,
        seq_name: str,
        yaml_path: Optional[str] = None,
    ) -> "HumotoSeq":
        """Build a HumotoSeq directly from a glb (and optional yaml).

        Used by the converter and by smoke/round-trip tests where no processed_dir
        exists yet. The template is filled in-memory from the glb.
        """
        glb = parse_glb(glb_path)

        text_short = ""
        text_segments: List[Tuple[int, int, str]] = []
        scene = ""
        declared_objects: List[str] = []
        start_frame = 1
        end_frame = glb.T
        if yaml_path is not None:
            with open(yaml_path, "r") as f:
                meta = yaml.safe_load(f)
            text_short = str(meta.get("short_script", ""))
            for seg in meta.get("long_script", []):
                text_segments.append(
                    (int(seg["start_frame"]), int(seg["end_frame"]), str(seg["script"]))
                )
            scene = str(meta.get("scene", ""))
            declared_objects = list(meta.get("objects", []))
            start_frame = int(meta.get("start_frame", 1))
            end_frame = int(meta.get("end_frame", glb.T))

        seq = cls(
            seq_name=seq_name,
            body_T_w=glb.body_T_w,
            objects=dict(glb.object_T_w),
            text_short=text_short,
            text_segments=text_segments,
            fps=glb.fps,
            T=glb.T,
            scene=scene,
            declared_objects=declared_objects,
            start_frame=start_frame,
            end_frame=end_frame,
            rig_hash="",  # from_glb doesn't know its on-disk rig hash
            _processed_dir=Path("."),
        )

        # Pre-fill in-memory template directly from glb
        main_sm = glb.body_sub_meshes[0]
        eyes_sm = glb.body_sub_meshes[1]
        seq._template = {
            "joint_names": glb.joint_names,
            "parent_idx": glb.parent_idx,
            "inv_bind": glb.inv_bind,                          # (65,4,4)
            "body_main_verts": main_sm.verts,                  # (V,3)
            "body_main_faces": main_sm.faces,                  # (F,3)
            "body_main_weights": main_sm.weights,              # (V,65)
            "body_eyes_verts": eyes_sm.verts,
            "body_eyes_faces": eyes_sm.faces,
            "body_eyes_weights": eyes_sm.weights,
        }
        # Cache object meshes from glb (for transform_object without disk)
        for name, sm in glb.object_meshes.items():
            seq._object_mesh_cache[name] = trimesh.Trimesh(
                vertices=sm.verts.copy(), faces=sm.faces.copy(), process=False
            )
        return seq

    # ------------------------------------------------------------------
    # Template
    # ------------------------------------------------------------------
    def _load_template(self) -> dict:
        if self._template is not None:
            return self._template
        if not self.rig_hash:
            raise ValueError(
                f"HumotoSeq {self.seq_name!r} has empty rig_hash; cannot "
                f"resolve per-rig template. This seq was likely built via "
                f"from_glb and no template is available on disk."
            )
        tpath = self._processed_dir / "template" / self.rig_hash / "template.npz"
        if not tpath.exists():
            raise FileNotFoundError(
                f"Missing template at {tpath}. Either the processed dir was "
                f"built by an older converter (no per-rig templates) or the "
                f"rig_hash for this seq is inconsistent with what was written."
            )
        d = np.load(tpath, allow_pickle=True)
        self._template = {
            "joint_names": list(d["joint_names"]),
            "parent_idx": d["parent_idx"].astype(np.int32),
            "inv_bind": d["inv_bind"].astype(np.float64),
            "body_main_verts": d["body_main_verts"].astype(np.float64),
            "body_main_faces": d["body_main_faces"].astype(np.uint32),
            "body_main_weights": d["body_main_weights"].astype(np.float64),
            "body_eyes_verts": d["body_eyes_verts"].astype(np.float64),
            "body_eyes_faces": d["body_eyes_faces"].astype(np.uint32),
            "body_eyes_weights": d["body_eyes_weights"].astype(np.float64),
        }
        return self._template

    # ------------------------------------------------------------------
    # LBS skinning
    # ------------------------------------------------------------------
    def _skin_one(
        self,
        verts: np.ndarray,    # (V, 3)
        faces: np.ndarray,    # (F, 3)
        weights: np.ndarray,  # (V, 65)
        t: int,
    ) -> trimesh.Trimesh:
        tpl = self._load_template()
        inv_bind = tpl["inv_bind"]                          # (65, 4, 4)
        T_pose_w = self.body_T_w[t]                         # (65, 4, 4) world (Z-up)
        # Per-joint world skinning matrices: M_j = T_pose_w[j] @ inv_bind[j]
        M = T_pose_w @ inv_bind                              # (65, 4, 4)

        V = verts.shape[0]
        v_h = np.concatenate([verts, np.ones((V, 1), dtype=np.float64)], axis=1)  # (V, 4)
        # Transform every vert by every joint: (V, 65, 4)
        # einsum: M[j, r, c] @ v_h[v, c] -> (V, j, r)
        v_per_joint = np.einsum("jrc,vc->vjr", M, v_h)       # (V, 65, 4)
        # Weighted sum across joints
        v_world_h = np.einsum("vj,vjr->vr", weights, v_per_joint)  # (V, 4)
        v_world = v_world_h[:, :3]                           # (V, 3)

        return trimesh.Trimesh(
            vertices=v_world, faces=faces, process=False
        )

    def skin_body(self, t: int) -> Dict[str, trimesh.Trimesh]:
        tpl = self._load_template()
        body = self._skin_one(
            tpl["body_main_verts"],
            tpl["body_main_faces"],
            tpl["body_main_weights"],
            t,
        )
        eyes = self._skin_one(
            tpl["body_eyes_verts"],
            tpl["body_eyes_faces"],
            tpl["body_eyes_weights"],
            t,
        )
        return {"body": body, "eyes": eyes}

    # ------------------------------------------------------------------
    # Object handling
    # ------------------------------------------------------------------
    def _load_object_mesh(self, name: str) -> trimesh.Trimesh:
        """Load the viewer-friendly mesh for object ``name``.

        Loads ``mesh_viz.obj`` (decimated to ~5k faces for heavy objects such as
        ``bed`` or ``kitchen_counter`` that would crash the viewer at full
        resolution). If ``mesh_viz.obj`` does not exist — e.g. the processed dir
        was built before the refinement step — a ``FileNotFoundError`` is raised
        with a hint to run ``refine_objects``.

        Run the refinement step to generate ``mesh_viz.obj``::

            python -m prepare_data.humoto.refine_objects --processed_dir <processed_dir>
        """
        if name in self._object_mesh_cache:
            return self._object_mesh_cache[name]
        path = self._processed_dir / "objects" / name / "mesh_viz.obj"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing viewer mesh {path}. "
                f"Generate it by running:\n"
                f"  python -m prepare_data.humoto.refine_objects "
                f"--processed_dir {self._processed_dir}"
            )
        mesh = trimesh.load(path, process=False, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object {name} is not a Trimesh")
        self._object_mesh_cache[name] = mesh
        return mesh

    def transform_object(self, name: str, t: int) -> trimesh.Trimesh:
        if name not in self.objects:
            raise KeyError(f"Object '{name}' not in sequence (have: {list(self.objects)})")
        mesh = self._load_object_mesh(name)
        T_w = self.objects[name][t]                          # (4, 4)
        v = mesh.vertices                                    # (V, 3)
        v_h = np.concatenate([v, np.ones((v.shape[0], 1), dtype=np.float64)], axis=1)
        v_w = (T_w @ v_h.T).T[:, :3]
        return trimesh.Trimesh(vertices=v_w, faces=mesh.faces, process=False)

    def scene_mesh(self, t: int) -> trimesh.Scene:
        scene = trimesh.Scene()
        body_dict = self.skin_body(t)
        scene.add_geometry(body_dict["body"], geom_name="body")
        scene.add_geometry(body_dict["eyes"], geom_name="eyes")
        for name in self.objects:
            # fail-loud: no silent skip (project rule: no silent fallbacks).
            scene.add_geometry(self.transform_object(name, t), geom_name=f"obj_{name}")
        return scene

    # ------------------------------------------------------------------
    # Raw-frame LBS helper (used by verify_roundtrip)
    # ------------------------------------------------------------------
    def _skin_main_body_raw(self, t: int) -> np.ndarray:
        """LBS the main body sub-mesh at frame `t` in the RAW glTF coordinate
        frame (Y-up). Returns (V, 3) float64.

        This undoes the R_WORLD rotation that glb_parse applied to body_T_w so
        the result is directly comparable to gt_loader.skin_body_at(...).
        """
        tpl = self._load_template()
        inv_bind = tpl["inv_bind"]                                # (65, 4, 4)
        verts = tpl["body_main_verts"]                            # (V, 3)
        weights = tpl["body_main_weights"]                        # (V, 65)

        R_inv4 = np.eye(4, dtype=np.float64)
        R_inv4[:3, :3] = R_WORLD[:3, :3].T  # inverse of Rx(+90)

        body_T_raw = R_inv4 @ self.body_T_w[t]                    # (65, 4, 4)
        M = body_T_raw @ inv_bind                                  # (65, 4, 4)

        V = verts.shape[0]
        v_h = np.concatenate(
            [verts, np.ones((V, 1), dtype=np.float64)], axis=1
        )                                                          # (V, 4)
        v_per_joint = np.einsum("jrc,vc->vjr", M, v_h)             # (V, 65, 4)
        v_h_out = np.einsum("vj,vjr->vr", weights, v_per_joint)    # (V, 4)
        return v_h_out[:, :3]

    # ------------------------------------------------------------------
    # Strict round-trip verification (real save -> load -> skin -> compare)
    # ------------------------------------------------------------------
    def verify_roundtrip(
        self,
        glb_path: str,
        tol_max_m: float = 1e-3,
        tol_med_m: float = 1e-4,
        dense: bool = False,
        n_frames: int = 5,
    ) -> dict:
        """Strict round-trip verification using an INDEPENDENT GT loader.

        This method tests the FULL save -> load -> skin chain. MUST be called
        on a HumotoSeq that was loaded FROM DISK (via `HumotoSeq.load`), not
        on one built in-memory via `from_glb` — only the on-disk path
        exercises the npz I/O that we want to validate.

        Pipeline:
            1. For each checked frame:
            2. BASELINE: `gt_loader.skin_body_at(glb_path, frame)` (raw Y-up).
            3. OURS: `_skin_main_body_raw(frame)` (undoes R_WORLD).
            4. COMPARE: per-vertex L2.
            GATE: max < tol_max_m AND median < tol_med_m for every frame.

        Frame sampling:
            dense=False (default): `n_frames` evenly-spaced samples across [0, T-1].
                The default 5 is adequate for time-independent bugs (LBS,
                inv_bind, orientation) but misses interpolation-localized bugs.
            dense=True: every frame. Use for the first few seqs of a bulk run
                to stress-test the animation resampler.
        """
        # Local import keeps gt_loader's pygltflib dep optional for normal use.
        from . import gt_loader  # type: ignore[attr-defined]

        # Runtime guard: MUST be on a disk-loaded seq, otherwise the test
        # silently degrades (the "_skin_main_body_raw" call would use the
        # in-memory template from `from_glb`, bypassing the npz save/load path).
        if not self._loaded_from_disk:
            raise RuntimeError(
                "verify_roundtrip requires a HumotoSeq loaded from disk via "
                "HumotoSeq.load(processed_dir, seq_name). The passed seq was "
                "built via from_glb which bypasses the npz I/O path we must "
                "validate. Write the seq to disk first, then reload."
            )

        tpl = self._load_template()
        v_template = tpl["body_main_verts"]
        T = self.T
        if T < 1:
            raise ValueError(f"Empty sequence T={T}")

        # Frame sampling: dense=every frame, else n_frames evenly spaced
        if dense:
            frames: List[int] = list(range(T))
        else:
            if n_frames < 2:
                raise ValueError(f"n_frames must be >= 2, got {n_frames}")
            if n_frames >= T:
                frames = list(range(T))
            else:
                frames = sorted(
                    {int(round(i * (T - 1) / (n_frames - 1))) for i in range(n_frames)}
                )

        per_frame: List[dict] = []
        gt_box_min = np.full(3, np.inf)
        gt_box_max = np.full(3, -np.inf)
        ours_box_min = np.full(3, np.inf)
        ours_box_max = np.full(3, -np.inf)
        all_pass = True

        for f in frames:
            v_gt = gt_loader.skin_body_at(glb_path, f)            # (V, 3) raw
            if v_gt.shape != v_template.shape:
                raise ValueError(
                    f"GT loader vertex count {v_gt.shape} mismatch with "
                    f"template {v_template.shape} — structural bug"
                )
            v_ours = self._skin_main_body_raw(f)                  # (V, 3) raw
            if v_ours.shape != v_gt.shape:
                raise ValueError(
                    f"_skin_main_body_raw shape {v_ours.shape} != GT {v_gt.shape}"
                )

            d = np.linalg.norm(v_ours - v_gt, axis=1)
            f_max = float(d.max())
            f_med = float(np.median(d))
            f_pass = (f_max < tol_max_m) and (f_med < tol_med_m)
            per_frame.append({
                "frame": int(f),
                "max_m": f_max,
                "median_m": f_med,
                "pass": bool(f_pass),
            })
            if not f_pass:
                all_pass = False

            gt_box_min = np.minimum(gt_box_min, v_gt.min(0))
            gt_box_max = np.maximum(gt_box_max, v_gt.max(0))
            ours_box_min = np.minimum(ours_box_min, v_ours.min(0))
            ours_box_max = np.maximum(ours_box_max, v_ours.max(0))

        return {
            "pass": bool(all_pass),
            "n_frames": T,
            "frames_checked": frames,
            "tol_max_m": tol_max_m,
            "tol_med_m": tol_med_m,
            "per_frame": per_frame,
            "per_frame_summary": ", ".join(
                f"f={r['frame']}:max={r['max_m']:.2e}/med={r['median_m']:.2e}"
                for r in per_frame
            ),
            "gt_bbox": (gt_box_min.tolist(), gt_box_max.tolist()),
            "ours_bbox": (ours_box_min.tolist(), ours_box_max.tolist()),
        }
