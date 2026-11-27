"""Interactive viser visualizer + headless renderer for the OMOMO
CHOIS-protocol processed dataset.

Reads SOLELY from ``{processed_dir}/{split}/*.npz`` windows produced by
``prepare_data/omomo/convert.py``. For a given window, runs SMPL-X forward
kinematics on the stored ``root_trans + root_orient_aa + pose_body_aa + betas``
(gender-correct model) to reconstruct the 10475-vert body mesh per frame and
places the object mesh using the stored ``(R, t)``.

World frame: Z-up, meters, 30 FPS.

CLI (interactive)::

    python -m prepare_data.omomo.omomo_viser \\
        --processed_dir data/data/omomo/umm \\
        --split test \\
        --port 8190 \\
        --fps 30 \\
        [--seq sub16_clothesstand_000_w0]

CLI (headless render)::

    python -m prepare_data.omomo.omomo_viser \\
        --processed_dir data/data/omomo/umm \\
        --split test \\
        --seq sub16_clothesstand_000_w0 \\
        render \\
        --frames 0 60 119 \\
        --out_dir /tmp/omomo_jpgs/

CLI arg ordering:
    ALL common args (``--processed_dir``, ``--split``, ``--raw_root``,
    ``--smplx_dir``, ``--mesh_source``, ``--seq``, ``--port``, ``--host``,
    ``--fps``) live on the TOP-LEVEL parser and must appear BEFORE the
    optional ``render`` subcommand. The ``render`` subcommand only adds
    its own render-specific args (``--frames``, ``--out_dir``, ``--width``,
    ``--height``)::

        # interactive viewer (no subcommand)
        python -m prepare_data.omomo.omomo_viser [common-args]

        # headless render: common BEFORE render, render-specific AFTER
        python -m prepare_data.omomo.omomo_viser [common-args] render [render-args]

    Passing a common arg AFTER ``render`` (e.g. ``render --processed_dir ...``)
    will now raise ``unrecognized arguments`` — failing loudly rather than
    silently overwriting the top-level value with a subparser default.

Module-independence note: this file does NOT import from sibling
``prepare_data.himo.*`` or ``prepare_data.humoto.*``. All helper logic
(ground plane, lighting, camera, offscreen rendering via pyrender) is
self-contained here.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# CRITICAL: must be set before importing pyrender so it picks the EGL
# backend — renders without a running X server / display. Any user-
# provided value in the environment is respected; only set a default.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender  # type: ignore
import torch
import trimesh
import viser
from PIL import Image

try:
    import smplx  # type: ignore
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "smplx package required for OMOMO viser viewer. "
        "`pip install smplx` or `conda activate umm`."
    ) from _e

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

# SMPL-X model files under raw OMOMO download.
DEFAULT_SMPLX_DIR = Path("data/data/omomo/raw/smpl_all_models/smplx")
# Default raw OMOMO root (for resolving obj_mesh_path relative paths).
DEFAULT_RAW_ROOT = Path("data/data/omomo/raw")
# Candidate directories inside raw root where RAW captured OBJ meshes live.
# (Only used for --mesh_source=raw debug/diagnostic mode; production uses
# rest_object_geo/*.ply because captured_objects/*.obj is in the raw-scan
# frame — huge extents (>=4m), off-centre, wrong orientation — while the
# (obj_trans, obj_rot) stored in the npz are for the CANONICAL rest-frame
# mesh at rest_object_geo/<name>.ply (centered, upright, correct scale).)
OBJ_MESH_RAW_SEARCH_DIRS = (
    "data/captured_objects",
    "captured_objects",
)
# Canonical (rest-frame) per-object PLY directory under the raw root.
OBJ_MESH_REST_DIR = "data/rest_object_geo"

BODY_COLOR_RGBA: Tuple[int, int, int, int] = (220, 200, 180, 255)
OBJ_COLOR_RGBA: Tuple[int, int, int, int] = (200, 130, 60, 255)
GROUND_COLOR_RGBA: Tuple[int, int, int, int] = (160, 160, 160, 255)


# ----------------------------------------------------------------------
# Geometry helpers (self-contained — no sibling-module imports)
# ----------------------------------------------------------------------

def _color_mesh(
    mesh: trimesh.Trimesh, rgba: Tuple[int, int, int, int]
) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` with a uniform vertex colour. Preserves
    vertices and faces; does not re-process / weld."""
    n = mesh.vertices.shape[0]
    colors = np.tile(np.asarray(rgba, dtype=np.uint8), (n, 1))
    return trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        vertex_colors=colors,
        process=False,
    )


def _make_ground_plane(half_size: float = 5.0, z: float = 0.0) -> trimesh.Trimesh:
    """Grey square at ``z``, side = 2*``half_size`` (Z-up world)."""
    verts = np.array(
        [
            [-half_size, -half_size, z],
            [half_size, -half_size, z],
            [half_size, half_size, z],
            [-half_size, half_size, z],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    col = np.tile(np.asarray(GROUND_COLOR_RGBA, dtype=np.uint8), (4, 1))
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=col, process=False)


def _look_at_z_up(
    eye: np.ndarray,
    target: np.ndarray,
    up: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Z-up ``look-at`` returning the camera-to-world 4x4 matrix in OpenGL
    conventions (camera looks down -Z, +Y is image-up)."""
    if up is None:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - eye
    fn = np.linalg.norm(forward)
    if fn < 1e-9:
        raise ValueError(f"camera eye and target coincide: {eye} {target}")
    forward /= fn
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        raise ValueError(f"camera up parallel to forward (eye={eye}, target={target})")
    right /= rn
    cam_up = np.cross(right, forward)
    M = np.eye(4, dtype=np.float64)
    M[:3, 0] = right
    M[:3, 1] = cam_up
    M[:3, 2] = -forward
    M[:3, 3] = eye
    return M


def _camera_pose_diagonal(target: np.ndarray, dist: float = 3.0) -> np.ndarray:
    """Diagonal front-right camera, roughly eye-height."""
    tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
    eye = np.array([tx + 0.8 * dist, ty - 1.0 * dist, tz + 0.7], dtype=np.float64)
    return _look_at_z_up(eye, np.array([tx, ty, tz], dtype=np.float64))


# ----------------------------------------------------------------------
# Object mesh resolver
# ----------------------------------------------------------------------

def _strip_obj_name(rel_path: str) -> str:
    """Strip the OMOMO-raw filename suffix to recover the bare object name.

    The npz stores the raw-scan path, e.g.
    ``captured_objects/clothesstand_cleaned_simplified.obj``. The rest-frame
    PLY filename is just the bare name, e.g. ``clothesstand.ply``. This
    helper returns ``clothesstand`` so the caller can construct either path.
    """
    base = Path(rel_path).name
    # Known suffix pattern for captured OMOMO scans.
    SUFFIXES = ("_cleaned_simplified.obj", "_cleaned_simplified.ply",
                "_simplified.obj", ".obj", ".ply")
    for suf in SUFFIXES:
        if base.endswith(suf):
            return base[: -len(suf)]
    raise ValueError(
        f"unknown obj suffix in {rel_path!r}; expected one of {SUFFIXES}"
    )


def _resolve_obj_mesh_path(
    raw_root: Path,
    rel_path: str,
    mesh_source: str = "rest",
) -> Path:
    """Resolve the object mesh path to render.

    The CANONICAL (rest-frame) mesh at
    ``<raw_root>/data/rest_object_geo/<name>.ply`` is preferred because the
    ``obj_trans`` / ``obj_rot`` transforms stored in each npz window are
    defined in the rest-frame coordinate system. The raw-scan mesh at
    ``<raw_root>/(data/)?captured_objects/<name>_cleaned_simplified.obj`` is
    in the unprocessed capture frame (extents >=4m, off-centre, wrong
    orientation) and MUST NOT be rendered with the stored transforms.

    Args:
        raw_root: raw OMOMO download root (e.g. ``data/data/omomo/raw``).
        rel_path: relative path stored in the npz's ``obj_mesh_path`` field.
        mesh_source: ``"rest"`` (default, production) or ``"raw"`` (debug).

    Returns:
        absolute path to the chosen mesh file.

    Raises:
        FileNotFoundError: no viable path exists. Intentionally fails loud
            when ``rest`` is selected and the PLY is missing — silent
            fallback to the raw-scan mesh would render garbage.
        ValueError: unknown ``mesh_source`` value.
    """
    if mesh_source not in ("rest", "raw"):
        raise ValueError(
            f"mesh_source must be 'rest' or 'raw', got {mesh_source!r}"
        )

    name = _strip_obj_name(rel_path)

    if mesh_source == "rest":
        rest_path = raw_root / OBJ_MESH_REST_DIR / f"{name}.ply"
        if rest_path.exists():
            return rest_path
        raise FileNotFoundError(
            f"rest mesh not found at {rest_path}; refusing to fall back to "
            f"raw captured_objects/*.obj which is not in canonical frame. "
            f"Pass --mesh_source raw only for diagnostic use."
        )

    # mesh_source == "raw"
    rel = rel_path.lstrip("/")
    candidates = [raw_root / rel]
    for d in OBJ_MESH_RAW_SEARCH_DIRS:
        candidates.append(raw_root / d / Path(rel).name)
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not locate raw object mesh {rel_path!r} under {raw_root}. Tried:\n  {tried}"
    )


# ----------------------------------------------------------------------
# SMPL-X forward kinematics helper (shared by viewer + renderer)
# ----------------------------------------------------------------------

class _SmplxFK:
    """Cache of per-gender smplx.SMPLX models + stateless forward kinematics
    on the five OMOMO-npz fields (root_trans, root_orient_aa, pose_body_aa,
    betas, gender).

    Uses ``flat_hand_mean=False, use_pca=False`` (: flipped
    from True so zero-pose hands render as a natural resting curl instead of
    flat splayed palms). The Q12 schema does not include hand_pose_aa (CHOIS
    protocol convention) so all hand poses default to zero; with
    ``flat_hand_mean=False`` SMPL-X's MANO-mean offset produces a plausible
    resting grasp shape. Original comment for posterity: matches HIMO config
    and
    OMOMO-raw SMPL-X conventions). CPU-only — fast enough for 120-frame
    windows (~0.3 s per sequence); avoids GPU import contention.
    """

    def __init__(self, smplx_dir: Path) -> None:
        self._smplx_dir = smplx_dir
        self._models: Dict[str, "smplx.SMPLX"] = {}

    def _get_model(self, gender: str, batch_size: int) -> "smplx.SMPLX":
        gender = gender.lower()
        if gender not in ("male", "female", "neutral"):
            raise ValueError(f"unsupported gender: {gender!r}")
        key = f"{gender}::{batch_size}"
        if key not in self._models:
            fname = f"SMPLX_{gender.upper()}.npz"
            model_path = self._smplx_dir / fname
            if not model_path.exists():
                raise FileNotFoundError(
                    f"SMPL-X model {model_path} not found (needed for gender {gender!r})"
                )
            self._models[key] = smplx.SMPLX(
                str(model_path),
                gender=gender,
                use_pca=False,
                flat_hand_mean=False,  # natural resting curl for zero hand_pose
                batch_size=batch_size,
                num_betas=16,
            )
        return self._models[key]

    def fk_sequence(
        self,
        root_trans: np.ndarray,      # (T, 3) float32
        root_orient_aa: np.ndarray,  # (T, 3) float32
        pose_body_aa: np.ndarray,    # (T, 21, 3) float32
        betas: np.ndarray,           # (16,) float32
        gender: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SMPL-X FK for a full sequence and return
        ``(verts (T, 10475, 3) float32, faces (F, 3) int32)`` in world frame."""
        T = int(root_trans.shape[0])
        if root_orient_aa.shape[0] != T or pose_body_aa.shape[0] != T:
            raise ValueError(
                f"sequence length mismatch: {root_trans.shape=} "
                f"{root_orient_aa.shape=} {pose_body_aa.shape=}"
            )

        model = self._get_model(gender, batch_size=T)
        with torch.no_grad():
            # pose_body_aa is (T, 21, 3) -> (T, 63) as smplx expects.
            body_pose = torch.from_numpy(
                pose_body_aa.reshape(T, 63).astype(np.float32)
            )
            global_orient = torch.from_numpy(root_orient_aa.astype(np.float32))  # (T, 3)
            transl = torch.from_numpy(root_trans.astype(np.float32))             # (T, 3)
            betas_t = torch.from_numpy(betas.reshape(-1).astype(np.float32))     # (16,)
            betas_b = betas_t.unsqueeze(0).expand(T, -1).contiguous()            # (T, 16)

            out = model(
                betas=betas_b,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=torch.zeros(T, 45),
                right_hand_pose=torch.zeros(T, 45),
                jaw_pose=torch.zeros(T, 3),
                leye_pose=torch.zeros(T, 3),
                reye_pose=torch.zeros(T, 3),
                expression=torch.zeros(T, 10),
                transl=transl,
                return_verts=True,
            )

        verts = out.vertices.cpu().numpy().astype(np.float32)  # (T, 10475, 3)
        faces = model.faces.astype(np.int32)                    # (F, 3)
        return verts, faces


# ----------------------------------------------------------------------
# OMOMO sequence loader
# ----------------------------------------------------------------------

class OmomoWindow:
    """Loads one ``*.npz`` window and produces per-frame body verts + object
    mesh transforms."""

    def __init__(
        self,
        npz_path: Path,
        fk: _SmplxFK,
        raw_root: Path,
        mesh_source: str = "rest",
    ) -> None:
        self.npz_path = npz_path
        self.name = npz_path.stem
        d = np.load(str(npz_path), allow_pickle=True)

        # Static fields
        self.betas: np.ndarray = np.asarray(d["betas"], dtype=np.float32).reshape(-1)
        self.gender: str = str(d["gender"].item())
        self.text: str = str(d["text"].item())
        self.obj_mesh_path: str = str(d["obj_mesh_path"].item())
        self.seq_name: str = str(d["seq_name"].item())
        self.window_idx: int = int(d["window_idx"].item())
        self.actual_len: int = int(d["actual_len"].item())
        self.T: int = int(d["T"].item())

        # Time-varying fields
        root_trans: np.ndarray = np.asarray(d["root_trans"], dtype=np.float32)
        root_orient_aa: np.ndarray = np.asarray(d["root_orient_aa"], dtype=np.float32)
        pose_body_aa: np.ndarray = np.asarray(d["pose_body_aa"], dtype=np.float32)
        self.obj_trans: np.ndarray = np.asarray(d["obj_trans"], dtype=np.float32)  # (T,3)
        self.obj_rot: np.ndarray = np.asarray(d["obj_rot"], dtype=np.float32)       # (T,3,3)

        # SMPL-X forward: (T, 10475, 3) verts + (F,3) faces
        self.body_verts, self.body_faces = fk.fk_sequence(
            root_trans, root_orient_aa, pose_body_aa, self.betas, self.gender
        )

        # Object mesh (loaded once, transformed per frame). Prefer canonical
        # rest-frame PLY because (obj_trans, obj_rot) are defined in that frame.
        obj_path = _resolve_obj_mesh_path(raw_root, self.obj_mesh_path, mesh_source=mesh_source)
        obj_loaded = trimesh.load(str(obj_path), force="mesh", process=False)
        if not isinstance(obj_loaded, trimesh.Trimesh):
            raise ValueError(
                f"expected a single mesh for {obj_path}, got {type(obj_loaded).__name__}"
            )
        self.obj_mesh_rest: trimesh.Trimesh = obj_loaded
        self.obj_mesh_file: Path = obj_path

    def body_mesh(self, t: int) -> trimesh.Trimesh:
        """Build a (10475, F) trimesh body for frame ``t``."""
        t = max(0, min(self.T - 1, int(t)))
        return trimesh.Trimesh(
            vertices=self.body_verts[t].astype(np.float64),
            faces=self.body_faces,
            process=False,
        )

    def object_mesh(self, t: int) -> trimesh.Trimesh:
        """Apply ``(R, trans)`` of frame ``t`` to the rest-pose object mesh."""
        t = max(0, min(self.T - 1, int(t)))
        R = self.obj_rot[t].astype(np.float64)    # (3, 3)
        trans = self.obj_trans[t].astype(np.float64)  # (3,)
        V = np.asarray(self.obj_mesh_rest.vertices, dtype=np.float64)  # (V, 3)
        V_w = V @ R.T + trans[None, :]  # (V, 3)
        return trimesh.Trimesh(
            vertices=V_w,
            faces=np.asarray(self.obj_mesh_rest.faces),
            process=False,
        )


# ----------------------------------------------------------------------
# Playback thread
# ----------------------------------------------------------------------

class _Player:
    """Background thread advancing ``frame_slider`` at ``fps``."""

    def __init__(self, frame_slider, total_frames_ref: Dict[str, int], fps: int) -> None:
        self._slider = frame_slider
        self._total_ref = total_frames_ref
        self._fps = max(1, int(fps))
        self._playing = False
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def set_playing(self, playing: bool) -> None:
        self._playing = playing

    def _run(self) -> None:
        period = 1.0 / float(self._fps)
        while not self._stop:
            t0 = time.time()
            if self._playing:
                total = max(1, int(self._total_ref.get("T", 1)))
                cur = int(self._slider.value)
                nxt = (cur + 1) % total
                self._slider.value = nxt
            elapsed = time.time() - t0
            time.sleep(max(0.0, period - elapsed))


# ----------------------------------------------------------------------
# Interactive viser app
# ----------------------------------------------------------------------

class OmomoViserApp:
    """Interactive viewer for OMOMO CHOIS-protocol windows."""

    def __init__(
        self,
        processed_dir: str,
        split: str = "test",
        port: int = 8190,
        host: str = "0.0.0.0",
        fps: int = 30,
        initial_seq: Optional[str] = None,
        raw_root: str = str(DEFAULT_RAW_ROOT),
        smplx_dir: str = str(DEFAULT_SMPLX_DIR),
        mesh_source: str = "rest",
        server: Optional[viser.ViserServer] = None,
    ) -> None:
        self._processed_dir = Path(processed_dir)
        self._split = split
        self._split_dir = self._processed_dir / split
        if not self._split_dir.exists():
            raise FileNotFoundError(f"{self._split_dir} not found")

        self._raw_root = Path(raw_root)
        self._mesh_source = mesh_source
        self._fps = fps
        self._available_seqs = sorted(p.stem for p in self._split_dir.glob("*.npz"))
        if not self._available_seqs:
            raise RuntimeError(f"no .npz windows under {self._split_dir}")
        if initial_seq is None or initial_seq not in self._available_seqs:
            initial_seq = self._available_seqs[0]

        self._fk = _SmplxFK(Path(smplx_dir))

        self._server = server or viser.ViserServer(host=host, port=port)
        self._server.scene.set_up_direction("+z")
        self._server.scene.add_grid(
            "/ground",
            width=10.0,
            height=10.0,
            position=(0.0, 0.0, 0.0),
            cell_size=0.5,
            section_size=1.0,
        )

        with self._server.gui.add_folder("Sequence"):
            self._gui_dropdown = self._server.gui.add_dropdown(
                "Select", options=tuple(self._available_seqs), initial_value=initial_seq
            )
            self._gui_load_btn = self._server.gui.add_button("Load selected")
            self._gui_seq_label = self._server.gui.add_text(
                "Loaded", initial_value="", disabled=True
            )
            self._gui_text = self._server.gui.add_text(
                "Text", initial_value="", disabled=True
            )
            self._gui_gender = self._server.gui.add_text(
                "Gender", initial_value="", disabled=True
            )
            self._gui_object = self._server.gui.add_text(
                "Object", initial_value="", disabled=True
            )
            self._gui_frame_label = self._server.gui.add_text(
                "Frame", initial_value="", disabled=True
            )

        with self._server.gui.add_folder("Playback"):
            self._frame_slider = self._server.gui.add_slider(
                "Frame", min=0, max=1, step=1, initial_value=0
            )
            self._play_btn = self._server.gui.add_button("Play / Pause")

        self._window: Optional[OmomoWindow] = None
        self._body_handle: Optional[viser.SceneNodeHandle] = None
        self._obj_handle: Optional[viser.SceneNodeHandle] = None

        self._total_ref: Dict[str, int] = {"T": 1}
        self._player = _Player(self._frame_slider, self._total_ref, fps)
        self._is_playing = False
        # Guards load_seq (state swap) against render_viser_frame (scene read/write)
        # when fired concurrently from the player-thread slider callback.
        self._lock = threading.Lock()

        @self._gui_load_btn.on_click
        def _load_cb(_evt) -> None:  # noqa: ANN001
            self.load_seq(self._gui_dropdown.value)

        @self._play_btn.on_click
        def _toggle_cb(_evt) -> None:  # noqa: ANN001
            self._is_playing = not self._is_playing
            self._player.set_playing(self._is_playing)
            logger.info("Play/Pause -> %s", self._is_playing)

        @self._frame_slider.on_update
        def _on_frame_cb(_evt) -> None:  # noqa: ANN001
            if self._window is not None:
                self.render_viser_frame(int(self._frame_slider.value))

        self.load_seq(initial_seq)
        self._player.start()

    # ------------------------------------------------------------------
    @property
    def current_seq_name(self) -> str:
        return self._window.name if self._window is not None else ""

    @property
    def current_window(self) -> Optional[OmomoWindow]:
        return self._window

    def load_seq(self, seq_name: str) -> None:
        # Pause the player FIRST, before touching state or acquiring the lock,
        # and drain for one frame period so any in-flight slider-callback from
        # the player thread completes before we swap self._window out under it.
        was_playing = self._is_playing
        self._player.set_playing(False)
        time.sleep(max(1.0 / max(1, self._player._fps), 0.05))  # noqa: SLF001

        with self._lock:
            if seq_name not in self._available_seqs:
                raise KeyError(f"{seq_name!r} not in split {self._split!r}")
            logger.info("load_seq: %s", seq_name)

            npz_path = self._split_dir / f"{seq_name}.npz"
            new_w = OmomoWindow(npz_path, self._fk, self._raw_root, mesh_source=self._mesh_source)

            self._window = new_w
            self._total_ref["T"] = int(new_w.T)
            self._frame_slider.max = max(0, new_w.T - 1)
            self._frame_slider.value = 0

            self._gui_seq_label.value = seq_name
            self._gui_text.value = new_w.text[:200]
            self._gui_gender.value = new_w.gender
            self._gui_object.value = Path(new_w.obj_mesh_path).name
            self._gui_frame_label.value = f"0/{new_w.T - 1}"

            self._render_viser_frame_locked(0)

        self._player.set_playing(was_playing)

    def render_viser_frame(self, t: int) -> None:
        with self._lock:
            self._render_viser_frame_locked(t)

    def _render_viser_frame_locked(self, t: int) -> None:
        """Inner render implementation. Caller MUST hold ``self._lock``."""
        w = self._window
        if w is None:
            return
        t = max(0, min(w.T - 1, int(t)))
        body = w.body_mesh(t)
        obj = w.object_mesh(t)
        self._body_handle = self._server.scene.add_mesh_trimesh(
            "/body", _color_mesh(body, BODY_COLOR_RGBA)
        )
        self._obj_handle = self._server.scene.add_mesh_trimesh(
            "/object", _color_mesh(obj, OBJ_COLOR_RGBA)
        )
        self._gui_frame_label.value = f"{t}/{w.T - 1}"

    def shutdown(self) -> None:
        self._player.stop()
        self._server.stop()


# ----------------------------------------------------------------------
# Headless JPEG renderer
# ----------------------------------------------------------------------

def _render_scene_to_jpeg(
    body_mesh: trimesh.Trimesh,
    obj_mesh: trimesh.Trimesh,
    out_path: Path,
    width: int = 960,
    height: int = 960,
    quality: int = 92,
) -> None:
    """Render a body + object + ground plane scene (Z-up) to a JPEG via
    pyrender offscreen (EGL). Framed on the body bbox centre."""
    # Frame BOTH body and object — pick centre of their combined bbox so the
    # object is never off-screen. Pull the camera back in proportion to the
    # diagonal extent so tall / wide scenes stay in frame.
    b_bbox = body_mesh.bounds    # (2, 3)
    o_bbox = obj_mesh.bounds     # (2, 3)
    both_min = np.minimum(b_bbox[0], o_bbox[0])
    both_max = np.maximum(b_bbox[1], o_bbox[1])
    center = 0.5 * (both_min + both_max)
    diag = float(np.linalg.norm(both_max - both_min))
    # Aim at upper-body region but biased toward combined-scene centre so the
    # object stays visible.
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
    scene.add(pyrender.Mesh.from_trimesh(_make_ground_plane(half_size=4.0, z=0.0), smooth=False))
    scene.add(pyrender.Mesh.from_trimesh(_color_mesh(body_mesh, BODY_COLOR_RGBA), smooth=False))
    scene.add(pyrender.Mesh.from_trimesh(_color_mesh(obj_mesh, OBJ_COLOR_RGBA), smooth=False))

    yfov = float(np.deg2rad(45.0))
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=width / float(height))
    scene.add(cam, pose=cam_pose)

    key = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=4.0)
    scene.add(key, pose=_look_at_z_up(eye=target + np.array([2.0, -2.0, 4.0]), target=target))
    fill = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)
    scene.add(fill, pose=_look_at_z_up(eye=target + np.array([-3.0, 2.0, 2.0]), target=target))
    rim = pyrender.DirectionalLight(color=np.ones(3, dtype=np.float32), intensity=2.0)
    scene.add(rim, pose=_look_at_z_up(eye=target + np.array([0.0, 3.0, 1.5]), target=target))

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    try:
        color, _depth = renderer.render(scene)
    finally:
        renderer.delete()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color).convert("RGB").save(str(out_path), format="JPEG", quality=quality)


def render_frame(
    processed_dir: str,
    split: str,
    seq_name: str,
    frame: int,
    out_path: str,
    raw_root: str = str(DEFAULT_RAW_ROOT),
    smplx_dir: str = str(DEFAULT_SMPLX_DIR),
    width: int = 960,
    height: int = 960,
    fk_cache: Optional[_SmplxFK] = None,
    window_cache: Optional[OmomoWindow] = None,
    mesh_source: str = "rest",
) -> None:
    """Headless render of one frame of one window to ``out_path`` (JPEG).

    Args:
        processed_dir: e.g. ``data/data/omomo/umm``.
        split: ``"test"`` or ``"train"``.
        seq_name: window stem, e.g. ``sub16_clothesstand_000_w0``.
        frame: integer in [0, T-1].
        out_path: output JPEG path.
        raw_root: OMOMO raw root for object mesh lookup.
        smplx_dir: directory holding ``SMPLX_{NEUTRAL,MALE,FEMALE}.npz``.
        width, height: output resolution (both default 960 -> >=800x800 guard).
        fk_cache, window_cache: optional reuse to avoid reloading models/meshes
            when rendering multiple frames of the same window.
    """
    fk = fk_cache if fk_cache is not None else _SmplxFK(Path(smplx_dir))
    if window_cache is not None and window_cache.name == seq_name:
        w = window_cache
    else:
        npz_path = Path(processed_dir) / split / f"{seq_name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"{npz_path} not found")
        w = OmomoWindow(npz_path, fk, Path(raw_root), mesh_source=mesh_source)

    body = w.body_mesh(frame)
    obj = w.object_mesh(frame)
    _render_scene_to_jpeg(body, obj, Path(out_path), width=width, height=height)


# ----------------------------------------------------------------------
# CLI — interactive viewer (default) + render subcommand
# ----------------------------------------------------------------------

def run_viewer(
    processed_dir: str,
    split: str = "test",
    port: int = 8190,
    host: str = "0.0.0.0",
    fps: int = 30,
    initial_seq: Optional[str] = None,
    raw_root: str = str(DEFAULT_RAW_ROOT),
    smplx_dir: str = str(DEFAULT_SMPLX_DIR),
    mesh_source: str = "rest",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = OmomoViserApp(
        processed_dir=processed_dir,
        split=split,
        port=port,
        host=host,
        fps=fps,
        initial_seq=initial_seq,
        raw_root=raw_root,
        smplx_dir=smplx_dir,
        mesh_source=mesh_source,
    )
    logger.info("listening *:%d  URL=http://localhost:%d", port, port)
    logger.info("split=%s  available seqs=%d", split, len(app._available_seqs))  # noqa: SLF001
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        app.shutdown()


def _build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argparser.

    Common args live SOLELY on the top-level parser. The ``render`` subparser
    carries ONLY render-specific options. This prevents the silent
    subparser-default-overwrites-top-level-value bug that a duplicated
    ``--processed_dir`` (or any shared option) would produce.
    """
    p = argparse.ArgumentParser(description=__doc__)

    # -- common args (top-level ONLY — never duplicated on subparsers) --
    p.add_argument("--processed_dir", type=str, default="data/data/omomo/umm")
    p.add_argument("--split", type=str, default="test", choices=("test", "train"))
    p.add_argument("--raw_root", type=str, default=str(DEFAULT_RAW_ROOT))
    p.add_argument("--smplx_dir", type=str, default=str(DEFAULT_SMPLX_DIR))
    p.add_argument(
        "--mesh_source",
        type=str,
        default="rest",
        choices=("rest", "raw"),
        help=(
            "Which object mesh to load. 'rest' (default, production) uses "
            "data/rest_object_geo/<name>.ply — the canonical frame the "
            "stored (obj_trans, obj_rot) apply to. 'raw' loads the raw scan "
            "under captured_objects/ and is for diagnostic / parity "
            "inspection only; do NOT use 'raw' for training viz."
        ),
    )
    p.add_argument("--port", type=int, default=8190)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--seq",
        type=str,
        default=None,
        help=(
            "Window stem (e.g. sub16_clothesstand_000_w0). Optional for the "
            "interactive viewer (defaults to first). REQUIRED when the "
            "'render' subcommand is used — the main() entry checks this."
        ),
    )

    # -- subcommand: headless render (render-specific options ONLY) --
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser(
        "render",
        help="Headless JPEG render of one window's frames.",
        description=(
            "Render JPEG frames of a single window to disk. All data-path "
            "options (--processed_dir, --split, --seq, --raw_root, "
            "--smplx_dir, --mesh_source) are TOP-LEVEL args and must be "
            "passed BEFORE the 'render' subcommand."
        ),
    )
    r.add_argument("--frames", type=int, nargs="+", required=True)
    r.add_argument("--out_dir", type=str, required=True)
    r.add_argument("--width", type=int, default=960)
    r.add_argument("--height", type=int, default=960)
    return p


def main() -> None:
    args = _build_argparser().parse_args()
    if getattr(args, "cmd", None) == "render":
        if args.seq is None:
            raise SystemExit(
                "error: --seq is required with the 'render' subcommand. "
                "Usage: python -m prepare_data.omomo.omomo_viser "
                "--seq <stem> render --frames <F> --out_dir <DIR>"
            )
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fk = _SmplxFK(Path(args.smplx_dir))
        npz_path = Path(args.processed_dir) / args.split / f"{args.seq}.npz"
        w = OmomoWindow(npz_path, fk, Path(args.raw_root), mesh_source=args.mesh_source)
        for f in args.frames:
            out_jpg = out_dir / f"{args.seq}_f{int(f):04d}.jpg"
            render_frame(
                processed_dir=args.processed_dir,
                split=args.split,
                seq_name=args.seq,
                frame=int(f),
                out_path=str(out_jpg),
                raw_root=args.raw_root,
                smplx_dir=args.smplx_dir,
                width=args.width,
                height=args.height,
                fk_cache=fk,
                window_cache=w,
                mesh_source=args.mesh_source,
            )
            logger.info("wrote %s", out_jpg)
        return

    run_viewer(
        processed_dir=args.processed_dir,
        split=args.split,
        port=args.port,
        host=args.host,
        fps=args.fps,
        initial_seq=args.seq,
        raw_root=args.raw_root,
        smplx_dir=args.smplx_dir,
        mesh_source=args.mesh_source,
    )


if __name__ == "__main__":
    main()
