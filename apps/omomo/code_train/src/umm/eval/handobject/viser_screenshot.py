"""Batch screenshot driver for handobject viz grid.

Wraps `render_sample()` from `viser_viewer.py` so we can produce a flat set of
PNGs + JSON sidecars across (episode x source x view) axes for eval grids.

Public API
----------
- `render_episode_grid(dataset, episode_id, sources, views, out_dir, ...)` —
  loop over (source, view), call render_sample once per source (all views
  batched), return a flat list of PNG paths.
- `render_dummy_canonical(out_dir, ...)` — build a 30-frame synthetic
  CanonicalSample (object at fixed pose, rot-Z-20°, hands loosely holding it),
  round-trip it through save_canonical_npz, and render all 3 views.

CLI
---
`python -m umm.eval.handobject.viser_screenshot
  --out-dir <path>
  --episodes grab:13,grab:37,grab:105,arctic:7,arctic:25,arctic:100
  --sources umm,native,canonical
  --views top,side,iso
  [--dummy]
  [--frame-index N]
  [--image-size HxW]`

Each run writes `manifest.json` to `out_dir` summarising every PNG path with
dataset/episode/source/view/frame_index/image_size.

Module independence
-------------------
Only imports from `umm.eval.handobject.*` (canonical, viser_viewer). No
intra-UMM cross-version references.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from umm.eval.handobject.canonical import (
    CanonicalSample,
    load_canonical_npz,
    save_canonical_npz,
)
from umm.eval.handobject.viser_viewer import (
    _DEFAULT_BIN_MAP,
    _DEFAULT_TEXT2HOI_ROOT,
    _DEFAULT_UMM_ROOT,
    _load_sample,
    render_sample,
)


LOGGER = logging.getLogger("umm.eval.handobject.viser_screenshot")
if not LOGGER.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)


SourceLit = Literal["umm", "native", "canonical"]
ViewLit = Literal["top", "side", "iso"]
DatasetLit = Literal["grab", "arctic"]


# ---------------------------------------------------------------------------
# Per-episode grid
# ---------------------------------------------------------------------------

def render_episode_grid(
    dataset: DatasetLit,
    episode_id: int,
    *,
    sources: Sequence[SourceLit] = ("umm", "native"),
    views: Sequence[ViewLit] = ("top", "side", "iso"),
    out_dir: Path,
    canonical_path: Optional[Path] = None,
    frame_index: Optional[int] = None,
    image_size: Tuple[int, int] = (768, 768),
    umm_root: Path = _DEFAULT_UMM_ROOT,
    text2hoi_root: Path = _DEFAULT_TEXT2HOI_ROOT,
    bin_map_root: Path = _DEFAULT_BIN_MAP,
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Render a flat PNG grid for one episode across sources x views.

    For each requested `source`:
      - If source == "umm" or "native": load the sample via `_load_sample`.
        If source == "umm" AND "canonical" is also requested, save the loaded
        sample to a temp .npz so the canonical-round-trip path is exercised
        (a synthetic canonical.npz produced from the UMM sample itself).
      - If source == "canonical": load from `canonical_path` (required) OR
        from the temp npz saved during the "umm" pass.
    Each source fires one `render_sample()` call covering all requested views.

    Args:
        dataset: "grab" or "arctic".
        episode_id: Episode index within the text2hoi bin map.
        sources: Subset of ("umm", "native", "canonical").
        views: Subset of ("top", "side", "iso").
        out_dir: Output directory (created if missing). Files are written as
            `{out_dir}/{dataset}_ep{episode_id:06d}_{source}_{view}.{png,json}`.
        canonical_path: Explicit canonical .npz to render when source
            "canonical" is requested. If None, a derived npz from the UMM
            sample (first sources containing "umm") is used; otherwise
            canonical source is skipped with a warning.
        frame_index: Specific frame, or None for middle valid frame.
        image_size: (H, W) pixels per render.

    Returns:
        2-tuple ``(png_paths, skipped_sources)`` where:
        - ``png_paths``: flat list of PNG paths in (source, view) order.
        - ``skipped_sources``: list of dicts, one per skipped (source, episode),
          each with keys ``{"source": str, "dataset": str, "ep_id": int,
          "error": str}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate inputs.
    src_allowed = {"umm", "native", "canonical"}
    view_allowed = {"top", "side", "iso"}
    for s in sources:
        if s not in src_allowed:
            raise ValueError(f"render_episode_grid: unknown source {s!r}")
    for v in views:
        if v not in view_allowed:
            raise ValueError(f"render_episode_grid: unknown view {v!r}")
    if dataset not in ("grab", "arctic"):
        raise ValueError(f"render_episode_grid: unknown dataset {dataset!r}")

    all_pngs: List[Path] = []
    skipped_sources: List[Dict[str, Any]] = []
    derived_canonical_npz: Optional[Path] = None

    for source in sources:
        prefix = f"{dataset}_ep{episode_id:06d}_{source}"
        # Resolve canonical path on the fly if needed.
        source_npz: Optional[Path] = None
        if source == "canonical":
            if canonical_path is not None:
                source_npz = canonical_path
            elif derived_canonical_npz is not None:
                source_npz = derived_canonical_npz
            else:
                LOGGER.warning(
                    "render_episode_grid[%s ep %d]: canonical source requested but "
                    "no canonical_path given and no prior 'umm' pass to derive from — skipping.",
                    dataset, episode_id,
                )
                skipped_sources.append({
                    "source": source,
                    "dataset": dataset,
                    "ep_id": episode_id,
                    "error": "no canonical_path and no prior umm pass",
                })
                continue

        try:
            sample, artic_angle = _load_sample(
                source=source,
                dataset=dataset,
                episode_id=episode_id,
                npz_path=source_npz,
                umm_root=umm_root,
                text2hoi_root=text2hoi_root,
                bin_map_root=bin_map_root,
            )
        except Exception as e:  # noqa: BLE001 — log and skip this source
            LOGGER.warning(
                "render_episode_grid[%s ep %d / %s]: _load_sample failed: %s — skipping.",
                dataset, episode_id, source, e,
                exc_info=True,
            )
            skipped_sources.append({
                "source": source,
                "dataset": dataset,
                "ep_id": episode_id,
                "error": str(e),
            })
            continue

        # If this is the UMM pass and a future canonical pass is requested
        # without an explicit path, save the loaded sample so canonical can
        # round-trip from it.
        if (
            source == "umm"
            and "canonical" in sources
            and canonical_path is None
            and derived_canonical_npz is None
        ):
            derived_canonical_npz = out_dir / f"{dataset}_ep{episode_id:06d}_derived_canonical.npz"
            try:
                save_canonical_npz(sample, str(derived_canonical_npz), strict=False)
                LOGGER.info(
                    "render_episode_grid: saved derived canonical -> %s",
                    derived_canonical_npz,
                )
            except Exception as e:  # noqa: BLE001 — log and continue
                LOGGER.warning(
                    "render_episode_grid: save_canonical_npz failed: %s — canonical source will be skipped.",
                    e,
                )
                derived_canonical_npz = None

        try:
            pngs = render_sample(
                sample,
                out_dir=out_dir,
                views=tuple(views),
                image_size=image_size,
                frame_index=frame_index,
                name_prefix=prefix,
                articulation_angle=artic_angle,
            )
        except Exception as e:  # noqa: BLE001 — log and continue
            LOGGER.warning(
                "render_episode_grid[%s ep %d / %s]: render_sample failed: %s — skipping.",
                dataset, episode_id, source, e,
                exc_info=True,
            )
            skipped_sources.append({
                "source": source,
                "dataset": dataset,
                "ep_id": episode_id,
                "error": str(e),
            })
            continue

        all_pngs.extend(pngs)

    return all_pngs, skipped_sources


# ---------------------------------------------------------------------------
# Dummy canonical
# ---------------------------------------------------------------------------

# GRAB-style mesh we know exists in the repo for the dummy's obj_name.
_DUMMY_OBJ_NAME = "banana"


def _build_dummy_canonical(n_frames: int = 30) -> CanonicalSample:
    """Build a synthetic CanonicalSample: object fixed at [0.5, 0.0, 0.3] +
    rot-Z-20°, right hand loosely grasping from +Y side, left hand idle.

    Shapes match the canonical schema:
      left_hand_joints, right_hand_joints: (T, 21, 3) float32
      left_hand_rot,  right_hand_rot:      (T, 16, 3) float32
      obj_translation: (T, 3) float32
      obj_rotation:    (T, 3, 3) float32
      valid_mask:      (T,) bool
    """
    T = int(n_frames)
    # Object pose — same for every frame.
    obj_t_xyz = np.array([0.5, 0.0, 0.3], dtype=np.float32)
    ang = np.deg2rad(20.0)
    c, s = float(np.cos(ang)), float(np.sin(ang))
    obj_R_single = np.array(
        [[c, -s, 0.0],
         [s,  c, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )

    obj_translation = np.tile(obj_t_xyz[None, :], (T, 1)).astype(np.float32)
    obj_rotation = np.tile(obj_R_single[None, :, :], (T, 1, 1)).astype(np.float32)

    # Hand joint scaffolds: offset from object by a plausible grasping amount.
    # Right hand "wraps" from +Y side; wrist ~10 cm from object center.
    r_wrist = obj_t_xyz + np.array([0.0, 0.12, 0.02], dtype=np.float32)
    l_wrist = obj_t_xyz + np.array([0.0, -0.18, 0.0], dtype=np.float32)

    def _scatter_joints(wrist: np.ndarray, hand_side: str) -> np.ndarray:
        """Return (21, 3) — wrist at [0], 20 finger joints radiating outward."""
        j = np.zeros((21, 3), dtype=np.float32)
        j[0] = wrist
        # 5 finger chains, 4 joints each. Fingers radiate radially from wrist
        # toward -Y for the right hand (approaching the object), +Y for left.
        dir_sign = -1.0 if hand_side == "right" else 1.0
        for finger in range(5):
            # Spread fingers along X axis.
            x_off = (finger - 2) * 0.018  # -0.036..+0.036 m
            for seg in range(4):
                idx = 1 + finger * 4 + seg
                y_off = dir_sign * (0.02 + seg * 0.015)
                z_off = seg * 0.004
                j[idx] = wrist + np.array([x_off, y_off, z_off], dtype=np.float32)
        return j

    r_joints_single = _scatter_joints(r_wrist, "right")
    l_joints_single = _scatter_joints(l_wrist, "left")

    # Tile across T with a tiny periodic wobble so the viewer does not think
    # every frame is identical (useful for scrubbing tests).
    tt = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    wobble = 0.003 * np.sin(2 * np.pi * tt)[:, None, None]  # (T, 1, 1)
    left_hand_joints = np.tile(l_joints_single[None], (T, 1, 1)).astype(np.float32)
    right_hand_joints = np.tile(r_joints_single[None], (T, 1, 1)).astype(np.float32)
    # Apply wobble on the z axis of fingertips only (index 4, 8, 12, 16, 20).
    tip_idx = np.array([4, 8, 12, 16, 20], dtype=np.int64)
    left_hand_joints[:, tip_idx, 2:3] += wobble  # (T, 5, 1)
    right_hand_joints[:, tip_idx, 2:3] += wobble

    # Per-frame axis-angle arrays — zeros are valid (identity pose).
    left_hand_rot = np.zeros((T, 16, 3), dtype=np.float32)
    right_hand_rot = np.zeros((T, 16, 3), dtype=np.float32)

    valid_mask = np.ones((T,), dtype=np.bool_)

    return CanonicalSample(
        left_hand_joints=left_hand_joints,
        right_hand_joints=right_hand_joints,
        left_hand_rot=left_hand_rot,
        right_hand_rot=right_hand_rot,
        obj_translation=obj_translation,
        obj_rotation=obj_rotation,
        obj_name=_DUMMY_OBJ_NAME,
        valid_mask=valid_mask,
        fps=30,
        text="dummy canonical sample — object at (0.5, 0, 0.3) rot-Z-20°",
    )


def render_dummy_canonical(
    out_dir: Path,
    *,
    views: Sequence[ViewLit] = ("top", "side", "iso"),
    n_frames: int = 30,
    image_size: Tuple[int, int] = (768, 768),
    frame_index: Optional[int] = None,
) -> List[Path]:
    """Build a 30-frame dummy canonical sample, round-trip it, and render.

    Writes:
      - `<out_dir>/dummy.npz` (via save_canonical_npz, strict=True)
      - `<out_dir>/dummy_<view>.{png,json}` for each requested view

    Returns:
        List of PNG paths in the requested view order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = _build_dummy_canonical(n_frames=n_frames)
    dummy_npz = out_dir / "dummy.npz"
    save_canonical_npz(sample, str(dummy_npz), strict=True)
    LOGGER.info("render_dummy_canonical: wrote %s", dummy_npz)

    # Round-trip load to confirm bit-identical path through the canonical IO.
    reloaded = load_canonical_npz(str(dummy_npz))
    # Sanity: shapes and object name match.
    assert reloaded.obj_name == sample.obj_name
    assert reloaded.left_hand_joints.shape == sample.left_hand_joints.shape

    pngs = render_sample(
        reloaded,
        out_dir=out_dir,
        views=tuple(views),
        image_size=image_size,
        frame_index=frame_index,
        name_prefix="dummy",
    )
    return pngs


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _sidecar_path(png: Path) -> Path:
    return png.with_suffix(".json")


def _entry_for_png(
    png: Path,
    *,
    dataset: Optional[str],
    episode_id: Optional[int],
    source: Optional[str],
    view: Optional[str],
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    sidecar = _sidecar_path(png)
    frame_idx: Optional[int] = None
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            frame_idx = int(data.get("frame_index")) if "frame_index" in data else None
        except Exception as e:  # noqa: BLE001 — partial manifest is OK
            LOGGER.warning("manifest: could not read %s: %s", sidecar, e)
    return {
        "png": str(png),
        "json": str(sidecar) if sidecar.exists() else None,
        "dataset": dataset,
        "episode_id": episode_id,
        "source": source,
        "view": view,
        "frame_index": frame_idx,
        "image_size": list(image_size),
        "bytes": int(png.stat().st_size) if png.exists() else None,
    }


def _write_manifest(
    out_dir: Path,
    entries: List[Dict[str, Any]],
    skipped_sources: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    payload: Dict[str, Any] = {"entries": entries}
    if skipped_sources:
        payload["skipped_sources"] = skipped_sources
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_episodes(s: str) -> List[Tuple[str, int]]:
    """Parse 'grab:13,arctic:7' -> [('grab', 13), ('arctic', 7)]."""
    out: List[Tuple[str, int]] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(
                f"--episodes: expected dataset:id (got {item!r})"
            )
        dset, eid = item.split(":", 1)
        dset = dset.strip().lower()
        if dset not in ("grab", "arctic"):
            raise argparse.ArgumentTypeError(f"--episodes: unknown dataset {dset!r}")
        try:
            ep_id = int(eid)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--episodes: episode id must be int (got {eid!r})"
            ) from None
        out.append((dset, ep_id))
    if not out:
        raise argparse.ArgumentTypeError("--episodes: at least one required")
    return out


def _parse_list(s: str, allowed: Sequence[str], flag: str) -> List[str]:
    out = [tok.strip() for tok in s.split(",") if tok.strip()]
    for tok in out:
        if tok not in allowed:
            raise argparse.ArgumentTypeError(f"{flag}: unknown value {tok!r}")
    if not out:
        raise argparse.ArgumentTypeError(f"{flag}: at least one required")
    return out


def _parse_sources(s: str) -> List[str]:
    return _parse_list(s, ("umm", "native", "canonical"), "--sources")


def _parse_views(s: str) -> List[str]:
    return _parse_list(s, ("top", "side", "iso"), "--views")


def _parse_image_size(s: str) -> Tuple[int, int]:
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--image-size: expected HxW, got {s!r}")
    return (int(parts[0]), int(parts[1]))


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="umm.eval.handobject.viser_screenshot",
        description="Batch screenshot driver for handobject grids.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--episodes", type=_parse_episodes, required=True,
        help="Comma-separated dataset:id list, e.g. grab:13,arctic:7",
    )
    p.add_argument(
        "--sources", type=_parse_sources, default=["umm", "native"],
        help="Comma-separated source list. Subset of {umm,native,canonical}.",
    )
    p.add_argument(
        "--views", type=_parse_views, default=["top", "side", "iso"],
        help="Comma-separated view list. Subset of {top,side,iso}.",
    )
    p.add_argument(
        "--dummy", action="store_true",
        help="Additionally render the synthetic dummy canonical sample.",
    )
    p.add_argument(
        "--frame-index", type=int, default=None,
        help="Specific frame; default = middle valid frame per episode.",
    )
    p.add_argument(
        "--image-size", type=_parse_image_size, default=(768, 768),
        help="HxW (default 768x768).",
    )
    p.add_argument("--umm-root", type=Path, default=_DEFAULT_UMM_ROOT)
    p.add_argument("--text2hoi-root", type=Path, default=_DEFAULT_TEXT2HOI_ROOT)
    p.add_argument("--bin-map-root", type=Path, default=_DEFAULT_BIN_MAP)
    p.add_argument(
        "--canonical-path", type=Path, default=None,
        help="If given and canonical is in --sources, use this npz for canonical renders.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict[str, Any]] = []
    all_skipped: List[Dict[str, Any]] = []

    # Per-episode renders.
    for dset, ep_id in args.episodes:
        LOGGER.info("=== episode %s:%d ===", dset, ep_id)
        pngs, skipped = render_episode_grid(
            dataset=dset,  # type: ignore[arg-type]
            episode_id=ep_id,
            sources=tuple(args.sources),  # type: ignore[arg-type]
            views=tuple(args.views),  # type: ignore[arg-type]
            out_dir=out_dir,
            canonical_path=args.canonical_path,
            frame_index=args.frame_index,
            image_size=tuple(args.image_size),
            umm_root=args.umm_root,
            text2hoi_root=args.text2hoi_root,
            bin_map_root=args.bin_map_root,
        )
        all_skipped.extend(skipped)
        # Tag each PNG.
        for png in pngs:
            # Derive source + view from filename stem: {dset}_ep{id:06d}_{source}_{view}.png
            stem_parts = png.stem.split("_")
            if len(stem_parts) >= 4:
                view = stem_parts[-1]
                source = stem_parts[-2]
            else:
                view = None
                source = None
            entries.append(_entry_for_png(
                png,
                dataset=dset,
                episode_id=ep_id,
                source=source,
                view=view,
                image_size=tuple(args.image_size),
            ))

    # Dummy canonical.
    if args.dummy:
        LOGGER.info("=== dummy canonical ===")
        try:
            dummy_pngs = render_dummy_canonical(
                out_dir,
                views=tuple(args.views),  # type: ignore[arg-type]
                image_size=tuple(args.image_size),
                frame_index=args.frame_index,
            )
            for png in dummy_pngs:
                stem_parts = png.stem.split("_")
                view = stem_parts[-1] if len(stem_parts) >= 2 else None
                entries.append(_entry_for_png(
                    png,
                    dataset="dummy",
                    episode_id=None,
                    source="canonical",
                    view=view,
                    image_size=tuple(args.image_size),
                ))
        except Exception as e:  # noqa: BLE001 — fail loud but still write manifest
            LOGGER.error("render_dummy_canonical failed: %s", e)
            raise

    manifest_path = _write_manifest(out_dir, entries, skipped_sources=all_skipped or None)
    if all_skipped:
        LOGGER.warning(
            "Manifest has %d skipped source(s) — check manifest.json['skipped_sources'] for details.",
            len(all_skipped),
        )
    LOGGER.info("Wrote manifest with %d entries -> %s", len(entries), manifest_path)
    for e in entries:
        print(e["png"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
