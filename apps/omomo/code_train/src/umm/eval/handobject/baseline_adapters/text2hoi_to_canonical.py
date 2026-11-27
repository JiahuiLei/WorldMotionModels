"""Text2HOI raw .npz -> canonical .npz adapter.

Converts the output of eval/handobject/text2hoi_sampler.py (one .npz file per
sample) into canonical schema files that 's eval_imos.py can score.

Input keys per raw .npz (see text2hoi_sampler.py docstring):
    x_lhand:   (T, 99) float32  -- [trans(0:3), rot6d_16joints(3:99)]
    x_rhand:   (T, 99) float32  -- same for right hand
    x_obj:     (T, 9 or 10) float32 -- [trans(0:3), rot6d(3:9), (angle(1) ARCTIC only)]
    j_lhand:   (T, 21, 3) float32 -- MANO world-frame joints (pre-computed)
    j_rhand:   (T, 21, 3) float32 -- same
    is_lhand:  () bool
    is_rhand:  () bool
    T_valid:   () int32
    T_padded:  () int32
    text:      () <U...
    obj_name:  () <U...  -- single-word GRAB name or ARCTIC name
    dataset:   () <U...  -- "grab" or "arctic"
    sample_idx: () int32
    prompt_idx: () int32

Output: <out-dir>/samples/NNNNNN.npz in canonical schema.

CLI:
    python -m umm.eval.handobject.baseline_adapters.text2hoi_to_canonical \\
        --raw-dir <dir_of_raw_npz_files> \\
        --out-dir <output_dir> \\
        --dataset {grab,arctic} \\
        [--max-samples N]
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
from typing import Optional

import numpy as np

from umm.eval.handobject.canonical import (
    CanonicalSample,
    save_canonical_npz,
    validate_canonical_sample,
)
from umm.eval.handobject.gt_loader import rot6d_to_axis_angle, rot6d_to_rotmat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GRAB object-name translation: Text2HOI single-word -> canonical space-separated
#
# Derived from two sources of truth:
#   1. PLY filenames in data/handobject/raw_registered/grab_unzip/contact_meshes/*.ply
#      (these are the single-word keys Text2HOI's search_object returns)
#   2. Canonical obj_name values across all GRAB umm splits (train/val/test)
#      (enumerated from data/handobject/umm/grab/{train,val,test}/*.npz)
#
# Canonical form: space-separated lowercase (e.g. "alarm clock").
# Text2HOI form: single-word lowercase (e.g. "alarmclock").
#
# Sized primitives (cube{large,medium,middle,small}, cylinder*, etc.) all map
# to the base canonical name because Text2HOI's text prompts reference just
# "cube" / "cylinder" / "pyramid" / "sphere" / "torus" without size qualifiers.
#
# Objects not in the canonical dataset (body, table) are kept as-is (identity
# mapping) so they fail loudly at canonical validation if they appear.
# ---------------------------------------------------------------------------

# Authoritative source: /tmp/Text2HOI/constants/grab_constants.py::object_proc AND grab_obj_name.
# Text2HOI's preprocessing_grab.py:122-125 applies `object_proc` to each episode's raw
# GRAB obj_name to produce `proc_obj_name`; raw names NOT in object_proc pass through
# unchanged. This is the EXACT rule canonical .npz stores (verified 2026-04-20 against
# grab train/val/test: 41 unique proc_obj_name values present).
#
# Consequences for the adapter:
# - Raw names like "coffeemug", "rubberduck", "wristwatch", "cubemiddle" are NOT in
#   object_proc -> they pass through as-is. Canonical dataset has no episodes of these
#   (Text2HOI's test_split prompts for those objects are rare); if Text2HOI's CLIP
#   search emits one, our strict-vocab eval will fail loudly (intentional — Text2HOI
#   emits an obj that we didn't train the classifier on).
# - All sized primitives listed in object_proc collapse to the base name ("cube",
#   "cylinder", "pyramid", "sphere", "torus"). cubemiddle is the ONE cube variant NOT
#   in object_proc and therefore NOT collapsed.

# Mirror of /tmp/Text2HOI/constants/grab_constants.py::object_proc as of commit
# captured 2026-04-20. If upstream changes, regenerate by copy-paste.
_OBJECT_PROC: dict = {
    "fryingpan": "frying pan",
    "waterbottle": "water bottle",
    "cubelarge": "cube",
    "cubemedium": "cube",
    "cubesmall": "cube",
    "cylinderlarge": "cylinder",
    "cylindermedium": "cylinder",
    "cylindersmall": "cylinder",
    "pyramidlarge": "pyramid",
    "pyramidmedium": "pyramid",
    "pyramidsmall": "pyramid",
    "spherelarge": "sphere",
    "spheremedium": "sphere",
    "spheresmall": "sphere",
    "stanfordbunny": "stanford bunny",
    "toruslarge": "torus",
    "torusmedium": "torus",
    "torussmall": "torus",
    "gamecontroller": "game controller",
    "piggybank": "piggy bank",
    "alarmclock": "alarm clock",
    "lightbulb": "light bulb",
}

# Mirror of /tmp/Text2HOI/constants/grab_constants.py::grab_obj_name (57 items).
# Used to validate that a raw name is a known GRAB object; anything else raises.
_GRAB_OBJ_NAME: frozenset = frozenset({
    "elephant", "headphones", "cylinderlarge", "hand", "spherelarge",
    "toruslarge", "wristwatch", "stapler", "flute", "table",
    "train", "pyramidmedium", "wineglass", "scissors", "gamecontroller",
    "spheresmall", "pyramidsmall", "knife", "mouse", "torusmedium",
    "watch", "cylindermedium", "phone", "banana", "cylindersmall",
    "apple", "toothbrush", "coffeemug", "bowl", "cup",
    "cubemedium", "pyramidlarge", "cubesmall", "alarmclock", "teapot",
    "duck", "camera", "waterbottle", "airplane", "fryingpan",
    "body", "lightbulb", "torussmall", "stamp", "hammer",
    "spheremedium", "binoculars", "doorknob", "toothpaste", "cubemiddle",
    "stanfordbunny", "cubelarge", "piggybank", "mug", "rubberduck",
    "eyeglasses", "flashlight",
})

# Build _GRAB_CANONICAL: apply object_proc where mapped, else identity.
_GRAB_CANONICAL: dict = {}
for _raw in _GRAB_OBJ_NAME:
    _GRAB_CANONICAL[_raw] = _OBJECT_PROC.get(_raw, _raw)


def _translate_obj_name_grab(raw_name: str) -> str:
    """Translate a Text2HOI GRAB object name to canonical space-separated form.

    Args:
        raw_name: Text2HOI single-word name (e.g. "alarmclock", "fryingpan").

    Returns:
        Canonical space-separated form (e.g. "alarm clock", "frying pan").

    Raises:
        KeyError: If raw_name is not in the known translation table.
    """
    canonical = _GRAB_CANONICAL.get(raw_name)
    if canonical is None:
        # Unknown name: not in GRAB PLY listing. Fail loud — silent pass-through
        # would corrupt downstream evaluation.
        raise KeyError(
            f"_translate_obj_name_grab: unknown GRAB object name {raw_name!r}. "
            f"Known names: {sorted(_GRAB_CANONICAL.keys())}"
        )
    return canonical


# ---------------------------------------------------------------------------
# Git SHA helper
# ---------------------------------------------------------------------------

def _git_sha() -> str:
    """Return current HEAD SHA (full). Returns 'unknown' if git is not available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        logger.warning(
            "_git_sha: `git rev-parse HEAD` returned code=%d stderr=%r",
            result.returncode, result.stderr.strip(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("_git_sha: git invocation failed: %s", exc)
    return "unknown"


# ---------------------------------------------------------------------------
# File SHA256
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core adapter: convert a single raw .npz to CanonicalSample
# ---------------------------------------------------------------------------

def _convert_single(raw_path: str, dataset: str) -> CanonicalSample:
    """Convert one raw Text2HOI .npz file to a CanonicalSample.

    Args:
        raw_path: Path to a raw .npz file produced by text2hoi_sampler.py.
        dataset:  "grab" or "arctic".

    Returns:
        Validated CanonicalSample (strict=False; strict orthonormality is
        checked by the caller via validate_canonical_sample).

    Raises:
        KeyError:   If a required key is missing from the raw file.
        ValueError: If shapes are inconsistent or obj_name is unknown.
    """
    raw = np.load(raw_path, allow_pickle=False)

    # --- Validate required keys ---
    required_keys = [
        "x_lhand", "x_rhand", "x_obj",
        "j_lhand", "j_rhand",
        "is_lhand", "is_rhand",
        "T_valid", "T_padded",
        "text", "obj_name", "dataset",
    ]
    for k in required_keys:
        if k not in raw:
            raise KeyError(
                f"_convert_single: missing key {k!r} in {raw_path!r}. "
                f"Present keys: {list(raw.keys())}"
            )

    # --- Scalar extraction ---
    T_valid = int(raw["T_valid"])
    T_padded = int(raw["T_padded"])
    is_lhand = bool(raw["is_lhand"])
    is_rhand = bool(raw["is_rhand"])
    text = str(raw["text"])
    obj_name_raw = str(raw["obj_name"])
    raw_dataset = str(raw["dataset"])

    if raw_dataset.lower() != dataset.lower():
        raise ValueError(
            f"_convert_single: raw file dataset={raw_dataset!r} != "
            f"expected dataset={dataset!r} at {raw_path!r}"
        )

    if not is_lhand and not is_rhand:
        raise ValueError(
            f"_convert_single: both is_lhand=False and is_rhand=False for {raw_path!r}. "
            "Text2HOI should never produce a sample with no active hand."
        )

    # --- Shape checks ---
    x_lhand = raw["x_lhand"].astype(np.float32)  # (T, 99)
    x_rhand = raw["x_rhand"].astype(np.float32)  # (T, 99)
    x_obj = raw["x_obj"].astype(np.float32)       # (T, 9) or (T, 10)
    j_lhand = raw["j_lhand"].astype(np.float32)   # (T, 21, 3)
    j_rhand = raw["j_rhand"].astype(np.float32)   # (T, 21, 3)

    T = x_lhand.shape[0]
    if T != T_padded:
        raise ValueError(
            f"_convert_single: x_lhand.shape[0]={T} != T_padded={T_padded} in {raw_path!r}"
        )
    if x_lhand.shape != (T, 99):
        raise ValueError(f"x_lhand expected ({T}, 99), got {x_lhand.shape}")
    if x_rhand.shape != (T, 99):
        raise ValueError(f"x_rhand expected ({T}, 99), got {x_rhand.shape}")
    if x_obj.shape[0] != T or x_obj.shape[1] not in (9, 10):
        raise ValueError(
            f"x_obj expected ({T}, 9) or ({T}, 10), got {x_obj.shape}"
        )
    if j_lhand.shape != (T, 21, 3):
        raise ValueError(f"j_lhand expected ({T}, 21, 3), got {j_lhand.shape}")
    if j_rhand.shape != (T, 21, 3):
        raise ValueError(f"j_rhand expected ({T}, 21, 3), got {j_rhand.shape}")

    # --- Hand joint positions (from pre-computed MANO decode) ---
    left_hand_joints = j_lhand   # (T, 21, 3) float32 world-frame meters
    right_hand_joints = j_rhand  # (T, 21, 3) float32

    # --- Hand axis-angle rotations ---
    # x_lhand[:, 3:99] = 16 joints * 6D rot = 96 dims
    lh_rot6d = x_lhand[:, 3:99].reshape(T, 16, 6)   # (T, 16, 6)
    rh_rot6d = x_rhand[:, 3:99].reshape(T, 16, 6)   # (T, 16, 6)
    left_hand_rot = rot6d_to_axis_angle(lh_rot6d)    # (T, 16, 3) float32
    right_hand_rot = rot6d_to_axis_angle(rh_rot6d)   # (T, 16, 3) float32

    # --- Object pose ---
    obj_translation = x_obj[:, 0:3].astype(np.float32)     # (T, 3)
    obj_rot6d_flat = x_obj[:, 3:9]                          # (T, 6)
    obj_rotation = rot6d_to_rotmat(obj_rot6d_flat)           # (T, 3, 3) float32

    # --- Valid mask ---
    valid_mask = (np.arange(T_padded) < T_valid).astype(np.bool_)  # (T,) bool

    # --- Object name translation ---
    if dataset == "grab":
        obj_name = _translate_obj_name_grab(obj_name_raw)
    else:
        # ARCTIC: obj_name is already canonical (e.g. "box", "ketchup")
        obj_name = obj_name_raw

    # --- Build CanonicalSample ---
    # /propagate is_lhand/is_rhand onto the in-memory sample so
    # downstream per-hand gating (compute_physical_realism_*,
    # canonical_to_text2hoi_input) sees the correct flags even when the
    # sample bypasses the NPZ save/load round-trip.
    sample = CanonicalSample(
        left_hand_joints=left_hand_joints,
        right_hand_joints=right_hand_joints,
        left_hand_rot=left_hand_rot,
        right_hand_rot=right_hand_rot,
        obj_translation=obj_translation,
        obj_rotation=obj_rotation,
        obj_name=obj_name,
        valid_mask=valid_mask,
        fps=30,
        text=text,
        is_lhand=is_lhand,
        is_rhand=is_rhand,
    )

    # --- Pre-save validation (strict=True: checks orthonormality + det>0) ---
    validate_canonical_sample(sample, strict=True)

    return sample, x_obj  # return raw x_obj for potential ARCTIC articulation extraction


# ---------------------------------------------------------------------------
# Main conversion loop
# ---------------------------------------------------------------------------

def convert(
    raw_dir: str,
    out_dir: str,
    dataset: str,
    max_samples: Optional[int] = None,
) -> None:
    """Convert all raw Text2HOI .npz files in raw_dir to canonical form.

    Args:
        raw_dir:     Directory containing raw .npz files (produced by text2hoi_sampler.py).
        out_dir:     Output directory. Canonical files written to <out_dir>/samples/NNNNNN.npz.
        dataset:     "grab" or "arctic".
        max_samples: Optional cap on the number of samples to convert.

    Raises:
        FileNotFoundError: If raw_dir does not exist or contains no .npz files.
        ValueError:        If dataset is not "grab" or "arctic".
    """
    if dataset not in ("grab", "arctic"):
        raise ValueError(f"convert: dataset must be 'grab' or 'arctic', got {dataset!r}")

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"convert: raw_dir not found: {raw_dir!r}")

    # Collect raw .npz files, sorted for deterministic ordering.
    raw_files = sorted(
        f for f in os.listdir(raw_dir)
        if f.endswith(".npz") and os.path.isfile(os.path.join(raw_dir, f))
    )
    if not raw_files:
        raise FileNotFoundError(
            f"convert: no .npz files found in raw_dir: {raw_dir!r}"
        )

    if max_samples is not None:
        raw_files = raw_files[:max_samples]

    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    manifest: dict = {}   # stem -> sha256
    n_converted = 0
    T_padded_default = None  # inferred from first sample

    for i, raw_fname in enumerate(raw_files):
        raw_path = os.path.join(raw_dir, raw_fname)
        out_stem = f"{i:06d}"
        out_path = os.path.join(samples_dir, f"{out_stem}.npz")

        logger.info("convert: [%d/%d] %s -> %s", i + 1, len(raw_files), raw_fname, out_stem)

        sample, raw_x_obj = _convert_single(raw_path, dataset)

        # Track T_padded from first sample.
        if T_padded_default is None:
            T_padded_default = int(sample.left_hand_joints.shape[0])

        # Build extra kwargs for ARCTIC articulation.
        extra_kwargs: dict = {}
        if dataset == "arctic" and raw_x_obj.shape[1] == 10:
            # x_obj[:, 9] = articulation angle (radians)
            obj_articulation_angle = raw_x_obj[:, 9].astype(np.float32)  # (T,) float32
            extra_kwargs["obj_articulation_angle"] = obj_articulation_angle

        # Also store is_lhand / is_rhand for downstream sanity checks.
        raw = np.load(raw_path, allow_pickle=False)
        extra_kwargs["is_lhand"] = np.array(bool(raw["is_lhand"]))
        extra_kwargs["is_rhand"] = np.array(bool(raw["is_rhand"]))

        # Write canonical .npz (save_canonical_npz already validates).
        # We save canonical fields via save_canonical_npz, then re-save with extras.
        # To include extra keys, we write manually via np.savez_compressed.
        _save_canonical_with_extras(sample, out_path, extra_kwargs, strict=True)

        # Record SHA256 for manifest.
        manifest[f"samples/{out_stem}.npz"] = _sha256_file(out_path)
        n_converted += 1

    # --- Write manifest ---
    manifest_path = os.path.join(out_dir, "MANIFEST.sha256")
    with open(manifest_path, "w") as f:
        for stem, digest in sorted(manifest.items()):
            f.write(f"{digest}  {stem}\n")
    logger.info("convert: wrote MANIFEST.sha256 (%d entries)", len(manifest))

    # --- Write config.json ---
    config = {
        "adapter_version": 1,
        "source_raw_dir": os.path.abspath(raw_dir),
        "dataset": dataset,
        "n_samples_converted": n_converted,
        "T_padded_default": T_padded_default,
        "commit_sha": _git_sha(),
        "timestamp": _utc_timestamp(),
    }
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("convert: wrote config.json")

    # --- Copy prompts.json if present ---
    src_prompts = os.path.join(raw_dir, "prompts.json")
    if os.path.isfile(src_prompts):
        import shutil
        shutil.copy2(src_prompts, os.path.join(out_dir, "prompts.json"))
        logger.info("convert: copied prompts.json")

    # --- Write README.md (provenance + non-removal flag) ---
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            f"# Canonical Samples — Text2HOI {dataset.upper()}\n\n"
            "**DO NOT DELETE THIS DIRECTORY.** These canonical .npz files are the\n"
            "scored output for baseline reproduction. Deleting them requires re-running\n"
            "the full Text2HOI inference pipeline.\n\n"
            "## Provenance\n\n"
            f"- Source: `{os.path.abspath(raw_dir)}`\n"
            f"- Dataset: `{dataset}`\n"
            f"- Adapter version: 1\n"
            f"- Commit: `{config['commit_sha']}`\n"
            f"- Timestamp: `{config['timestamp']}`\n"
            f"- Samples: {n_converted}\n\n"
            "See `config.json` for full provenance and `MANIFEST.sha256` for file hashes.\n"
        )
    logger.info("convert: wrote README.md")

    print(
        f"[text2hoi_to_canonical] DONE: {n_converted} samples -> {os.path.abspath(samples_dir)}"
    )


def _save_canonical_with_extras(
    sample: CanonicalSample,
    path: str,
    extra_kwargs: dict,
    strict: bool = True,
) -> None:
    """Validate and save CanonicalSample + extra keys to a compressed .npz.

    Core canonical fields are written using the standard schema. Extra keys
    (e.g. obj_articulation_angle, is_lhand, is_rhand) are appended; they are
    NOT part of the canonical schema and are ignored by load_canonical_npz.

    Args:
        sample:       Validated CanonicalSample.
        path:         Destination file path.
        extra_kwargs: Dict of additional arrays to store alongside canonical fields.
        strict:       If True, re-validates orthonormality/det>0 of obj_rotation.
    """
    validate_canonical_sample(sample, strict=strict)

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)

    fps = int(sample.fps)
    np.savez_compressed(
        path,
        left_hand_joints=sample.left_hand_joints,    # (T, 21, 3) float32
        right_hand_joints=sample.right_hand_joints,  # (T, 21, 3) float32
        left_hand_rot=sample.left_hand_rot,          # (T, 16, 3) float32
        right_hand_rot=sample.right_hand_rot,        # (T, 16, 3) float32
        obj_translation=sample.obj_translation,      # (T, 3)     float32
        obj_rotation=sample.obj_rotation,            # (T, 3, 3)  float32
        obj_name=np.array(sample.obj_name),
        valid_mask=sample.valid_mask,                # (T,)       bool
        fps=np.array(fps),
        text=np.array(sample.text),
        **extra_kwargs,
    )


def _utc_timestamp() -> str:
    """Return current UTC time in ISO 8601 format."""
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m umm.eval.handobject.baseline_adapters.text2hoi_to_canonical",
        description=(
            "Convert Text2HOI raw .npz samples to canonical schema. "
            "Writes <out-dir>/samples/NNNNNN.npz + MANIFEST.sha256 + config.json + README.md."
        ),
    )
    p.add_argument(
        "--raw-dir",
        required=True,
        help="Directory of raw .npz files produced by text2hoi_sampler.py.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory. Created if missing.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["grab", "arctic"],
        help="Dataset name: grab or arctic.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on number of raw files to convert (for debugging).",
    )
    return p


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args()
    convert(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        dataset=args.dataset,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
