# lib_io

Unified I/O library for robotics dataset storage. Handles efficient save/load of RGB frames, depth maps, instance segmentation masks, numpy arrays, and packed multi-segment data — all using zip-based file organization.

## Installation

```bash
# Base install
pip install -e ext/lib_io

# With TurboJPEG for faster JPEG encode/decode
pip install -e "ext/lib_io[turbo]"
```

## Quick Start

```python
from lib_io import UnifiedIO
import numpy as np

io = UnifiedIO()

# RGB frames (T, H, W, 3) → zip of JPEGs
rgbs = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
io.save_rgbs(rgbs, "data/rgb.zip")
loaded_rgbs = io.load_rgbs("data/rgb.zip")

# Depth maps (T, H, W) → zip of float16 NPZs
depths = [np.random.rand(480, 640).astype(np.float32) for _ in range(10)]
io.save_depths(depths, "data/depth.zip")
loaded_depths = io.load_depths("data/depth.zip")

# Instance segmentation (T, H, W) int → zip of RLE JSONs
segs = np.random.randint(0, 5, (10, 480, 640), dtype=np.int64)
io.save_inst_segs(segs, "data/seg.zip")
uid_frames, uid_counts = io.load_inst_segs("data/seg.zip")

# Arbitrary numpy data → compressed NPZ
io.save_npzs("data/states.npz", joints=np.zeros((10, 7)), ee_pos=np.zeros((10, 3)))
data = io.load_npzs("data/states.npz")  # {"joints": ..., "ee_pos": ...}

# JSON text
io.save_text({"task": "pick up cup", "success": True}, "data/meta.json")
meta = io.load_text("data/meta.json")

# Packed multi-segment data (per-frame UID segments with metadata)
packed = [{1: {"rgb": np.zeros((64, 64, 3), dtype=np.uint8)}}]
packed_meta = [{"frame": 0, "n_segs": 1}]
io.save_packed_zip(packed, packed_meta, "data/packed.zip")
meta = io.load_packed_meta("data/packed.zip")
segments = io.load_packed_segments("data/packed.zip")
```

## API Reference

### `UnifiedIO`

| Method | Description |
|--------|-------------|
| `save_rgbs(rgbs, save_path)` | Save list/array of RGB frames as JPEGs in a zip |
| `load_rgbs(path, indices=None)` | Load RGB frames from zip; optionally filter by frame indices |
| `save_depths(depths, path, dtype=float16)` | Save depth maps as float16 NPZs in a zip |
| `load_depths(path, indices=None)` | Load depth maps from zip |
| `save_inst_segs(segs, path)` | Save instance segmentation (int array) as RLE JSON in a zip |
| `load_inst_segs(path, max_workers=1, indices=None)` | Load instance segs; returns `(uid_frames, uid_counts)` |
| `save_npzs(path, **kwargs)` | Save keyword numpy arrays as a compressed NPZ file |
| `load_npzs(path)` | Load NPZ file as a `dict[str, ndarray]` |
| `save_text(text, path)` | Save any JSON-serializable object as a JSON file |
| `load_text(path)` | Load JSON file |
| `save_packed_zip(packed, packed_meta, zip_path)` | Save packed per-UID segment data + metadata into a zip |
| `load_packed_meta(zip_path)` | Load only the metadata from a packed zip |
| `load_packed_segments(zip_path, query_list=None)` | Load segments from a packed zip, optionally filtered by `(t, uid)` tuples |

### Helper Functions

| Function | Description |
|----------|-------------|
| `encode_uid_frames(seg)` | Encode `(T, H, W)` int segmentation array to RLE annotation format |
| `decode_uid_frames(annotation, max_workers=1)` | Decode RLE annotations back to `(T, H, W)` array + per-frame uid counts |
| `prevent_keyboard_interrupt(func)` | Decorator to defer `KeyboardInterrupt` until a function completes (safe saves) |

## File Format Details

| Data type | Format | Notes |
|-----------|--------|-------|
| RGB | `{t:06d}.jpg` inside zip | TurboJPEG if available, else PIL |
| Depth | `{t:06d}.npz` inside zip | float16 compressed |
| Inst. seg | `{t:06d}.json` inside zip | COCO RLE encoding via pycocotools |
| NPZ | `.npz` | `numpy.savez_compressed` |
| Packed | `{t}_{uid}.npz` + `meta.json` inside zip | per-segment numpy arrays |
