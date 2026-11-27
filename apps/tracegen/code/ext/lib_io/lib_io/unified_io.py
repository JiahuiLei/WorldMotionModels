import os
import os.path as osp
import numpy as np
import zipfile
import json
from io import BytesIO
from PIL import Image
import imageio.v2 as imageio
from pycocotools import mask as mask_utils
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import List, Union, Dict, Any, Optional
import time
import signal
from functools import wraps

try:
    from turbojpeg import TurboJPEG
except ImportError:
    TurboJPEG = None


def prevent_keyboard_interrupt(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        interrupted = False
        original_handler = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            nonlocal interrupted
            interrupted = True
            logging.info(f"Received SIGINT. Will stop after {func.__name__} completes.")

        signal.signal(signal.SIGINT, handler)
        try:
            return func(*args, **kwargs)
        finally:
            signal.signal(signal.SIGINT, original_handler)
            if interrupted:
                raise KeyboardInterrupt

    return wrapper


def decode_uid_frames(annotation, max_workers=1, profile=False, target_uids=None):
    """Decode RLE instance segmentation annotations into dense UID frames.

    Args:
        annotation: List of per-frame annotation lists (each item is a list of
            ``{"id": int, "rle_segmentation": rle}`` dicts).
        max_workers: Number of threads for parallel decoding.
        profile: If True, print per-UID decode timing.
        target_uids: Optional set of integer UIDs to decode. When provided,
            objects whose ``id`` is not in this set are skipped entirely,
            avoiding the pycocotools RLE decode cost for unused instances.

    Returns:
        Tuple of:
          - ``uid_frames``: ``(T, H, W)`` int64 array of UID values (0 = background).
          - ``uid_cnts``: List of ``{uid: pixel_count}`` dicts, one per frame.

    Example::

        frames, cnts = decode_uid_frames(annotation, target_uids={3, 7, 12})
    """
    if not annotation:
        return np.array([]), []

    first_rle = annotation[0][0]["rle_segmentation"]
    mask_shape = mask_utils.decode(first_rle).shape

    def process_frame(frame_anns):
        frame = np.zeros(mask_shape, dtype=np.int64)
        uid_cnt = {}
        for k in frame_anns:
            uid = int(k["id"])
            if target_uids is not None and uid not in target_uids:
                continue
            if profile:
                t_last = time.time()
            mask = mask_utils.decode(k["rle_segmentation"]) > 0
            cnt = np.sum(mask)
            uid_cnt[uid] = cnt
            frame[mask] = uid
            if profile:
                t_now = time.time()
                print(
                    f"  Load Insts:  [{uid}] cnt={cnt} {(t_now - t_last)*1000.0:.4f} ms"
                )
        return frame, uid_cnt

    if max_workers == 1:
        results = [process_frame(ann) for ann in annotation]
    else:
        # Use ThreadPoolExecutor for parallel decoding
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_frame, annotation))

    uid_frames, uid_cnts = list(zip(*results))
    return np.stack(uid_frames, axis=0), list(uid_cnts)


def encode_uid_frames(seg):
    if seg.ndim == 2:
        seg = seg[np.newaxis, ...]

    annotation = []
    for t in range(seg.shape[0]):
        frame_annotation = []
        for uid in np.unique(seg[t]):
            if uid == 0:
                continue
            mask = seg[t] == uid
            rle = mask_utils.encode(np.asfortranarray(mask))
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("utf-8")
            frame_annotation.append({"id": int(uid), "rle_segmentation": rle})
        annotation.append(frame_annotation)
    return annotation


class UnifiedIO:
    def __init__(self):
        self.jpeg = TurboJPEG() if TurboJPEG else None
        if not self.jpeg:
            logging.warning("TurboJPEG not available, falling back to PIL/imageio.")

    @prevent_keyboard_interrupt
    def save_rgbs(self, rgbs: Union[np.ndarray, List[np.ndarray]], save_path: str):
        if not save_path.endswith(".zip"):
            raise ValueError("save_path must end with .zip")
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        with zipfile.ZipFile(save_path, "w") as zip_ref:
            for t, img in enumerate(rgbs):
                if self.jpeg:
                    # TurboJPEG expects BGR for encoding if input is BGR, but we assume RGB input.
                    # TurboJPEG.encode takes `pixel_format` arg. Default TJPF_BGR?
                    # PyTurboJPEG wrapper usually expects BGR/RGB based on flag or assumes BGR.
                    # Actually PyTurboJPEG encode: encode(img, quality=85, pixel_format=TJPF_BGR)
                    # We assume inputs are RGB.
                    # TJPF_RGB is 0.
                    # Let's check typical usage or just swap if needed.
                    # Assuming standard PyTurboJPEG: it handles it.
                    # For safety, standard numpy is RGB.
                    # We want to output standard JPEG (YUV).
                    try:
                        # pixel_format=0 is TJPF_RGB
                        encoded_img = self.jpeg.encode(img, pixel_format=0)
                        zip_ref.writestr(f"{t:06d}.jpg", encoded_img)
                        continue
                    except Exception as e:
                        logging.warning(
                            f"TurboJPEG encode failed: {e}. Fallback to PIL."
                        )

                buf = BytesIO()
                Image.fromarray(img).save(buf, format="JPEG")
                zip_ref.writestr(f"{t:06d}.jpg", buf.getvalue())

    def load_rgbs(
        self, path: str, indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        if not path.endswith(".zip"):
            raise ValueError("path must end with .zip")

        with zipfile.ZipFile(path, "r") as z:
            all_files = set(z.namelist())
            if indices is None:
                indices = sorted(
                    [
                        int(osp.splitext(f)[0])
                        for f in all_files
                        if f.endswith(".jpg") and osp.splitext(f)[0].isdigit()
                    ]
                )

            images, loaded_indices = [], []
            for idx in indices:
                fn = f"{idx:06d}.jpg"
                if fn not in all_files:
                    continue

                data = z.read(fn)
                if self.jpeg:
                    try:
                        img = self.jpeg.decode(data, pixel_format=0)  # TJPF_RGB
                    except Exception as e:
                        logging.debug(
                            f"TurboJPEG decode failed, falling back to imageio: {e}"
                        )
                        img = imageio.imread(BytesIO(data))
                else:
                    img = imageio.imread(BytesIO(data))

                images.append(img)
                loaded_indices.append(idx)
            return images

    @prevent_keyboard_interrupt
    def save_npzs(self, path: str, **kwargs):
        os.makedirs(osp.dirname(path), exist_ok=True)
        np.savez_compressed(path, **kwargs)

    def load_npzs(self, path: str) -> Dict[str, np.ndarray]:
        return dict(np.load(path, allow_pickle=True))

    @prevent_keyboard_interrupt
    def save_inst_segs(self, segs: Union[np.ndarray, List[np.ndarray]], path: str):
        """
        Save instance segmentation maps to a zip of JSON files.
        Args:
            segs: (T, H, W) array or List of (H, W) arrays.
            path: Path to zip file.
        """
        if not path.endswith(".zip"):
            raise ValueError("path must end with .zip")
        os.makedirs(osp.dirname(path), exist_ok=True)

        if isinstance(segs, list):
            segs = np.stack(
                segs, axis=0
            )  # encode_uid_frames handles stacking if needed but let's be explicit or just use helper

        # We can reuse encode_uid_frames which returns List[List[Dict]] (list of frames)
        annotations = encode_uid_frames(segs)

        with zipfile.ZipFile(path, "w") as zip_ref:
            for t, frame_ann in enumerate(annotations):
                json_str = json.dumps(frame_ann)
                zip_ref.writestr(f"{t:06d}.json", json_str)

        return annotations

    def list_inst_uids(
        self,
        path: str,
        index: int,
    ) -> set[int]:
        """List available UIDs in an instance seg frame without decoding masks.

        Parses only the JSON metadata (``"id"`` field) from the ZIP entry,
        skipping the expensive RLE decode step entirely.

        Args:
            path: Path to the instance segmentation ZIP file.
            index: Frame index to query.

        Returns:
            Set of integer UIDs present in the given frame.

        Example::

            uids = io.list_inst_uids("instances/seq_001_head.zip", index=15)
        """
        with zipfile.ZipFile(path, "r") as z:
            fn = f"{index:06d}.json"
            if fn not in z.namelist():
                return set()
            with z.open(fn) as f:
                annotations = json.load(f)
        return {int(ann["id"]) for ann in annotations}

    def load_inst_segs(
        self,
        path: str,
        max_workers: int = 1,
        indices: Optional[List[int]] = None,
        target_uids: Optional[set] = None,
    ) -> Union[tuple[np.ndarray, List[Dict[int, int]]], tuple[np.ndarray, list]]:
        """Load instance segmentation from a zip of JSON files.

        Args:
            path: Path to the ZIP file of per-frame JSON annotations.
            max_workers: Threads for parallel RLE decoding.
            indices: Frame indices to load. If None, loads all frames.
            target_uids: Optional set of integer UIDs to decode. Objects not in
                this set are skipped, avoiding unnecessary RLE decode work.

        Returns:
            Tuple of ``(uid_frames, uid_cnts)`` — see ``decode_uid_frames``.
        """
        if not path.endswith(".zip"):
            raise ValueError("path must end with .zip")

        with zipfile.ZipFile(path, "r") as z:
            all_files = set(z.namelist())
            if indices is None:
                indices = sorted(
                    [
                        int(osp.splitext(f)[0])
                        for f in all_files
                        if f.endswith(".json") and osp.splitext(f)[0].isdigit()
                    ]
                )

            annotation = []
            loaded_indices = []
            for idx in indices:
                fn = f"{idx:06d}.json"
                if fn not in all_files:
                    continue
                with z.open(fn) as f:
                    annotation.append(json.load(f))
                loaded_indices.append(idx)

        return decode_uid_frames(
            annotation, max_workers=max_workers, target_uids=target_uids
        )

    @prevent_keyboard_interrupt
    def save_depths(
        self, depths: Union[np.ndarray, List[np.ndarray]], path: str, dtype=np.float16
    ):
        """
        Save depth maps to a zip of stored npz files.
        Args:
            depths: (T, H, W) array or List of (H, W) arrays.
            path: Path to zip file.
            dtype: Dtype to save depth maps (default: float16).
        """
        if not path.endswith(".zip"):
            raise ValueError("path must end with .zip")
        os.makedirs(osp.dirname(path), exist_ok=True)

        with zipfile.ZipFile(path, "w") as zip_ref:
            for t, depth in enumerate(depths):
                depth = depth.astype(dtype)
                buf = BytesIO()
                np.savez_compressed(buf, depth=depth)
                zip_ref.writestr(f"{t:06d}.npz", buf.getvalue())

    def load_depths(
        self, path: str, indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Load depth maps from a zip of stored npz files.
        """
        if not path.endswith(".zip"):
            raise ValueError("path must end with .zip")

        with zipfile.ZipFile(path, "r") as z:
            all_files = set(z.namelist())
            if indices is None:
                indices = sorted(
                    [
                        int(osp.splitext(f)[0])
                        for f in all_files
                        if f.endswith(".npz") and osp.splitext(f)[0].isdigit()
                    ]
                )

            depths, loaded_indices = [], []
            for idx in indices:
                fn = f"{idx:06d}.npz"
                if fn not in all_files:
                    continue

                data = z.read(fn)
                with np.load(BytesIO(data)) as npz:
                    depths.append(npz["depth"])
                loaded_indices.append(idx)
            return depths

    @prevent_keyboard_interrupt
    def save_text(self, text: Any, path: str):
        os.makedirs(osp.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(text, f, indent=2)

    def load_text(self, path: str) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    @prevent_keyboard_interrupt
    def save_packed_zip(
        self, packed: List[Dict], packed_meta: List[Dict], zip_path: str
    ):
        """
        Save packed data and metadata into a zip file.
        Args:
            packed: List of frame dictionaries. Each frame dict contains {uid: {data_dict}}.
            packed_meta: List of frame metadata dictionaries.
            zip_path: Path to the output zip file.
        """
        if not zip_path.endswith(".zip"):
            raise ValueError("zip_path must end with .zip")
        if osp.dirname(zip_path):
            os.makedirs(osp.dirname(zip_path), exist_ok=True)

        with zipfile.ZipFile(zip_path, "w") as zip_ref:
            # Save meta.json
            zip_ref.writestr("meta.json", json.dumps(packed_meta))

            # Save each segment as npz
            for t, frame_data in enumerate(packed):
                for uid, segment_data in frame_data.items():
                    if uid == "motion_mask":
                        buf = BytesIO()
                        # Use save to avoid the save_compressed limitation for boolean, actually save_compressed is fine
                        np.savez_compressed(buf, motion_mask=segment_data)
                        zip_ref.writestr(f"{t}_mask.npz", buf.getvalue())
                        continue
                    # segment_data is a dict of numpy arrays, e.g. {'rgb': ..., 'depth': ...}
                    buf = BytesIO()
                    np.savez_compressed(buf, **segment_data)
                    zip_ref.writestr(f"{t}_{uid}.npz", buf.getvalue())

    def load_packed_meta(self, zip_path: str) -> List[Dict]:
        """
        Load packed metadata from a zip file.
        """
        if not zip_path.endswith(".zip"):
            raise ValueError("zip_path must end with .zip")

        with zipfile.ZipFile(zip_path, "r") as z:
            with z.open("meta.json") as f:
                ret = json.load(f)
        ret = [{int(k): v for k, v in it.items()} for it in ret]
        return ret

    def load_packed_segments(
        self, zip_path: str, query_list: Optional[List[tuple]] = None
    ) -> Dict[int, Dict[int, Dict[str, np.ndarray]]]:
        """
        Load packed segments from a zip file.
        Args:
            zip_path: Path to the zip file.
            query_list: Optional list of (time, uid) tuples to load.
                        If None, loads all segments.
        Returns:
            Dict[time, Dict[uid, Dict[str, np.ndarray]]]
        """
        if not zip_path.endswith(".zip"):
            raise ValueError("zip_path must end with .zip")

        results = {}

        with zipfile.ZipFile(zip_path, "r") as z:
            all_files = set(z.namelist())

            # Determine which files to load
            files_to_load = []
            if query_list is None:
                # Load all .npz files
                for f in all_files:
                    if f.endswith(".npz"):
                        base = osp.splitext(f)[0]
                        parts = base.split("_")
                        if len(parts) == 2 and parts[0].isdigit():
                            t = int(parts[0])
                            if parts[1] == "mask":
                                files_to_load.append((t, "motion_mask", f))
                            elif parts[1].isdigit():
                                uid = int(parts[1])
                                files_to_load.append((t, uid, f))
            else:
                for t, uid in query_list:
                    if uid == "motion_mask":
                        fname = f"{t}_mask.npz"
                    else:
                        fname = f"{t}_{uid}.npz"
                    if fname in all_files:
                        files_to_load.append((t, uid, fname))
                    else:
                        logging.warning(f"Segment {fname} not found in {zip_path}")

            # Load files
            for t, uid, fname in files_to_load:
                data = z.read(fname)
                with np.load(BytesIO(data)) as npz:
                    if uid == "motion_mask":
                        segment_data = npz["motion_mask"]
                    else:
                        # Convert NpzFile to dict to keep data in memory after closing
                        segment_data = dict(npz)

                if t not in results:
                    results[t] = {}
                results[t][uid] = segment_data

        return results
