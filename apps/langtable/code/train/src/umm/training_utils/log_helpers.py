"""Logging utilities for UMM training (TensorBoard, WandB, GCS sync).

Provides ``CustomLogger`` (a ``TensorBoardLogger`` subclass) that
additionally logs to Weights & Biases and optionally syncs TensorBoard
event files to Google Cloud Storage in a background thread.  Also
provides helper functions for resume logic, version numbering, source
backup, and ``logging`` module configuration.

Example:
    >>> from umm.training_utils.log_helpers import get_logger
    >>> logger = get_logger(cfg)
    >>> logger.log_image("train", "rgb", img_tensor, step, batch_idx=0)
"""

# ! 2024.8.21: add wandb logger when call tensorboared logger

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
import wandb
import os, os.path as osp
from datetime import datetime
import logging, platform
import imageio
import torch
import numpy as np
import trimesh
import cv2
import shutil
from pytorch_lightning.utilities import rank_zero_only
from PIL import Image
from typing_extensions import override
import threading
import time

try:
    from google.cloud import storage
except ImportError:
    storage = None
import ssl
import urllib3

# Configure SSL settings at module level
os.environ["WANDB_DISABLE_SSL"] = "true"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CustomLogger(TensorBoardLogger):
    """TensorBoard logger extended with WandB integration and optional GCS sync.

    Logs scalar metrics to both TensorBoard and Weights & Biases, saves
    images and videos to local disk (and optionally to TensorBoard/WandB),
    and in ``xcloud`` mode syncs the entire log directory to a GCS bucket
    in a background thread.

    Args:
        cfg: Experiment configuration dict-like object.
        save_dir: Root directory for saving TensorBoard logs.
        name: Experiment name (used as a subdirectory).
        version: Experiment version number.
        wandb_name_postfix: Optional postfix appended to the WandB run name.
        *args: Extra positional arguments forwarded to ``TensorBoardLogger``.
        **kwargs: Extra keyword arguments forwarded to ``TensorBoardLogger``.

    Example:
        >>> logger = CustomLogger(cfg=cfg, save_dir="./logs", name="exp", version=0)
    """

    def __init__(
        self,
        cfg=None,
        save_dir="./logs",
        name="debug",
        version=0,
        wandb_name_postfix="",
        *args,
        **kwargs,
    ):
        """Initialize CustomLogger with TensorBoard, optional WandB, and optional GCS sync.

        Args:
            cfg: Experiment config with keys like ``xcloud``, ``use_wandb``,
                ``gcs_bucket_name``, ``gcs_sync_interval``, etc.
            save_dir: Root directory for TensorBoard event files.
            name: Experiment name subdirectory.
            version: Integer version number for the run.
            wandb_name_postfix: Optional string appended to WandB run name.
            *args: Forwarded to ``TensorBoardLogger.__init__``.
            **kwargs: Forwarded to ``TensorBoardLogger.__init__``.
        """
        # this is prior to setup logger, so use print, the log will be caught by xcloud

        self.tmp_flag = False
        # if xcloud save all the tensorboard locally to a tmp folder, then periodically sync to the gcs
        self.xcloud = cfg.get("xcloud", False)
        # When wandb is active, skip TB media (add_video costs ~2.4s/call at 720p).
        # Can override with cfg.add_media_to_tb = True if TB media is needed.
        _use_wandb = cfg.get("use_wandb", True) and not self.xcloud
        self.add_media_to_tb = cfg.get("add_media_to_tb", not _use_wandb) and not self.xcloud
        print(f"CustomLogger: add_media_to_tb={self.add_media_to_tb}")

        if self.xcloud:
            self.gcs_bucket_name = cfg.get("gcs_bucket_name", "")
            self.gcs_sync_interval = cfg.get("gcs_sync_interval", 120)
            print(f"CustomLogger: xcloud mode, sync interval={self.gcs_sync_interval}")
            check_prefix = f"/gcs/{self.gcs_bucket_name}/"
            assert save_dir.startswith(check_prefix)

            self.gcs_log_dir = osp.join(
                save_dir[len(check_prefix) :], name, f"version_{version}"
            )

            # ! switch to tmp dir
            save_dir = osp.join(os.getcwd(), f"./tmp/local_tb_logs")
            print(
                f"CustomLogger: xcloud mode, save all logs  to {save_dir}, and sync to {self.gcs_log_dir}"
            )

            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.gcs_bucket_name)
            self.uploaded_files = {}

        super().__init__(save_dir=save_dir, name=name, version=version, *args, **kwargs)
        self.image_save_dir = osp.join(self.log_dir, "images")
        self.video_save_dir = osp.join(self.log_dir, "videos")
        self.organize_subfolder_interval = cfg.get(
            "log_organize_subfolder_interval", 5000
        )
        # ! also include the wandb logger
        self.use_wandb = cfg.get("use_wandb", True) and not self.xcloud
        if self.use_wandb:
            print("Using wandb logger")
            self.wandb_dir = osp.join(self.log_dir, "wandb")
            os.makedirs(self.wandb_dir, exist_ok=True)
            node_name = platform.node()
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # self.wandb_logger = WandbLogger(
            #     name=f"v{cfg.__version_id__} "
            #     + cfg.expname
            #     + f" {node_name} {current_time}",
            #     version=f"{cfg.__version_id__}_{node_name}_{current_time}",
            #     project=cfg.expname,
            #     save_dir=self.wandb_dir,
            #     tags=[node_name],
            # )
            wandb_name_postfix = str(wandb_name_postfix)
            self.wandb_logger = WandbLogger(
                name=f"v{cfg.__version_id__}; " + f"{node_name}; " + wandb_name_postfix,
                version=f"{cfg.__version_id__}_{node_name}_{current_time}"
                + wandb_name_postfix,
                project=cfg.expname,
                save_dir=self.wandb_dir,
                tags=[node_name],
            )

        if self.xcloud:
            print("CustomLogger: xcloud mode, start the GCS log upload thread")
            # Must sync the remote checkpoints/last.ckpt to local dir if remote exists
            remote_ckpt_path = osp.join(
                self.gcs_log_dir.lstrip("/"), "checkpoints", "last.ckpt"
            )
            local_ckpt_dir = osp.join(self.log_dir, "checkpoints")
            local_ckpt_path = osp.join(local_ckpt_dir, "last.ckpt")
            # Get the blob from GCS
            blob = self.bucket.blob(remote_ckpt_path)
            if blob.exists():
                print(
                    f"Found remote checkpoint at gs://{self.bucket.name}/{remote_ckpt_path}, downloading to {local_ckpt_path}"
                )
                # Create local checkpoints directory if it doesn't exist
                os.makedirs(local_ckpt_dir, exist_ok=True)
                # Download the checkpoint
                blob.download_to_filename(local_ckpt_path)
                print(f"Downloaded remote checkpoint to {local_ckpt_path}")
                print(f"ls {local_ckpt_dir}: {os.listdir(local_ckpt_dir)}")
            else:
                print(
                    f"No remote checkpoint found at gs://{self.bucket.name}/{remote_ckpt_path}"
                )
            self._start_upload_thread()

        return

    @rank_zero_only
    def _start_upload_thread(self):
        """Start a daemon thread that periodically uploads logs to GCS.

        Example:
            >>> logger._start_upload_thread()  # called internally during __init__
        """
        print("Starting GCS log upload thread")
        self._stop_event = threading.Event()
        self._upload_thread = threading.Thread(target=self._upload_logs)
        self._upload_thread.start()

    @rank_zero_only
    def _upload_logs(self):
        """Background loop that calls ``_upload_logs_to_gcs`` at regular intervals.

        Runs until ``_stop_event`` is set. Errors are logged but do not
        terminate the thread.

        Example:
            >>> # Runs as target of the background upload thread.
        """
        while not self._stop_event.is_set():
            try:
                self._upload_logs_to_gcs()
            except Exception as e:
                logging.error(f"CRITICAL LOGGER: Error uploading logs to GCS: {e}")
                logging.error("Retrying later, skip this error")
            time.sleep(self.gcs_sync_interval)  # Adjust the interval as needed

    @rank_zero_only
    def _upload_logs_to_gcs(self):
        """Walk the local log directory and upload changed files to GCS.

        Tracks file modification times in ``self.uploaded_files`` to skip
        files that have not changed since the last upload.

        Example:
            >>> logger._upload_logs_to_gcs()
        """
        for root, _, files in os.walk(self.log_dir):
            for filename in files:
                local_path = osp.join(root, filename)
                # Get the last modification time of the file
                mtime = os.path.getmtime(local_path)
                # Check if the file has been uploaded before and if it has changed
                if local_path in self.uploaded_files:
                    if self.uploaded_files[local_path] == mtime:
                        continue  # File hasn't changed; skip upload
                # Construct the GCS path
                relative_path = os.path.relpath(local_path, self.log_dir)
                gcs_path = osp.join(self.gcs_log_dir, relative_path)
                # Upload the file to GCS
                try:
                    blob = self.bucket.blob(gcs_path)
                    blob.upload_from_filename(local_path)
                    self.uploaded_files[local_path] = (
                        mtime  # Update the modification time
                    )
                    logging.info(
                        f"Uploaded {local_path} to gs://{self.bucket.name}/{gcs_path}"
                    )
                except Exception as e:
                    logging.error(f"Error uploading {local_path} to GCS: {e}")

    def finalize(self, status):
        """Finalize the logger, flushing any pending GCS uploads.

        Args:
            status: Training status string (unused directly, required by
                Lightning interface).

        Example:
            >>> logger.finalize("success")
        """
        if self.xcloud:
            self._stop_event.set()
            self._upload_thread.join()
            self._upload_logs_to_gcs()  # Final upload of any remaining logs

    # @rank_zero_only
    def log_metrics(self, metrics, step=None):
        """Log scalar metrics to TensorBoard and (optionally) WandB.

        Args:
            metrics: Dictionary of metric name to scalar value.
            step: Global step number for the x-axis.

        Example:
            >>> logger.log_metrics({"loss": 0.5, "accuracy": 0.9}, step=100)
        """
        # TensorBoard rejects plain int 0 ("you tried to log 0").
        # Convert all values to float to avoid this PL/TB limitation.
        metrics = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
        super().log_metrics(metrics, step)
        if self.use_wandb:
            try:
                # Try to log with SSL disabled
                old_ssl_context = ssl._create_default_https_context
                ssl._create_default_https_context = ssl._create_unverified_context

                self.wandb_logger.log_metrics(metrics, step)

                # Restore SSL context
                ssl._create_default_https_context = old_ssl_context
            except Exception as e:
                # Log the error but don't crash
                print(f"Warning: Failed to log to wandb: {e}")
                # Continue with other loggers if any

    def identify_save_dir_name(self, global_step):
        """Compute the subfolder name for organizing saved media by step range.

        Args:
            global_step: The current global training step.

        Returns:
            String of the form ``"{lower}_{upper}"`` where ``lower`` and
            ``upper`` are the bounding step-range multiples.

        Example:
            >>> logger.identify_save_dir_name(12345)
            '10000_15000'
        """
        lower = int(np.floor(global_step / self.organize_subfolder_interval))
        upper = int(np.ceil(global_step / self.organize_subfolder_interval))
        if upper == lower:
            upper += 1
        lower *= self.organize_subfolder_interval
        upper *= self.organize_subfolder_interval
        return f"{lower}_{upper}"

    @rank_zero_only
    def log_image(
        self, phase, key, img, global_step, batch_idx, sample_idx=0, ext="jpg"
    ):
        """Log an image to TensorBoard, WandB, and local disk.

        Args:
            phase: Training phase string (e.g. ``"train"``, ``"val"``).
            key: Descriptive key for the image (e.g. ``"rgb_obs"``).
            img: Image tensor of shape (C, H, W) or (H, W), values in [0, 1].
            global_step: Current global training step.
            batch_idx: Batch index within the current phase loop.
            sample_idx: Index of the sample within the batch.
            ext: File extension for the saved image (default ``"jpg"``).

        Example:
            >>> logger.log_image("train", "rgb", img_chw, 1000, 0)
        """
        # ! warning, the batch_idx is used for non-train phase
        # because the global_step in the pl_model only count the training steps, so for non-train step, use batch_idx to differentiate samples inside the loop, here the step servers more like the epoch for non-train phases
        # sample idx is if viz multiple samples in the minibatch
        # ! warning, the TB logger won't distinguish the sample_idx
        # * both log for tb and local file

        key = f"{phase}_image/{key}"
        if sample_idx > 0:
            key = f"{key}_s{sample_idx}"
        # if batch_idx > 0:
        #     key = f"{key}_b{batch_idx}"

        assert img.ndim == 3 or img.ndim == 2, "the shape should be 3HW, 1HW, or HW"
        img = torch.as_tensor(img)
        # if phase != "train":
        #     # assert phase in key, "the phase should be in the key"
        #     # ! assume the non-train global-step interval is larger than the length of the  non-train dataset
        #     _log_step = global_step + batch_idx
        # else:
        _log_step = global_step
        if self.add_media_to_tb:
            self.experiment.add_image(key, img.detach(), _log_step)
        if self.use_wandb:
            # ! set the maximum size of the image, the hight no larger than 2048
            wandb_img = img.detach().permute(1, 2, 0).cpu().numpy()
            if wandb_img.shape[0] > 1024:
                new_h, new_w = 1024, int(1024 / wandb_img.shape[0] * wandb_img.shape[1])
                wandb_img = cv2.resize(wandb_img.copy(), (new_w, new_h))
            wandb_img = wandb.Image(wandb_img)
            self.wandb_logger.experiment.log({key: [wandb_img]}, step=_log_step)

        save_dir = osp.join(
            self.image_save_dir, self.identify_save_dir_name(global_step)
        )
        os.makedirs(save_dir, exist_ok=True)

        save_key = key.replace("/", "_")
        save_fn = osp.join(
            save_dir,
            f"g{global_step}_{phase}_b{batch_idx}_s{sample_idx}_{save_key}.{ext}",
        )

        if isinstance(img, torch.Tensor):
            save_img = img.detach().cpu().numpy()
        else:
            save_img = img
        save_img = save_img.transpose(1, 2, 0)
        save_img = (save_img * 255).astype("uint8")
        imageio.imsave(save_fn, save_img)

        return

    @rank_zero_only
    def log_video(
        self,
        phase,
        key,
        video,
        global_step,
        batch_idx,
        sample_idx=0,
        ext=".mp4",
        caption=None,
    ):
        """Log a video to TensorBoard, WandB, and local disk.

        Args:
            phase: Training phase string (e.g. ``"train"``, ``"val"``).
            key: Descriptive key for the video (e.g. ``"rollout"``).
            video: Video tensor of shape (T, 3, H, W), values in [0, 1].
            global_step: Current global training step.
            batch_idx: Batch index within the current phase loop.
            sample_idx: Index of the sample within the batch.
            ext: Comma-separated file extensions (e.g. ``".mp4,.gif"``).
            caption: Optional caption string for the WandB video entry.

        Example:
            >>> logger.log_video("val", "rollout", vid_t3hw, 5000, 0)
        """
        # * both log for tb and local file
        start_time = time.time()
        key = f"{phase}_video/{key}"
        if sample_idx > 0:
            key = f"{key}_s{sample_idx}"
        # if batch_idx > 0:
        #     key = f"{key}_b{batch_idx}"

        assert video.ndim == 4 and video.shape[1] == 3, "the shape should be T3HW"
        # if phase != "train":
        #     # assert phase in key, "the phase should be in the key"
        #     _log_step = global_step + batch_idx
        # else:
        _log_step = global_step
        if self.add_media_to_tb:
            self.experiment.add_video(key, video[None], _log_step)
        if self.use_wandb:
            if isinstance(video, torch.Tensor):
                video = video.detach().cpu().numpy()
            if video.dtype != np.uint8:
                wandb_vid = wandb.Video(
                    (video * 255).astype("uint8"), fps=10, format="mp4", caption=caption
                )
            else:
                wandb_vid = wandb.Video(video, fps=10, format="mp4", caption=caption)
            self.wandb_logger.experiment.log({key: wandb_vid}, step=_log_step)

        save_dir = osp.join(
            self.video_save_dir, self.identify_save_dir_name(global_step)
        )
        if isinstance(video, torch.Tensor):
            video = video.detach().cpu().numpy()
        video = video.transpose(0, 2, 3, 1)  # TCHW -> TWHC
        video = (video * 255).astype("uint8")

        os.makedirs(save_dir, exist_ok=True)
        save_key = key.replace("/", "_")
        for _ext in ext.split(","):
            save_fn = osp.join(
                save_dir,
                f"g{global_step}_{phase}_b{batch_idx}_s{sample_idx}_{save_key}{_ext}",
            )
            if _ext == ".gif":
                imageio.mimwrite(save_fn, video, loop=1000)
            else:
                imageio.mimwrite(save_fn, video)

        logging.info(f"Log video takes {time.time() - start_time:.2f} seconds")
        return

    def _setup_wandb_with_ssl_bypass(self):
        """Setup wandb with SSL bypass for cluster environments"""
        try:
            # Store original SSL context
            self._original_ssl_context = ssl._create_default_https_context
            # Set unverified context
            ssl._create_default_https_context = ssl._create_unverified_context

            # Try to initialize wandb
            if hasattr(self, "wandb_logger"):
                print("WandB logger configured with SSL bypass")
        except Exception as e:
            print(f"Warning: Failed to setup wandb SSL bypass: {e}")


def get_logger(cfg):
    """Create a ``CustomLogger``, configure Python logging, and back up sources.

    Resolves resume/version logic via ``config_resume``, instantiates the
    ``CustomLogger``, sets up console + file logging handlers, and copies
    project source files into the log directory for reproducibility.

    Args:
        cfg: Experiment ``Config`` object (must have ``expname``,
            ``precision``, ``debug_logging``, and optionally ``resume``).

    Returns:
        A fully initialised ``CustomLogger`` instance.

    Example:
        >>> logger = get_logger(cfg)
        >>> print(logger.log_dir)
    """
    cfg = config_resume(cfg)
    # get logger
    logger = CustomLogger(
        cfg=cfg,
        save_dir=cfg.get("log_dir", "./logs"),
        name=cfg.expname,
        version=cfg.__version_id__,
        wandb_name_postfix=cfg.get("wandb_name_postfix", ""),
    )
    print(f"Logger log_dir: {logger.log_dir}")
    os.makedirs(logger.log_dir, exist_ok=True)

    # config logging
    time_stamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    setup_logging(
        version=cfg.__version_id__,
        name=cfg.expname,
        precision=cfg.precision,
        log_file_path=osp.join(logger.log_dir, f"{time_stamp}.log"),
        level=logging.DEBUG if cfg.debug_logging else logging.INFO,
        # use_file=not getattr(cfg, "xcloud", False),
        use_file=True,  # ! when use xcloud, will sync the local log dir to gcs periodically in another thread, so fine.
    )

    # backup the source file
    backup_source_files(logger.log_dir)

    return logger


def backup_source_files(log_dir, filter_exts=[".py", ".yaml", ".yml", ".json", ".txt"]):
    """Copy project source files into the log directory for reproducibility.

    Recursively copies files matching ``filter_exts`` from a set of
    predefined project subdirectories (``train.py``, ``src/``, ``eval/``,
    ``configs/``, ``utils/``) into ``<log_dir>/src_bck/``.

    Args:
        log_dir: Path to the experiment log directory.
        filter_exts: List of file extensions to include in the backup.

    Example:
        >>> backup_source_files("/logs/exp/version_0")
    """
    backup_dir = osp.join(log_dir, "src_bck")
    os.makedirs(backup_dir, exist_ok=True)
    src_fn_or_dirs = ["train.py", "src", "eval", "configs", "utils"]
    proj_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))

    for src in src_fn_or_dirs:
        src_path = osp.join(proj_root, src)

        # Skip if source doesn't exist
        if not osp.exists(src_path):
            continue

        # If it's a file, copy it if it matches the filter
        if osp.isfile(src_path):
            if any(src_path.endswith(ext) for ext in filter_exts):
                dest_path = osp.join(backup_dir, src)
                os.makedirs(osp.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)

        # If it's a directory, recursively copy filtered files
        elif osp.isdir(src_path):
            dest_base = osp.join(backup_dir, src)
            for root, dirs, files in os.walk(src_path):
                # Calculate relative path from src_path to root
                rel_path = osp.relpath(root, src_path)
                dest_dir = (
                    osp.join(dest_base, rel_path) if rel_path != "." else dest_base
                )

                # Copy files that match the filter
                for file in files:
                    if any(file.endswith(ext) for ext in filter_exts):
                        os.makedirs(dest_dir, exist_ok=True)
                        src_file = osp.join(root, file)
                        dest_file = osp.join(dest_dir, file)
                        shutil.copy2(src_file, dest_file)

    return


def config_resume(cfg):
    """Resolve resume logic and assign a version ID to the config.

    Interprets ``cfg.resume`` as either a version number or a checkpoint
    path.  Sets ``cfg.__version_id__`` and ``cfg.__init_ckpt_path__``
    accordingly.  If ``cfg.resume`` is ``None``, the next available
    version number is auto-assigned.

    Args:
        cfg: Experiment ``Config`` object (must have ``resume``,
            ``expname``, and optionally ``log_dir``).

    Returns:
        The same ``cfg`` object with ``__version_id__`` and
        ``__init_ckpt_path__`` populated.

    Raises:
        ValueError: If the resume argument is neither a valid version
            number nor an existing checkpoint path.

    Example:
        >>> cfg = config_resume(cfg)
        >>> print(cfg.__version_id__, cfg.__init_ckpt_path__)
    """
    # * manually control the version here
    cfg.__init_ckpt_path__ = None
    version_id = None
    if cfg.resume is not None:
        try:  # 1.) try to identify the version number
            version_id = int(cfg.resume)
            assert version_id >= 0
            ckpt_path = osp.join(
                "./logs",
                cfg.expname,
                f"version_{version_id}",
                "checkpoints",
                "last.ckpt",
            )
            if osp.exists(ckpt_path):  # if has the last.ckpt, resume from that
                cfg.__init_ckpt_path__ = ckpt_path
            # otherwise, does not set the init ckpt path, use the version number but start from scratch
        except (ValueError, AssertionError) as e:
            # if provide a valid path of ckpt, identify the version number from the path
            if osp.exists(cfg.resume):
                cfg.__init_ckpt_path__ = cfg.resume
                version_id = int(
                    osp.basename(osp.dirname(osp.dirname(cfg.resume))).split(
                        "version_"
                    )[-1]
                )
            else:
                raise ValueError(
                    f"Cannot find the ckpt path {cfg.resume} and it's not a valid version number"
                )
    if version_id is None:
        version_id: int = find_next_version(
            osp.join(cfg.get("log_dir", "./logs"), cfg.expname)
        )
    cfg.__version_id__ = version_id
    print(f"Config version_id: {version_id}")
    return cfg


def find_next_version(dir):
    """Find the smallest unused version number in a log directory.

    Scans ``dir`` for subdirectories named ``version_<N>`` and returns
    the smallest non-negative integer *N* that is not yet taken.

    Args:
        dir: Path to the parent log directory containing ``version_*``
            subdirectories.

    Returns:
        The next available integer version number (starting from 0).

    Example:
        >>> find_next_version("./logs/my_experiment")
        3
    """
    try:
        subs = os.listdir(dir)
    except (FileNotFoundError, OSError) as e:
        return 0  # start from 0
    v_lists = []
    for d in subs:
        if not d.startswith("version_"):
            continue
        path = osp.join(dir, d)
        if not osp.isdir(path):
            continue
        _v = int(d.split("version_")[1])
        v_lists.append(_v)
    # now find the smallest one that does not fit
    version = 0
    while True:
        if version not in v_lists:
            return version
        version += 1


class HostnameFilter(logging.Filter):
    """Logging filter that injects the machine hostname into log records.

    Example:
        >>> handler = logging.StreamHandler()
        >>> handler.addFilter(HostnameFilter())
    """

    hostname = platform.node()

    def filter(self, record):
        """Add ``hostname`` attribute to the log record.

        Args:
            record: The ``logging.LogRecord`` to augment.

        Returns:
            Always ``True`` (the record is never suppressed).

        Example:
            >>> record = logging.LogRecord(...)
            >>> HostnameFilter().filter(record)
            True
        """
        record.hostname = HostnameFilter.hostname
        return True


def setup_logging(
    version, name, precision, log_file_path, level=logging.INFO, use_file=True
):
    """Configure the root Python logger with console and optional file handlers.

    Sets up a structured format including experiment version, name, precision,
    hostname, log level, and timestamp.  Verbose third-party loggers are
    clamped to WARNING.

    Args:
        version: Experiment version id (embedded in the log prefix).
        name: Experiment name (embedded in the log prefix).
        precision: Float precision (16 or 32, embedded in the log prefix).
        log_file_path: Absolute path to the log file.
        level: Logging level for the root logger. Defaults to logging.INFO.
        use_file: Whether to add a FileHandler. Defaults to True.

    Example:
        >>> setup_logging(0, "exp", 32, "/tmp/train.log")
    """
    os.makedirs(osp.dirname(log_file_path), exist_ok=True)
    logging.getLogger().setLevel(level)

    # List of logger names to keep at WARNING level even in debug mode
    # to avoid overwhelming verbose output from third-party libraries
    verbose_loggers = [
        "torch.autograd.graph",
        "torch",
        "PIL",
        "matplotlib",
        "urllib3",
        "boto3",
        "botocore",
        "websockets",
    ]

    for ln in logging.root.manager.loggerDict:
        lgr = logging.getLogger(ln)
        # Keep verbose third-party loggers at WARNING level
        if any(ln.startswith(vl) for vl in verbose_loggers):
            lgr.setLevel(logging.WARNING)
        else:
            lgr.setLevel(level)

    console_formatter = logging.Formatter(
        f"|(v={version},P{precision}){name}|%(hostname)s|%(levelname)s|%(asctime)s| %(message)s   (%(filename)s:%(lineno)d)",
        "%b-%d-%H:%M:%S",
    )  # pathname
    file_formatter = logging.Formatter(
        f"|(v={version},P{precision}){name}|%(hostname)s|%(levelname)s|%(asctime)s| %(message)s   (%(pathname)s:%(lineno)d)",
        "%b-%d-%H:%M:%S",
    )  # pathname

    if use_file:
        file_handler = logging.FileHandler(
            filename=log_file_path, mode="w+", encoding="utf-8"
        )
        file_handler.addFilter(HostnameFilter())
        file_handler.setFormatter(file_formatter)

    # # lightning logger
    # logger = logging.getLogger("pytorch_lightning")
    # logger.handlers[0].addFilter(HostnameFilter())
    # logger.handlers[0].setFormatter(console_formatter)
    # if use_file:
    #     logger.addHandler(file_handler)

    # remove all other handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()  # root
    console_handler = logging.StreamHandler()
    console_handler.addFilter(HostnameFilter())
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if use_file:
        logger.addHandler(file_handler)
    return
