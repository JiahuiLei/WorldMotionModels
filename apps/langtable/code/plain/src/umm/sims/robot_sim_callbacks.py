"""PyTorch Lightning callbacks for robot simulation evaluation.

Provides ``RobotSimCallback`` which periodically runs a robot simulation
evaluator (e.g., ManiSkill) during training, logs scalar metrics and
rollout videos, and factory helpers to instantiate the callback from a
YAML config dict.

Example:
    >>> from umm.sims.robot_sim_callbacks import get_eval_callback
    >>> callbacks = get_eval_callback(cfg, video_root="/tmp/videos")
    >>> trainer = pl.Trainer(callbacks=callbacks)
"""

import pytorch_lightning as pl
from pytorch_lightning import Callback
import torch
from functools import partial
import logging
import os, sys, os.path as osp
from pytorch_lightning.utilities import rank_zero_only
import gc
import imageio
import numpy as np

logger = logging.getLogger(__name__)

# CURRENT_DIR variable removed - using proper package imports instead


class RobotSimCallback(Callback):
    """Lightning callback that runs robot simulation evaluations during training.

    At configurable step intervals the callback switches the model to eval
    mode, executes a rollout via the provided evaluator, logs success-rate
    metrics and a sample video, then restores training mode.

    Args:
        evaluator_maker: Callable that accepts a ``video_dir`` keyword
            argument and returns an evaluator object whose ``evaluate``
            method drives the simulation rollout.
        video_root: Directory where rollout videos are stored.
        eval_every_n_steps: Run evaluation every this many global training
            steps.
        eval_at_beginning: If ``True``, also run evaluation before the
            first training step.  Defaults to ``True``.
        safe_run: If ``True``, catch and log evaluation exceptions instead
            of crashing the training loop.  Defaults to ``True``.

    Example:
        >>> from functools import partial
        >>> maker = partial(MyEvaluator, num_envs=10)
        >>> cb = RobotSimCallback(maker, "/tmp/vids", eval_every_n_steps=500)
    """

    def __init__(
        self,
        evaluator_maker,
        video_root,
        eval_every_n_steps,
        eval_at_beginning=True,
        safe_run=True,
    ):
        """Initialize the robot simulation evaluation callback.

        Args:
            evaluator_maker: Callable accepting a ``video_dir`` keyword
                that returns an evaluator with an ``evaluate`` method.
            video_root: Directory where rollout videos are saved.
            eval_every_n_steps: Run evaluation every N global training steps.
            eval_at_beginning: If True, evaluate before the first step.
            safe_run: If True, catch evaluation errors instead of crashing.

        Example:
            >>> cb = RobotSimCallback(maker, "/tmp/vids", 500)
        """
        super().__init__()
        self.evaluator_maker = evaluator_maker

        self.eval_every_n_steps = eval_every_n_steps
        self.evaluated_steps = []
        self.video_root = video_root
        self.eval_at_beginning = eval_at_beginning
        self.safe_run = safe_run
        os.makedirs(self.video_root, exist_ok=True)

        self.evaluator = evaluator_maker(video_dir=osp.join(self.video_root))
        logging.info(f"RobotSimCallback: eval_every_n_steps={self.eval_every_n_steps}")

        self.evaluated_at_first = False
        return

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """Trigger evaluation at the start of selected training batches.

        Evaluation fires when ``trainer.global_step`` is a multiple of
        ``eval_every_n_steps`` (and ``> 0``), or on the very first batch
        if ``eval_at_beginning`` is enabled.

        Args:
            trainer: The PyTorch Lightning ``Trainer`` instance.
            pl_module: The ``LightningModule`` being trained.
            batch: The current training batch (unused).
            batch_idx: Index of the current batch (unused).
        """
        # Evaluate every n training steps
        if (
            trainer.global_step % self.eval_every_n_steps == 0
            and trainer.global_step > 0
        ) or (
            not self.evaluated_at_first and self.eval_at_beginning
        ):  # eval at beginning!!
            self.evaluated_at_first = True
            if trainer.global_step in self.evaluated_steps:
                return
            try:
                self._run_evaluation(trainer, pl_module)
                self.evaluated_steps.append(trainer.global_step)
            except Exception as e:
                logging.error(f"RobotSimCallback: evaluation failed: {e}")
                if not self.safe_run:
                    raise e
                return

    @torch.no_grad()
    @rank_zero_only
    def _run_evaluation(self, trainer, pl_module):
        """Execute a full simulation evaluation and log results.

        Switches the model to eval mode, runs the sim rollout evaluator
        (which manages per-task ManiSkill envs internally), logs scalar
        metrics under the ``SIM_EVAL/`` prefix, attempts to log a sample
        rollout video, then restores the previous training state.

        Args:
            trainer: The PyTorch Lightning ``Trainer`` instance.
            pl_module: The ``LightningModule`` being trained.
        """

        logging.info(f"RobotSimCallback: running evaluation")
        # Set model to eval mode
        is_training = pl_module.training
        pl_module.eval()
        torch.cuda.empty_cache()

        # Use EMA model if available, otherwise the base model
        if hasattr(pl_module, "ema_model") and pl_module.ema_model is not None:
            eval_model = pl_module.ema_model
        else:
            eval_model = pl_module.model

        results = self.evaluator.evaluate(eval_model)
        gc.collect()

        # Log results
        if results:
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    pl_module.log(
                        f"SIM_EVAL/{key}",
                        torch.as_tensor(value).float(),
                        on_step=True,
                        on_epoch=True,
                        prog_bar=key == "success_once",
                    )
        torch.cuda.empty_cache()
        logging.info(f"Latest results: {results}")

        # Try to log a sample rollout video from any task
        self._log_rollout_video(pl_module, trainer)

        # Set model back to train mode
        if is_training:
            pl_module.train()
        logging.info(f"RobotSimCallback: evaluation completed")
        return

    def _log_rollout_video(self, pl_module, trainer):
        """Find and log the first available rollout video."""
        try:
            for task_dir in sorted(os.listdir(self.video_root)):
                task_path = osp.join(self.video_root, task_dir)
                if not osp.isdir(task_path):
                    continue
                # ManiSkill RecordEpisode saves videos directly in task_dir
                video_fns = [
                    f for f in sorted(os.listdir(task_path))
                    if f.endswith(".mp4")
                ]
                if video_fns:
                    video_fn = video_fns[-1]
                    video = imageio.mimread(osp.join(task_path, video_fn))
                    video = np.stack(video)
                    pl_module.logger.log_video(
                        "val",
                        f"SIM_EVAL/{task_dir}_rollout",
                        video.transpose(0, 3, 1, 2) / 255.0,
                        trainer.global_step,
                        0,
                        caption=f"SIM rollout: {task_dir}",
                    )
                    return  # Log just one sample
        except Exception as e:
            logging.warning(f"Failed to log rollout video: {e}")


def get_evaluator_maker(cfg):
    """Create a partially-applied evaluator constructor from the experiment config.

    Reads the ``robot_env`` section of *cfg* to determine the simulator
    backend (currently only ``"maniskill"`` is supported) and returns a
    ``functools.partial`` object that, when called with a ``video_dir``
    keyword argument, produces the concrete evaluator instance.

    Args:
        cfg: Experiment configuration dictionary.  Must contain a
            ``robot_env`` key with at least a ``sim`` field.

    Returns:
        A ``functools.partial`` object wrapping the evaluator class.

    Raises:
        NotImplementedError: If the requested simulator type is not
            supported.

    Example:
        >>> maker = get_evaluator_maker({"robot_env": {"sim": "maniskill"}})
        >>> evaluator = maker(video_dir="/tmp/videos")
    """
    sim_cfg = cfg.get("robot_env", {})
    sim_type = sim_cfg.get("sim", "maniskill")

    if sim_type == "maniskill":
        from umm.sims.maniskill_evaluator import ManiskillSimRollout

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        return partial(
            ManiskillSimRollout,
            cfg=cfg,
            device=device,
        )
    elif sim_type == "maniskill_legacy":
        # Legacy single-task evaluator from ext/lib_mani
        from lib_mani.maniskill_evaluator import ManiSkillEvaluator

        return partial(
            ManiSkillEvaluator,
            robot_link_pcl_fn=sim_cfg.get("robot_link_pcl_fn", None),
            env_id=sim_cfg.get("env_id", "StackCube-v1"),
            sim_backend=sim_cfg.get("sim_backend", "physx_cpu"),
            control_mode=sim_cfg.get("control_mode", "pd_ee_delta_pos"),
            num_eval_envs=sim_cfg.get("num_eval_envs", 10),
            max_episode_steps=sim_cfg.get("max_episode_steps", None),
            num_eval_episodes=sim_cfg.get("num_eval_episodes", 10),
            H=sim_cfg.get("H", 256),
            W=sim_cfg.get("W", 256),
            max_viz_envs=sim_cfg.get("max_viz_envs", -1),
        )
    else:
        raise NotImplementedError(f"Simulator {sim_type} not implemented")


# @rank_zero_only
def get_eval_callback(cfg, video_root=None):
    """Build and return a list of evaluation callbacks based on the config.

    If ``cfg`` does not contain a ``robot_env`` key the function returns an
    empty list, making it safe to call unconditionally.

    Args:
        cfg: Experiment configuration dictionary (parsed from YAML).
        video_root: Directory for storing rollout videos.  Defaults to
            ``None``.

    Returns:
        A list containing a single ``RobotSimCallback`` if the config
        includes ``robot_env``; otherwise an empty list.

    Example:
        >>> callbacks = get_eval_callback(cfg, video_root="outputs/videos")
        >>> trainer = pl.Trainer(callbacks=callbacks)
    """
    callbacks = []

    # Existing robot_env branch — preserved.
    if "robot_env" in cfg:
        evaluator_maker = get_evaluator_maker(cfg)
        callbacks.append(
            RobotSimCallback(
                evaluator_maker,
                video_root,
                eval_every_n_steps=cfg.get("robot_env", {}).get(
                    "eval_every_n_steps", 5000
                ),
                eval_at_beginning=cfg.get("robot_env", {}).get(
                    "eval_at_beginning", True
                ),
                safe_run=cfg.get("robot_env", {}).get("safe_run", True),
            )
        )

    # WMM LangTable Protocol A callback. Additive: runs alongside
    # robot_env / ms3_eval if configured. Lazy-imported so disabling
    # eval_protocol_a avoids pulling in the optional language_table dep.
    ep_a_cfg = cfg.get("callbacks", {}).get("eval_protocol_a") if cfg.get("callbacks") else None
    if ep_a_cfg is not None and bool(ep_a_cfg.get("enabled", False)):
        from umm.sims.wmm_lt_eval_callback import EvalProtocolACallback

        callbacks.append(EvalProtocolACallback(cfg))
        logging.info(
            "get_eval_callback: registered EvalProtocolACallback "
            "(early_fire_at=%s, every_n_iter=%s)",
            ep_a_cfg.get("early_fire_at"),
            ep_a_cfg.get("every_n_iter"),
        )

    # WMM OMOMO / CHOIS-Table-1 eval callback.
    # Additive: reads cfg.callbacks.omomo_eval. Disabled by default; opt-in via
    # enabled=True. Lazy-imported so configs without omomo_eval don't pull in
    # the full eval harness dep chain.
    om_cfg = cfg.get("callbacks", {}).get("omomo_eval") if cfg.get("callbacks") else None
    if om_cfg is not None and bool(om_cfg.get("enabled", False)):
        from umm.callbacks.wmm_omomo_eval import WmmOmomoEvalCallback

        log_dir_om = osp.join(video_root or ".", "..", "omomo_eval")
        log_dir_om = osp.abspath(log_dir_om)
        os.makedirs(log_dir_om, exist_ok=True)

        callbacks.append(
            WmmOmomoEvalCallback(
                gt_dir=om_cfg["gt_dir"],
                smplx_dir=om_cfg["smplx_dir"],
                sdf_folder=om_cfg["sdf_folder"],
                obj_geo_dir=om_cfg["obj_geo_dir"],
                classifier_ckpt=om_cfg["classifier_ckpt"],
                val_every_n_iter=int(om_cfg.get("val_every_n_iter", 10000)),
                log_dir=log_dir_om,
                gpu=int(om_cfg.get("gpu", 0)),
                debug_threshold=int(om_cfg.get("debug_threshold", 100)),
                max_bs_start=int(om_cfg.get("max_bs_start", 8)),
                eval_sample_ratio=float(om_cfg.get("eval_sample_ratio", 1.0)),
            )
        )
        logging.info(
            "get_eval_callback: registered WmmOmomoEvalCallback "
            "(val_every_n_iter=%s, eval_sample_ratio=%s)",
            om_cfg.get("val_every_n_iter"),
            om_cfg.get("eval_sample_ratio"),
        )

    # New MS3 on-the-fly eval branch. Additive: does not replace
    # the robot_env callback above — both may run together if configured.
    if cfg.get("ms3_eval") is not None:
        ms3 = cfg["ms3_eval"]
        import importlib
        from eval.ms3.evaluator import MS3EvalCallback

        ckpt_dir = ms3.get("ckpt_dir")
        if ckpt_dir is None:
            # Derive from video_root sibling (trainer.logger.log_dir is the
            # parent of video_root in UMM's layout).
            ckpt_dir = os.path.join(
                (video_root or "."), "..", "ms3_eval_ckpts",
            )
        ckpt_dir = os.path.abspath(ckpt_dir)

        task = ms3["task"]

        # dynamic adapter dispatch. Default preserves the pre-wmm
        # maniskill_bs pipeline byte-identically. New wmm configs set
        # ``ms3_eval.adapter: eval.ms3.policies.wmm_adapter:WmmAdapter``.
        adapter_spec = ms3.get(
            "adapter",
            "eval.ms3.policies.maniskill_bs_adapter:ManiskillBSAdapter",
        )
        if ":" not in adapter_spec:
            raise ValueError(
                f"ms3_eval.adapter must be 'module.path:ClassName', "
                f"got {adapter_spec!r}"
            )
        mod_name, cls_name = adapter_spec.split(":", 1)
        AdapterCls = getattr(importlib.import_module(mod_name), cls_name)

        # eval_effort scales periodic in-training eval to a cheap subset.
        # Code default is 1.0 (backward-compatible: legacy configs without the
        # field run the full protocol protocol). The 4 wmm MS3 YAMLs set 0.05 so
        # periodic eval runs ~5 episodes instead of the full 100-ep protocol.
        # A final benchmark run can override eval_effort=1.0 to recover full protocol.
        eval_effort = float(ms3.get("eval_effort", 1.0))
        if not (0.0 < eval_effort <= 1.0):
            raise ValueError(
                f"ms3_eval.eval_effort must be in (0, 1], got {eval_effort}"
            )

        n_episodes_config = int(ms3.get("n_episodes", 10))
        n_episodes_effective = max(1, int(round(n_episodes_config * eval_effort)))

        num_envs_config = int(ms3.get("num_envs", 1))
        # Cap num_envs at n_episodes_effective so we don't spawn more parallel
        # envs than episodes to run (wasted forkserver overhead).
        num_envs_effective = min(num_envs_config, n_episodes_effective)

        logger.info(
            "ms3_eval: eval_effort=%.3f -> n_episodes=%d (config=%d), "
            "num_envs=%d (config=%d)",
            eval_effort,
            n_episodes_effective,
            n_episodes_config,
            num_envs_effective,
            num_envs_config,
        )

        def _factory(pl_module, _AdapterCls=AdapterCls, _task=task):
            # Prefer from_pl_module (live-wrap factory classmethod);
            # fall back to direct constructor with pl_module kwarg.
            if hasattr(_AdapterCls, "from_pl_module"):
                return _AdapterCls.from_pl_module(
                    pl_module=pl_module, task=_task
                )
            return _AdapterCls(ckpt_path=None, task=_task, pl_module=pl_module)

        record_video = bool(ms3.get("record_video", False))
        video_dir_cfg = ms3.get("video_dir", None)
        if video_dir_cfg is None and record_video:
            # Derive from video_root sibling — same pattern as ckpt_dir above.
            # Result: logs/<expname>/version_<V>/ms3_eval_videos/
            # rather than an orphan outputs/ path divorced from log_dir.
            video_dir_cfg = os.path.abspath(
                os.path.join((video_root or "."), "..", "ms3_eval_videos")
            )
        video_fps_cfg = int(ms3.get("video_fps", 20))
        callbacks.append(
            MS3EvalCallback(
                task=task,
                ckpt_dir=ckpt_dir,
                num_envs=num_envs_effective,
                n_episodes=n_episodes_effective,
                eval_every_n_steps=int(
                    ms3.get("eval_every_n_train_steps", 5000)
                ),
                max_ep_steps=int(ms3.get("max_ep_steps", 100)),
                seed=int(ms3.get("seed", 1)),
                policy_factory=_factory,
                top_k=int(ms3.get("top_k", 5)),
                record_video=record_video,
                video_dir=video_dir_cfg,
                video_fps=video_fps_cfg,
            )
        )

    # TG eval callback (scope extension — wires TgEvalCallback
    # into the config-driven callback list). Additive: coexists with
    # ``robot_env`` and ``ms3_eval`` branches above.
    # Truthy guard (not `is not None`) so an empty dict ``tg_eval: {}`` does
    # not hit a cryptic `KeyError` on the mandatory ``env_name`` lookup
    # below — empty dict signals "disabled", same as absent.
    if cfg.get("tg_eval"):
        tg = cfg["tg_eval"]
        from umm.training_utils.callbacks import TgEvalCallback

        callbacks.append(
            TgEvalCallback(
                env_name=tg["env_name"],
                test_data_root=tg["test_data_root"],
                splits=tg.get("splits", None),  # retained kwarg; callback ignores it
                eval_every_n_steps=int(tg.get("eval_every_n_steps", 10000)),
                batch_size=int(tg.get("batch_size", 4)),
                debug_every_n_steps=int(tg.get("debug_every_n_steps", -1)),
                debug_subset_n_samples=int(tg.get("debug_subset_n_samples", 16)),
                num_workers=int(tg.get("num_workers", 4)),
                eval_effort=float(tg.get("eval_effort", 1.0)),
                log_prefix=tg.get("log_prefix", "tg_eval"),
                inference_kwargs=dict(tg.get("inference_kwargs") or {}),
            )
        )

    return callbacks
