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
                # CHOIS-faithful sampling defaults (chosen 2026-04-30):
                # M1 (replace) + M3 (smoothing 0.05) on strict partial
                # waypoints. YAML can override; the callback warns loudly
                # if any of these drift from the canonical default.
                partial_chois=str(om_cfg.get("partial_chois", "strict")),
                chois_mode=str(om_cfg.get("chois_mode", "replace")),
                chois_guide_scale=float(om_cfg.get("chois_guide_scale", 1.0)),
                smooth_lambda=float(om_cfg.get("smooth_lambda", 0.05)),
            )
        )
        logging.info(
            "get_eval_callback: registered WmmOmomoEvalCallback "
            "(val_every_n_iter=%s, eval_sample_ratio=%s)",
            om_cfg.get("val_every_n_iter"),
            om_cfg.get("eval_sample_ratio"),
        )

    # WMM GRAB / handobject_new eval callback.
    # Additive: reads cfg.callbacks.grab_eval. Disabled by default; opt-in via
    # enabled=True. Lazy-imported so configs without grab_eval don't pull in
    # the canonical_eval / IK dep chain.
    gr_cfg = cfg.get("callbacks", {}).get("grab_eval") if cfg.get("callbacks") else None
    if gr_cfg is not None and bool(gr_cfg.get("enabled", False)):
        from umm.training_utils.callbacks.grab_eval import WmmGrabEvalCallback

        log_dir_gr = osp.abspath(osp.join(video_root or ".", "..", "grab_eval"))
        os.makedirs(log_dir_gr, exist_ok=True)

        callbacks.append(
            WmmGrabEvalCallback(
                test_root=gr_cfg.get("test_root", "data_wmm/handobject_new/canonical"),
                split=gr_cfg["split"],
                run_tag=gr_cfg["run_tag"],
                mano_dir=gr_cfg.get("mano_dir", "ext/baselines/diffh2o/dataset/mano"),
                diffh2o_dir=gr_cfg.get("diffh2o_dir"),
                clf_path=gr_cfg.get("clf_path"),
                sbj_model_path=gr_cfg.get("sbj_model_path"),
                data_root=gr_cfg.get("data_root", "data_wmm/handobject/imos_deps"),
                mesh_dir=gr_cfg.get("mesh_dir"),
                val_every_n_iter=int(gr_cfg.get("val_every_n_iter", 10000)),
                log_dir=log_dir_gr,
                predictions_root=gr_cfg.get("predictions_root", "predictions"),
                eval_sample_ratio=float(gr_cfg.get("eval_sample_ratio", 0.04)),
                debug_threshold=int(gr_cfg.get("debug_threshold", 100)),
                n_viz_seqs=int(gr_cfg.get("n_viz_seqs", 0)),
                soft_budget_s=float(gr_cfg.get("soft_budget_s", 480.0)),
                hard_budget_s=float(gr_cfg.get("hard_budget_s", 600.0)),
                cleanup_predictions=bool(gr_cfg.get("cleanup_predictions", True)),
                scheduler_mode=gr_cfg.get("scheduler_mode", "ddim_gen"),
                n_steps_debug=int(gr_cfg.get("n_steps_debug", 10)),
                n_steps_prod=(
                    int(gr_cfg["n_steps_prod"])
                    if gr_cfg.get("n_steps_prod") is not None else None
                ),
                max_T=int(gr_cfg.get("max_T", 200)),
                num_pts=int(gr_cfg.get("num_pts", 256)),
                skip_mirrored=bool(gr_cfg.get("skip_mirrored", True)),
            )
        )
        logging.info(
            "get_eval_callback: registered WmmGrabEvalCallback "
            "(split=%s, run_tag=%s, val_every_n_iter=%s, eval_sample_ratio=%s)",
            gr_cfg["split"], gr_cfg["run_tag"],
            gr_cfg.get("val_every_n_iter"), gr_cfg.get("eval_sample_ratio"),
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
        # field run the full reference protocol). The 4 wmm MS3 YAMLs set 0.05 so
        # periodic eval runs ~5 episodes instead of the full 100-ep protocol.
        # A final benchmark run can override eval_effort=1.0 to recover full reference.
        eval_effort = float(ms3.get("eval_effort", 1.0))
        if not (0.0 < eval_effort <= 1.0):
            raise ValueError(
                f"ms3_eval.eval_effort must be in (0, 1], got {eval_effort}"
            )

        n_episodes_cfg = int(ms3.get("n_episodes", 10))
        n_episodes_effective = max(1, int(round(n_episodes_cfg * eval_effort)))

        num_envs_cfg = int(ms3.get("num_envs", 1))
        # Cap num_envs at n_episodes_effective so we don't spawn more parallel
        # envs than episodes to run (wasted forkserver overhead).
        num_envs_effective = min(num_envs_cfg, n_episodes_effective)

        logger.info(
            "ms3_eval: eval_effort=%.3f -> n_episodes=%d (configured=%d), "
            "num_envs=%d (configured=%d)",
            eval_effort,
            n_episodes_effective,
            n_episodes_cfg,
            num_envs_effective,
            num_envs_cfg,
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

    # TG eval callback (scope extension — wires 's TgEvalCallback
    # into the config-driven callback list). Additive: coexists with
    # ``robot_env`` and ``ms3_eval`` branches above.
    # Multi-env support: this codebase's YAML loader (configargparse
    # YAMLConfigFileParser) does NOT preserve list-of-dicts at the top
    # level — it flattens to the last item. To support multi-env eval
    # (e.g., joint droid+epic training with two on-fly metrics), we use
    # multiple flat top-level keys: ``tg_eval`` (primary, used for
    # ckpt_monitor by convention) and optional ``tg_eval_aux``,
    # ``tg_eval_aux2``, ``tg_eval_aux3`` for additional envs. Each is a
    # regular dict with the same shape as ``tg_eval``. Each becomes its
    # own callback logging to ``{log_prefix}/{env_name}/...``.
    _tg_eval_blocks: list = []
    for _key in ("tg_eval", "tg_eval_aux", "tg_eval_aux2", "tg_eval_aux3"):
        if cfg.get(_key):
            _tg_eval_blocks.append(cfg[_key])

    if _tg_eval_blocks:
        # ``dummy_id_rot`` falls back to the training dataset's value so eval
        # automatically mirrors training when the YAML opts in. Per-eval
        # ``dummy_id_rot`` override still wins. Use first dataset entry as
        # the source of truth (mixed-dataset configs MUST share the same
        # dummy_id_rot setting; assert that below).
        try:
            train_dummy_id_rot = bool(
                cfg["dataset"]["configs"][0].get("dummy_id_rot", False)
            )
        except (KeyError, IndexError, TypeError):
            train_dummy_id_rot = False
        # Sanity: require all dataset configs to share dummy_id_rot. Mixed
        # configs with conflicting flags would feed the model OOD rotation
        # channels for one of the envs.
        for _dsi, _dsc in enumerate(cfg.get("dataset", {}).get("configs", [])):
            _dsi_flag = bool(_dsc.get("dummy_id_rot", False))
            if _dsi_flag != train_dummy_id_rot:
                raise ValueError(
                    f"dataset.configs[{_dsi}].dummy_id_rot={_dsi_flag} differs "
                    f"from configs[0].dummy_id_rot={train_dummy_id_rot}; "
                    f"mixed-dataset training requires the same dummy_id_rot."
                )

        for _ei, tg in enumerate(_tg_eval_blocks):
            # ( 20260424-2353-a89fef1b): callback_class dispatch.
            # Default ``TgEvalCallback`` is byte-identical to the pre-
            # branch; ``Tg128EvalCallback`` adds the K_train-vs-P sliding
            # window plumbing for K_train != P configs (e.g., K=128 DROID).
            callback_class = tg.get("callback_class", "TgEvalCallback")

            common_kwargs = dict(
                env_name=tg["env_name"],
                test_data_root=tg["test_data_root"],
                splits=tg.get("splits", None),
                eval_every_n_steps=int(tg.get("eval_every_n_steps", 10000)),
                batch_size=int(tg.get("batch_size", 4)),
                debug_every_n_steps=int(tg.get("debug_every_n_steps", -1)),
                debug_subset_n_samples=int(tg.get("debug_subset_n_samples", 16)),
                num_workers=int(tg.get("num_workers", 4)),
                eval_effort=float(tg.get("eval_effort", 1.0)),
                log_prefix=tg.get("log_prefix", "tg_eval"),
                inference_kwargs=dict(tg.get("inference_kwargs") or {}),
                dummy_id_rot=bool(tg.get("dummy_id_rot", train_dummy_id_rot)),
            )

            if callback_class == "TgEvalCallback":
                from umm.training_utils.callbacks import TgEvalCallback

                callbacks.append(TgEvalCallback(**common_kwargs))
            elif callback_class == "Tg128EvalCallback":
                from umm.training_utils.callbacks import Tg128EvalCallback

                # K_train: read from dataset.configs[0].num_kps (the trained K).
                # In mixed configs all entries must share num_kps (same model
                # was trained on the same K window).
                try:
                    K_train = int(cfg["dataset"]["configs"][0]["num_kps"])
                except (KeyError, IndexError, TypeError) as e:
                    raise ValueError(
                        "Tg128EvalCallback requires cfg.dataset.configs[0].num_kps "
                        f"to derive K_train; got error: {e!r}"
                    )
                anchor_count = int(
                    cfg.get("model", {})
                    .get("diffusion", {})
                    .get("sliding_window_anchors", K_train // 2)
                )
                callbacks.append(
                    Tg128EvalCallback(
                        K_train=K_train,
                        anchor_count=anchor_count,
                        **common_kwargs,
                    )
                )
            else:
                raise ValueError(
                    f"unknown tg_evals[{_ei}].callback_class={callback_class!r} "
                    f"(expected 'TgEvalCallback' or 'Tg128EvalCallback')"
                )

    # Tg128SlidingVizCallback.
    # Periodic visual monitor for K=128 training: runs full P=400 sliding-
    # window inference on a fixed 1-sample reference and writes K400sw MP4s.
    # Independent of ``tg_eval`` block — fires at its own faster cadence
    # (typically 500 steps vs eval at 10000) and only when explicitly wired
    # in the YAML (truthy guard, same convention as ``tg_eval``).
    if cfg.get("tg128_sliding_viz"):
        sw = cfg["tg128_sliding_viz"]
        try:
            ds0 = cfg["dataset"]["configs"][0]
            K_train = int(ds0["num_kps"])
            # sw-viz-debug fix: per-frame timestamps for the LMM temporal
            # RoPE are read from the dataset config — the SINGLE SOURCE OF
            # TRUTH for the time axis the model was trained on. Required
            # keys; no fallback defaults — a missing base_fps/t_rate is a
            # broken config and must fail loud.
            sw_base_fps = float(ds0["base_fps"])
            sw_t_rate = int(ds0["t_rate"])
            # iter 04b fix: dummy_id_rot must mirror the dataset setting so
            # SW-viz inference uses the same rot_mask as training (and the
            # K=400 TgEvalCallback path). Defaults to False if absent.
            sw_dummy_id_rot = bool(ds0.get("dummy_id_rot", False))
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                "tg128_sliding_viz requires cfg.dataset.configs[0] to define "
                "num_kps, base_fps, and t_rate (the SW viz must use the same "
                f"time axis as training); got error: {e!r}"
            )
        anchor_count = int(
            cfg.get("model", {})
            .get("diffusion", {})
            .get("sliding_window_anchors", K_train // 2)
        )
        from umm.training_utils.callbacks import Tg128SlidingVizCallback

        callbacks.append(
            Tg128SlidingVizCallback(
                env_name=sw["env_name"],
                test_data_root=sw["test_data_root"],
                viz_every_n_train_steps=int(sw["viz_every_n_train_steps"]),
                K_train=K_train,
                anchor_count=anchor_count,
                image_size=int(sw.get("image_size", 384)),
                inference_kwargs=dict(sw.get("inference_kwargs") or {}),
                base_fps=sw_base_fps,
                t_rate=sw_t_rate,
                dummy_id_rot=sw_dummy_id_rot,
            )
        )

    return callbacks
