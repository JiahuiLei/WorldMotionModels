"""In-training Protocol A eval callback for WMM LangTable.

Runs the canonical Language Table Protocol A evaluation (5 reward
factories × ``num_rollouts`` rollouts × ``BLOCK_8`` × ``seed=0`` ×
``max_steps=200``) at configurable training iterations, logs per-reward
and mean success rates to the trainer logger, writes MP4 videos + a
``results.json`` per fire, and hard-fails on train/eval format mismatch
via ``umm.sims.wmm_lt_protocol_a_invariants``.

Wired into the trainer by ``umm.sims.robot_sim_callbacks.get_eval_callback``
when ``cfg.callbacks.eval_protocol_a.enabled`` is truthy. All config
fields are read from the scoped ``cfg.callbacks.eval_protocol_a.*``
subtree — this callback deliberately does NOT read ``cfg.viz_try`` or
any other global.

Example (in YAML)::

    callbacks:
      eval_protocol_a:
        enabled: True
        early_fire_at: [2000]
        every_n_iter: 10000
        num_rollouts: 50
        env_seed: 0
        max_steps: 200
        hist: 5
        window: 20
        act_horizon: 10
        viz_try: True   # production — wrap run in try/except
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import time
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities import rank_zero_only

from umm.sims.wmm_lt_protocol_a_invariants import (
    assert_action_compatible,
    assert_observation_compatible,
    oracle_replay_sanity,
)


log = logging.getLogger(__name__)


# Mirror of eval/langtable_official/eval_protocol_a.py:REWARDS. Lazy-
# resolved inside methods so importing this module does NOT require
# language_table to be importable (optional dep).
_REWARD_KEY_TO_CLS_PATH = {
    "blocktoblock": (
        "language_table.environments.rewards.block2block",
        "BlockToBlockReward",
    ),
    "blocktoabsolutelocation": (
        "language_table.environments.rewards.block2absolutelocation",
        "BlockToAbsoluteLocationReward",
    ),
    "blocktoblockrelativelocation": (
        "language_table.environments.rewards.block2block_relative_location",
        "BlockToBlockRelativeLocationReward",
    ),
    "blocktorelativelocation": (
        "language_table.environments.rewards.block2relativelocation",
        "BlockToRelativeLocationReward",
    ),
    "separate": (
        "language_table.environments.rewards.separate_blocks",
        "SeparateBlocksReward",
    ),
}


def _resolve_rewards() -> dict[str, Any]:
    """Resolve reward classes lazily (language_table is an optional dep)."""
    import importlib
    out: dict[str, Any] = {}
    for name, (mod_path, cls_name) in _REWARD_KEY_TO_CLS_PATH.items():
        mod = importlib.import_module(mod_path)
        out[name] = getattr(mod, cls_name)
    return out


def _build_env(reward_cls: Any, seed: int, block_mode_name: str = "BLOCK_8") -> Any:
    """Build a LangTable env with all rendering flags matching Protocol A protocol.

    Mirrors ext/lib_langtable/language_table/eval/main.py:80-83 except for
    seed, which the caller passes. ``block_mode`` is BLOCK_8 per the evaluation protocol.
    """
    from language_table.environments import blocks, language_table  # type: ignore[import-not-found]
    block_variant = getattr(blocks.LanguageTableBlockVariants, block_mode_name)
    env = language_table.LanguageTable(
        block_mode=block_variant,
        reward_factory=reward_cls,
        render_text_in_image=True,
        render_depth=True,
        render_segmentation=True,
        render_semantic_segmentation=True,
        render_instance_poses=True,
        seed=seed,
    )
    return env


class EvalProtocolACallback(pl.Callback):
    """Runs full LangTable Protocol A evaluation at configurable iterations.

    Fire condition:

    - ``trainer.global_step in early_fire_at`` (list of specific steps), OR
    - ``every_n_iter > 0 and trainer.global_step % every_n_iter == 0 and
      trainer.global_step > 0``.

    Args:
        cfg: The full training :class:`Config`. Reads scoped fields at
            ``cfg.callbacks.eval_protocol_a.*`` only — never global
            ``cfg.viz_try``.

    Scoped fields (``cfg.callbacks.eval_protocol_a.*``):

        tasks (list[str] | None): Which Protocol A reward factories to
            evaluate. ``None`` or missing => all 5 keys of
            ``_REWARD_KEY_TO_CLS_PATH`` (full Protocol A sweep, protocol
            default). An explicit empty list ``[]`` is honoured as
            "evaluate zero tasks" — the per-reward loop is skipped
            entirely and a warning is logged. Every entry must be one
            of the 5 reward keys; unknown entries raise ``ValueError``.
    """

    def __init__(self, cfg: Any):
        super().__init__()
        scoped = cfg.get("callbacks", {}).get("eval_protocol_a", {})
        self.enabled: bool = bool(scoped.get("enabled", False))
        self.early_fire_at: list[int] = list(scoped.get("early_fire_at", []) or [])
        self.every_n_iter: int = int(scoped.get("every_n_iter", 0))
        self.num_rollouts: int = int(scoped.get("num_rollouts", 50))
        self.env_seed: int = int(scoped.get("env_seed", 0))
        self.max_steps: int = int(scoped.get("max_steps", 200))
        self.hist: int = int(scoped.get("hist", 5))
        self.window: int = int(scoped.get("window", 20))
        self.act_horizon: int = int(scoped.get("act_horizon", 10))
        self.viz_try: bool = bool(scoped.get("viz_try", True))
        self.block_mode: str = str(scoped.get("block_mode", "BLOCK_8"))

        # Scope of reward factories to evaluate. None/missing => all 5
        # keys of _REWARD_KEY_TO_CLS_PATH (full Protocol A sweep). An
        # explicit empty list [] is preserved and treated as "evaluate
        # zero tasks" (skip the per-reward loop entirely — logged as a
        # warning in _run_full_protocol_a).
        tasks_raw = scoped.get("tasks")
        if tasks_raw is None:
            self.tasks: list[str] = list(_REWARD_KEY_TO_CLS_PATH)
        else:
            if not isinstance(tasks_raw, (list, tuple)):
                raise ValueError(
                    f"callbacks.eval_protocol_a.tasks must be a list/None, got "
                    f"{type(tasks_raw).__name__}: {tasks_raw!r}"
                )
            for name in tasks_raw:
                if name not in _REWARD_KEY_TO_CLS_PATH:
                    raise ValueError(
                        f"Unknown eval_protocol_a.tasks entry {name!r}; "
                        f"expected any of {list(_REWARD_KEY_TO_CLS_PATH)}"
                    )
            self.tasks = list(tasks_raw)

        # One-shot guards.
        self._sanity_done: bool = False
        self._fired_steps: set[int] = set()

        log.info(
            "EvalProtocolACallback: enabled=%s early_fire_at=%s every_n_iter=%d "
            "num_rollouts=%d max_steps=%d block_mode=%s viz_try=%s tasks=%s",
            self.enabled,
            self.early_fire_at,
            self.every_n_iter,
            self.num_rollouts,
            self.max_steps,
            self.block_mode,
            self.viz_try,
            self.tasks,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @rank_zero_only
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        """Run oracle-replay sanity ONCE on a throwaway env at fit start.

        Guarded by ``self._sanity_done`` — Lightning may call ``setup``
        for multiple stages; we only want to burn the 20-step oracle
        rollout once.
        """
        if stage != "fit" or not self.enabled or self._sanity_done:
            return
        try:
            rewards = _resolve_rewards()
            sanity_env = _build_env(
                reward_cls=rewards["blocktoblock"],
                seed=self.env_seed,
                block_mode_name=self.block_mode,
            )
            try:
                from language_table.replay.push_oracle_rrt_slowdown import (  # type: ignore[import-not-found]
                    ObstacleOrientedPushOracleBoard2dRRT,
                )
                oracle = ObstacleOrientedPushOracleBoard2dRRT(sanity_env, use_ee_planner=True)
                oracle_replay_sanity(sanity_env, oracle, n_steps=20)
            finally:
                close_fn = getattr(sanity_env, "close", None)
                if close_fn is not None:
                    try:
                        close_fn()
                    except Exception as close_e:
                        log.warning("EvalProtocolACallback: sanity env.close failed: %s", close_e)
            self._sanity_done = True
            log.info("EvalProtocolACallback: oracle_replay_sanity PASS")
        except AssertionError:
            # Invariants must not be swallowed silently. If viz_try, log at
            # ERROR and flag _sanity_done=True so we don't retry on every
            # stage; but keep training going. If viz_try=False, re-raise.
            log.exception("EvalProtocolACallback: oracle_replay_sanity FAILED")
            if not self.viz_try:
                raise
            log.error(
                "EvalProtocolACallback: proceeding with eval WITHOUT validated "
                "invariants (viz_try=True) — results may be unreliable."
            )
            self._sanity_done = True  # don't retry; sanity is one-shot
        except Exception:
            log.exception("EvalProtocolACallback: sanity setup failed (non-assert)")
            if not self.viz_try:
                raise
            self._sanity_done = True

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Fire a full Protocol A run when the schedule matches."""
        if not self.enabled:
            return
        step = int(trainer.global_step)
        fire = step in self.early_fire_at
        fire = fire or (self.every_n_iter > 0 and step > 0 and step % self.every_n_iter == 0)
        if not fire or step in self._fired_steps:
            return
        self._fired_steps.add(step)

        if self.viz_try:
            try:
                self._run_full_protocol_a(pl_module, trainer)
            except Exception:
                log.exception(
                    "EvalProtocolACallback: protocol_a run at step %d FAILED "
                    "(viz_try=True — continuing training)", step,
                )
        else:
            self._run_full_protocol_a(pl_module, trainer)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _pick_eval_model(self, pl_module: pl.LightningModule) -> torch.nn.Module:
        """Pick EMA copy when available, base model otherwise.

        EMA failures are NOT silently swallowed — they log at WARN level
        so stale-weight eval is visible in the training log. The eval
        still proceeds (using whatever state the EMA copy is in), since
        the alternative is dropping the eval entirely.
        """
        has_ema_model = getattr(pl_module, "ema_model", None) is not None
        has_ema = getattr(pl_module, "ema", None) is not None
        if getattr(pl_module, "use_ema", False) and has_ema_model and has_ema:
            # Ensure EMA weights are up to date and device-aligned.
            try:
                pl_module.ema.copy_to(pl_module.ema_model.parameters())
            except Exception as e:
                log.warning(
                    "EvalProtocolACallback: EMA copy_to FAILED — eval will "
                    "use STALE ema_model weights: %r", e,
                )
            try:
                from umm.models.wmm.model import sync_model_buffers  # type: ignore[attr-defined]
                sync_model_buffers(pl_module.model, pl_module.ema_model)
            except Exception as e:
                log.warning(
                    "EvalProtocolACallback: sync_model_buffers FAILED — "
                    "eval ema_model buffers may be STALE: %r", e,
                )
            eval_model = pl_module.ema_model
        else:
            if getattr(pl_module, "use_ema", False) and not (has_ema_model and has_ema):
                log.warning(
                    "EvalProtocolACallback: use_ema=True but "
                    "ema=%s / ema_model=%s — falling back to pl_module.model",
                    has_ema, has_ema_model,
                )
            eval_model = pl_module.model
        eval_model.eval()
        return eval_model

    @torch.no_grad()
    def _run_full_protocol_a(self, pl_module: pl.LightningModule, trainer: pl.Trainer) -> None:
        """Execute all 5 rewards × num_rollouts."""
        visualizer = getattr(pl_module, "langtable_visualizer", None)
        assert visualizer is not None and getattr(visualizer, "flag", False), (
            "EvalProtocolACallback: pl_module.langtable_visualizer is None or "
            "disabled — cfg.model.langtable block is missing or empty."
        )

        eval_model = self._pick_eval_model(pl_module)
        device = pl_module.device
        step = int(trainer.global_step)

        # Guard empty-scope BEFORE any language_table import so machines
        # without the optional dep don't crash on a no-op call.
        if not self.tasks:
            log.warning(
                "EvalProtocolACallback[step=%d]: tasks=[] — skipping per-reward "
                "eval loop entirely (explicit empty scope).", step,
            )
            return

        log_dir = getattr(trainer, "log_dir", None)
        if not log_dir:
            log_dir = getattr(trainer.logger, "log_dir", None)
        if not log_dir:
            log.warning(
                "EvalProtocolACallback: trainer has no log_dir — "
                "falling back to CWD %r", os.getcwd(),
            )
            log_dir = "."
        out_root = osp.join(log_dir, f"protocol_a_iter_{step:08d}")
        videos_root = osp.join(out_root, "videos")
        os.makedirs(videos_root, exist_ok=True)

        rewards = _resolve_rewards()
        from language_table.replay.push_oracle_rrt_slowdown import (  # type: ignore[import-not-found]
            ObstacleOrientedPushOracleBoard2dRRT,
        )

        t0 = time.time()
        all_results: dict[str, dict[str, Any]] = {}
        schema_asserted = False

        for reward_name in self.tasks:
            reward_cls = rewards[reward_name]
            log.info("EvalProtocolACallback[step=%d]: starting reward %s (%d rollouts)",
                     step, reward_name, self.num_rollouts)
            env = _build_env(reward_cls=reward_cls, seed=self.env_seed,
                             block_mode_name=self.block_mode)
            reward_videos_dir = osp.join(videos_root, reward_name)
            os.makedirs(reward_videos_dir, exist_ok=True)

            successes = 0
            per_ep: list[dict[str, Any]] = []
            try:
                for ep_num in range(self.num_rollouts):
                    # Plan-feasibility retry — mirror eval/main.py:107-117.
                    oracle = ObstacleOrientedPushOracleBoard2dRRT(env, use_ee_planner=True)
                    plan_ok = False
                    for _ in range(20):
                        env.reset()
                        raw_state = env.compute_state()
                        if oracle.get_plan(raw_state):
                            plan_ok = True
                            break
                    if not plan_ok:
                        log.warning(
                            "EvalProtocolACallback[step=%d][%s] ep %d: no feasible init after 20 retries",
                            step, reward_name, ep_num,
                        )
                        per_ep.append({"ep": ep_num, "success": False, "reason": "no_feasible_init"})
                        continue

                    # First episode of each reward: assert env obs schema
                    # against the training contract BEFORE the rollout
                    # exercises infer_agent.
                    if not schema_asserted:
                        assert_observation_compatible(env._compute_observation())
                        schema_asserted = True

                    succ, succ_steps, _trace, _ctx, _uid, _pcl, viz_frames, instruction, _det = (
                        visualizer.rollout_full_states_trivial_policy(
                            agent=eval_model,
                            seed=self.env_seed + ep_num,
                            max_steps=self.max_steps,
                            hist=self.hist,
                            window=self.window,
                            act_horizon=self.act_horizon,
                            device=device,
                            env=env,  # patched helper to honor caller env
                            strict_action_check=True,  # protocol eval path — fail loud
                            # Above plan-feasibility retry already
                            # reset+validated the env; helper must NOT
                            # reset again (would discard the validated
                            # init).
                            skip_env_reset=True,
                        )
                    )
                    if succ:
                        successes += 1
                    per_ep.append({
                        "ep": ep_num,
                        "success": bool(succ),
                        "steps": int(succ_steps),
                    })
                    # Write MP4 for this episode.
                    self._write_mp4(reward_videos_dir, reward_name, ep_num,
                                    bool(succ), viz_frames, instruction)

            finally:
                close_fn = getattr(env, "close", None)
                if close_fn is not None:
                    try:
                        close_fn()
                    except Exception as close_e:
                        log.warning("EvalProtocolACallback: env.close failed: %s", close_e)

            total = max(self.num_rollouts, 1)
            rate = successes / total
            all_results[reward_name] = {
                "reward": reward_name,
                "successes": successes,
                "total": self.num_rollouts,
                "rate": rate,
                "episodes": per_ep,
            }
            log.info("EvalProtocolACallback[step=%d][%s] done: %d/%d = %.3f",
                     step, reward_name, successes, self.num_rollouts, rate)
            # Checkpoint partial results after every reward.
            self._write_results_json(out_root, all_results, step, elapsed=time.time() - t0)
            # Log per-reward to wandb/TB.
            try:
                pl_module.log(
                    f"protocol_a/{reward_name}_success",
                    torch.tensor(rate, dtype=torch.float32),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    rank_zero_only=True,
                )
            except Exception:
                log.exception("pl_module.log failed for reward %s", reward_name)

        # Final avg.
        if all_results:
            avg = float(np.mean([r["rate"] for r in all_results.values()]))
            try:
                pl_module.log(
                    "protocol_a/avg_success",
                    torch.tensor(avg, dtype=torch.float32),
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    rank_zero_only=True,
                )
            except Exception:
                log.exception("pl_module.log failed for avg_success")
            log.info("EvalProtocolACallback[step=%d] SUMMARY avg=%.3f", step, avg)

        # Log rollout MP4s to wandb, grouped by reward name so the UI
        # surfaces them under media/videos/protocol_a/videos/<reward>/.
        # pl_module.log() only handles scalars; MP4 upload requires the
        # raw wandb.experiment (same pattern training-time langtable_viz
        # uses via self.pl.logger.log_video(...)).
        self._log_mp4s_to_wandb(trainer, out_root, all_results)

    @staticmethod
    def _log_mp4s_to_wandb(
        trainer: pl.Trainer,
        out_root: str,
        all_results: dict,
    ) -> None:
        """Upload per-reward rollout MP4s grouped by reward to wandb.

        Non-fatal: the disk MP4s + scalar rates are already logged before
        this call. If wandb upload fails (no wandb, offline mode, network
        hiccup, etc.) we log a warning and return — training is not
        impacted, and the MP4s stay on disk under
        ``<log_dir>/protocol_a_iter_<step>/videos/<reward>/``.
        """
        from glob import glob

        logger_exp = getattr(trainer.logger, "experiment", None)
        if logger_exp is None:
            log.info(
                "EvalProtocolACallback: trainer.logger.experiment is None; "
                "skipping wandb video upload (scalars + disk MP4s unaffected)."
            )
            return
        try:
            import wandb  # type: ignore[import-untyped]
        except Exception as e:  # wandb not installed
            log.warning(
                "EvalProtocolACallback: wandb import failed — skipping video "
                "upload (scalars + disk MP4s unaffected): %r", e,
            )
            return

        videos_root = osp.join(out_root, "videos")
        payload: dict[str, Any] = {
            "trainer/global_step": int(trainer.global_step),
        }
        for reward_name in all_results:
            videodir = osp.join(videos_root, reward_name)
            mp4_paths = sorted(glob(osp.join(videodir, "*.mp4")))
            if not mp4_paths:
                log.warning(
                    "EvalProtocolACallback: no MP4s found under %s "
                    "for wandb upload", videodir,
                )
                continue
            try:
                payload[f"protocol_a/videos/{reward_name}"] = [
                    wandb.Video(p, format="mp4") for p in mp4_paths
                ]
            except Exception as e:
                log.warning(
                    "EvalProtocolACallback: wandb.Video wrap failed for "
                    "%s reward %s: %r", videodir, reward_name, e,
                )
        # Only call log if we actually built some video entries beyond the
        # global_step marker.
        video_keys = [k for k in payload if k.startswith("protocol_a/videos/")]
        if not video_keys:
            return
        try:
            logger_exp.log(payload)
            log.info(
                "EvalProtocolACallback: uploaded %d reward videos to wandb "
                "(keys=%s)", len(video_keys), sorted(video_keys),
            )
        except Exception as e:
            log.warning(
                "EvalProtocolACallback: wandb experiment.log failed — "
                "scalars + disk MP4s unaffected: %r", e,
            )

    @staticmethod
    def _write_mp4(
        out_dir: str,
        reward_name: str,
        ep_num: int,
        succ: bool,
        viz_frames: list[np.ndarray],
        instruction: str,
    ) -> None:
        """Dump per-episode MP4 using imageio."""
        import imageio  # local — keep top-level imports minimal
        if not viz_frames:
            log.warning("EvalProtocolACallback: empty viz_frames for %s ep %d", reward_name, ep_num)
            return
        status = "success" if succ else "failure"
        fn = osp.join(out_dir, f"{reward_name}_ep{ep_num:03d}_{status}.mp4")
        # Stack frames; LT renders uint8 HxWx3 per-frame already.
        frames = [np.asarray(f, dtype=np.uint8) for f in viz_frames]
        try:
            imageio.mimwrite(fn, frames, fps=10)
        except Exception:
            log.exception("EvalProtocolACallback: imageio.mimwrite failed for %s", fn)

    @staticmethod
    def _write_results_json(out_root: str, all_results: dict, step: int, elapsed: float) -> None:
        os.makedirs(out_root, exist_ok=True)
        payload = {
            "step": int(step),
            "elapsed_sec": float(elapsed),
            "rewards": {
                k: {kk: vv for kk, vv in v.items() if kk != "episodes"}
                for k, v in all_results.items()
            },
            "rewards_detailed": all_results,
        }
        with open(osp.join(out_root, "results.json"), "w") as f:
            json.dump(payload, f, indent=2)


# Tiny helper asserting action before env.step — exposed for future
# callers that want to test the action converter outside the callback.
def assert_action_ok(action, env) -> None:
    """Shim for ``assert_action_compatible`` with scoped env slot."""
    assert_action_compatible(action, env.action_space)
