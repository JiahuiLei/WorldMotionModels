"""On-the-fly humanoid retargeting eval callback for w1.0_hh / wmm.

Mirrors :class:`umm.callbacks.wmm_omomo_eval.WmmOmomoEvalCallback` in
structure: a single ``WmmHhEvalCallback(pl.Callback)`` with two
branches selected by ``val_every_n_iter`` vs ``debug_threshold``:

  * **Debug** (``val_every_n_iter < debug_threshold``):
    2 sids, ``agent.hh_joint_infer(use_boost=False)`` only, NO mujoco
    rollout, NO 4-panel video, NO IK refinement. Fast NaN-finite check
    on the inference output. Designed for ``--fast_dev_run`` debug flow
    where any mujoco step would dominate the iteration.

  * **Production** (``val_every_n_iter >= debug_threshold``):
    10 sids evenly sampled from
    ``data_splits/amass_humanoid_v2/full_test.json`` (every 201st:
    indices ``[0, 201, 402, ..., 1809]``). Full pipeline:
      1. Build human Q via SMPL-X HumanModel forward.
      2. ``agent.hh_joint_infer(use_boost=False)`` → robot Q.
      3. ``G1Interface.RK`` + ``adjust`` → root_pos / root_quat /
         dof_pos.
      4. Save kin npz to
         ``<log_dir>/eval_onfly/step_<N>/kin/<sid>.npz``.
      5. Spawn mujoco rollout (subprocess, ``--thread <mjc_threads>``)
         to ``mjc_rollout/``.
      6. Compute SR-seq / gerr / lerr per
         ``eval/retargeting_new/scripts/metrics.py``.
      7. Render 4-panel video (input human / kin robot / qpos-projected
         / phys) → ``viz/<sid>.mp4``.
      8. Write ``summary.json``; log scalars + ``wandb.Video`` to wandb.

  * Failures wrapped in ``try/except``: log warning, continue training
    (eval is never allowed to kill the run).

Strict additive — only registered when
``cfg.callbacks.hh_eval.enabled = True``.
"""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4-panel viz constants (mirrors eval/humanoid/3visualize.py)
# ---------------------------------------------------------------------------
_VIZ_H, _VIZ_W = 720, 1280
_VIZ_SKIP = 3
_R_ALPHA = 0.8
_HUMAN_C = [int(c * _R_ALPHA + 255 * (1.0 - _R_ALPHA)) for c in [8, 76, 97]] + [255]
_KIN_C = [219, 80, 74, 255]
_QPOS_C = [255, 215, 0, 255]
_PHY_C = [56, 142, 60, 255]
_BG_C = [255, 255, 255, 255]
_GD_C = [128, 128, 128, 255]
_SAP_L = 1.0


def _ensure_repo_root_on_path() -> str:
    """Add repo root to sys.path so ``eval.retargeting_new.scripts`` etc. import.

    Resolves: src/umm/callbacks/wmm_hh_eval.py -> parents[3] = repo root.
    """
    repo_root = str(Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


# ---------------------------------------------------------------------------
# Camera + alignment helpers (ported from eval/humanoid/3visualize.py)
# ---------------------------------------------------------------------------

def _get_viz_T_wc(stand_at, look_at) -> torch.Tensor:
    """Compute camera SE(3) (world-from-camera) from stand/look points."""
    from torch import nn
    viz_from = torch.as_tensor(stand_at, dtype=torch.float32)
    viz_to = torch.as_tensor(look_at, dtype=torch.float32)
    z = nn.functional.normalize(viz_to - viz_from, dim=-1)
    down = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    x = torch.cross(down, z)
    y = torch.cross(z, x)
    R_wc = torch.stack([x, y, z], dim=1)  # (3, 3)
    T_wc = torch.eye(4, dtype=torch.float32)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = viz_from
    return T_wc


def _align_phy_to_kin(phy_Q: torch.Tensor, kin_Q: torch.Tensor) -> torch.Tensor:
    """Align mujoco-rolled trajectory to kinematic frame (matches metrics.py).

    g1i.FK returns CUDA tensors when device='cuda:0', but downstream uses
    np.arctan2 + euler2mat (CPU-only). Move both to CPU at entry; output is
    CPU. Callers re-upload if they need GPU; here it's only used for metric
    + viz (both CPU).
    """
    from transforms3d.euler import euler2mat
    phy_Q = phy_Q.detach().cpu().clone() if torch.is_tensor(phy_Q) else torch.as_tensor(phy_Q).clone()
    kin_Q = kin_Q.detach().cpu() if torch.is_tensor(kin_Q) else torch.as_tensor(kin_Q)
    phy_center = phy_Q[0, 0, :2, 3].clone()
    phy_Q[:, :, :2, 3] = phy_Q[:, :, :2, 3] - phy_center[None, None, :]
    T_align = torch.eye(4)
    ref_forward = kin_Q[0, 0, :2, 0]
    phy_forward = phy_Q[0, 0, :2, 0]
    angle = float(np.arctan2(float(ref_forward[1]), float(ref_forward[0]))
                  - np.arctan2(float(phy_forward[1]), float(phy_forward[0])))
    R_align = euler2mat(0.0, 0.0, angle, "sxyz")
    T_align[:3, :3] = torch.from_numpy(R_align).float()
    phy_Q = torch.einsum("ij,...jk->...ik", T_align, phy_Q)
    exe_center = kin_Q[0, 0, :2, 3].clone()
    phy_Q[:, :, :2, 3] = phy_Q[:, :, :2, 3] + exe_center[None, None, :]
    return phy_Q


# ---------------------------------------------------------------------------
# Metric helpers (ported from eval/retargeting_new/scripts/metrics.py)
# ---------------------------------------------------------------------------

def _fail_detect(
    human_T: torch.Tensor,
    phy_T: torch.Tensor,
    fail_N: int = 5,
    human_up_deg_th: float = 45.0,
    robot_down_deg_th: float = 45.0,
    robot_down_height_th: float = 0.2,
) -> np.ndarray:
    """Boolean (T,) — True = standing (succ at frame t)."""
    # Move to CPU before any numpy ops — g1i.FK / SMPL-X HumanModel may
    # return CUDA tensors and `tensor @ np.array(...)` raises
    # "can't convert cuda:0 device type tensor to numpy" otherwise.
    human_up_vec = human_T[:, :3, 1].detach().cpu().numpy() if torch.is_tensor(human_T) else np.asarray(human_T[:, :3, 1])
    phy_up_vec = phy_T[:, :3, 2].detach().cpu().numpy() if torch.is_tensor(phy_T) else np.asarray(phy_T[:, :3, 2])
    phy_height = phy_T[:, 2, 3].detach().cpu().numpy() if torch.is_tensor(phy_T) else np.asarray(phy_T[:, 2, 3])
    human_up_deg = np.arccos(human_up_vec @ np.array([0.0, 0.0, 1.0])) * 180.0 / np.pi
    phy_up_deg = np.arccos(phy_up_vec @ np.array([0.0, 0.0, 1.0])) * 180.0 / np.pi
    human_up = human_up_deg < human_up_deg_th
    phy_down = (phy_up_deg > robot_down_deg_th) & (phy_height < robot_down_height_th)
    cnt = 0
    standing = []
    for i in range(len(human_up)):
        if bool(human_up[i]) and bool(phy_down[i]):
            cnt += 1
        else:
            cnt = 0
        if cnt >= fail_N:
            standing.append(False)
            break
        else:
            standing.append(True)
    standing = standing + [False] * (len(human_up) - len(standing))
    return np.array(standing)


def _keypoint_track_err(
    kin_T: torch.Tensor, phy_T: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (global_err, local_err) — both shape (T, P)."""
    global_err = (kin_T[..., :3, 3] - phy_T[..., :3, 3]).norm(dim=-1)
    kin_lw = torch.linalg.inv(kin_T[:, 0])  # (T, 4, 4)
    phy_lw = torch.linalg.inv(phy_T[:, 0])
    kin_local = torch.einsum("tij,tnjk->tnik", kin_lw, kin_T)
    phy_local = torch.einsum("tij,tnjk->tnik", phy_lw, phy_T)
    local_err = (kin_local[..., :3, 3] - phy_local[..., :3, 3]).norm(dim=-1)
    return global_err, local_err


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------

class WmmHhEvalCallback(pl.Callback):
    """On-the-fly humanoid retargeting eval for w1.0_hh.

    Args:
        val_every_n_iter: gradient-step interval between evals.
        log_dir: output root for ``eval_onfly/step_<N>/`` artifacts.
        test_split_fn: path to ``full_test.json`` (list of (sid, length)).
        human_data_root: directory of frozen GT human npz files
            (default ``logs/eval/humanoid/amass_human_test``).
        smplx_dir: SMPL-X model directory (``data/smpls`` typically).
        humanoid_mesh_fn: G1 link-mesh npz used by 4-panel viz.
        urdf_fn: G1 URDF for ``G1Interface``.
        debug_threshold: ``val_every_n_iter < debug_threshold`` → debug branch.
        n_seqs_prod / n_seqs_debug: number of sequences sampled per branch.
        ctx_len / window_len / window_overlap / gen_steps: inference knobs.
        humanoid_type: e.g. ``"g1"``.
        human_type: must be ``"smplx"``.
        cond_step: condition timestep (0 = clean).
        viz_skip: stride for the 4-panel video frames.
        mjc_threads: parallel mujoco rollout threads.
        viz_stand_at_offset / viz_look_z: camera config.
        gpu: target GPU index (informational; pl_module.device wins).
    """

    def __init__(
        self,
        val_every_n_iter: int,
        log_dir: str,
        test_split_fn: str,
        human_data_root: str,
        smplx_dir: str,
        humanoid_mesh_fn: str,
        urdf_fn: str,
        debug_threshold: int = 100,
        n_seqs_prod: int = 10,
        n_seqs_debug: int = 2,
        sampling_stride: int = 201,
        ctx_len: int = 9,
        window_len: int = 30,
        window_overlap: int = 5,
        gen_steps: int = 10,
        humanoid_type: str = "g1",
        human_type: str = "smplx",
        cond_step: int = 0,
        viz_skip: int = 3,
        mjc_threads: int = 16,
        gpu: int = 0,
        max_budget_s: float = 1800.0,
    ):
        super().__init__()
        self._val_every_n_iter = int(val_every_n_iter)
        self._debug_threshold = int(debug_threshold)
        self._log_dir = str(Path(log_dir).resolve())
        self._test_split_fn = str(Path(test_split_fn).resolve())
        self._human_data_root = str(Path(human_data_root).resolve())
        self._smplx_dir = str(Path(smplx_dir).resolve())
        self._humanoid_mesh_fn = str(Path(humanoid_mesh_fn).resolve())
        self._urdf_fn = str(Path(urdf_fn).resolve())
        self._n_seqs_prod = int(n_seqs_prod)
        self._n_seqs_debug = int(n_seqs_debug)
        self._sampling_stride = int(sampling_stride)
        self._ctx_len = int(ctx_len)
        self._window_len = int(window_len)
        self._window_overlap = int(window_overlap)
        self._gen_steps = int(gen_steps)
        self._humanoid_type = humanoid_type
        self._human_type = human_type
        self._cond_step = int(cond_step)
        self._viz_skip = int(viz_skip)
        self._mjc_threads = int(mjc_threads)
        self._gpu = int(gpu)
        self._max_budget_s = float(max_budget_s)

        Path(self._log_dir).mkdir(parents=True, exist_ok=True)

        # Lazy: G1Interface + HumanModel + RobotMesh (initialized on first
        # eval to avoid paying the FK setup cost when the callback never
        # fires — e.g. during smoke tests).
        self._g1i: Any = None
        self._hm: Any = None
        self._robot_mesh: Any = None

        # Cache the parsed test split.
        self._test_split: Optional[List[Tuple[str, int]]] = None

    # ------------------------------------------------------------------
    # PL hook
    # ------------------------------------------------------------------

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Fire eval at ``global_step % val_every_n_iter == 0``."""
        step = trainer.global_step
        if step == 0 or (step % self._val_every_n_iter) != 0:
            return

        _LOG.info("WmmHhEvalCallback: starting eval at step=%d", step)
        t0 = time.time()
        try:
            self._run_eval(pl_module, step, trainer)
        except Exception as exc:  # pylint: disable=broad-except
            fail_path = osp.join(
                self._log_dir, f"hh_eval_FAILED_step{step}.txt"
            )
            with open(fail_path, "w") as fh:
                fh.write(
                    f"step={step}\nhost={socket.gethostname()}\n\n"
                )
                traceback.print_exc(file=fh)
            _LOG.error(
                "WmmHhEvalCallback: eval at step=%d FAILED — see %s. "
                "Continuing training. Error: %s",
                step, fail_path, exc, exc_info=True,
            )
            return

        wall = time.time() - t0
        if wall > self._max_budget_s:
            _LOG.warning(
                "WmmHhEvalCallback: eval at step=%d took %.1fs > budget %.1fs",
                step, wall, self._max_budget_s,
            )

    # ------------------------------------------------------------------
    # Branch dispatcher
    # ------------------------------------------------------------------

    def _run_eval(
        self, pl_module: pl.LightningModule, step: int, trainer: pl.Trainer
    ) -> None:
        is_debug = self._val_every_n_iter < self._debug_threshold
        out_root = osp.join(self._log_dir, f"step_{step}")
        os.makedirs(out_root, exist_ok=True)

        # Pick which agent: prefer EMA model if available (matches viz path).
        agent = pl_module.model
        if getattr(pl_module, "use_ema", False):
            ema_model = getattr(pl_module, "ema_model", None)
            if ema_model is not None:
                # ctx_record_buffer is a plain dict[uid_tuple → deque],
                # NOT a registered nn.Module buffer (deques aren't
                # tensors). EMA's sync_model_buffers only syncs
                # registered buffers, so the ema_model.ctx_record_buffer
                # never receives ctx history → on-the-fly inference would
                # silently fall back to identity ctx and produce wrong
                # metrics. Patch by reference-share before each fire.
                # Reference-share is safe: this callback is synchronous,
                # eval is hh_joint_infer-only (no train-side mutation).
                if getattr(pl_module.model, "ctx_record_flag", False):
                    ema_model.ctx_record_buffer = (
                        pl_module.model.ctx_record_buffer
                    )
                agent = ema_model

        was_training = agent.training
        agent.eval()
        try:
            if is_debug:
                metrics = self._run_debug(agent, step, out_root, pl_module)
            else:
                metrics = self._run_production(
                    agent, step, out_root, pl_module
                )
        finally:
            if was_training:
                agent.train()

        # Write summary.json
        summary_path = osp.join(out_root, "summary.json")
        with open(summary_path, "w") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        _LOG.info(
            "WmmHhEvalCallback: step=%d wrote summary to %s",
            step, summary_path,
        )

        # Log scalars to wandb
        scalar_metrics: Dict[str, float] = {}
        for k in ("SR-seq", "gerr", "lerr", "wall_time_s"):
            v = metrics.get(k, None)
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and (v != v)  # NaN
            ):
                scalar_metrics[f"hh_eval/{k}"] = float(v)
        if scalar_metrics:
            try:
                pl_module.log_dict(
                    scalar_metrics,
                    on_step=True, on_epoch=False,
                    sync_dist=False, rank_zero_only=True,
                    batch_size=1,
                )
                _LOG.info(
                    "WmmHhEvalCallback: logged %d scalars to wandb",
                    len(scalar_metrics),
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning("WmmHhEvalCallback: scalar log failed: %s", exc)

    # ------------------------------------------------------------------
    # Debug branch — fast smoke test, no mujoco / no viz
    # ------------------------------------------------------------------

    def _run_debug(
        self,
        agent,
        step: int,
        out_root: str,
        pl_module: pl.LightningModule,
    ) -> Dict[str, Any]:
        """Run 2 sequences end-to-end (inference only). Verify no NaN."""
        t_start = time.time()
        sids = self._select_sids(self._n_seqs_debug)
        device = pl_module.device

        self._lazy_init_models(device)

        ok = 0
        for sid, _slen in sids:
            try:
                human_Q = self._build_human_q(sid)
                human_Q = human_Q.to(device)
                T_len = human_Q.shape[0]
                timestep = (
                    torch.arange(T_len, device=device).float()[None] / 30.0
                )
                with torch.no_grad():
                    gen_human_Q, gen_robot_Q, _ = agent.hh_joint_infer(
                        ctx_len=self._ctx_len,
                        windows_size=self._window_len,
                        window_overlap=self._window_overlap,
                        human_Q=human_Q,
                        human_time=timestep,
                        gen_steps=self._gen_steps,
                        condition_timestep=self._cond_step,
                        humanoid_type=self._humanoid_type,
                        human_type=self._human_type,
                        device=device,
                        verbose=False,
                        use_boost=False,
                    )
                if not torch.isfinite(gen_robot_Q).all():
                    raise ValueError(f"NaN/Inf in gen_robot_Q for sid={sid}")
                ok += 1
                _LOG.info(
                    "WmmHhEvalCallback DEBUG: sid=%s OK shape=%s",
                    sid, tuple(gen_robot_Q.shape),
                )
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "WmmHhEvalCallback DEBUG: sid=%s FAILED: %s", sid, exc,
                )

        wall = time.time() - t_start
        return {
            "branch": "debug",
            "step": step,
            "sids": [s for s, _ in sids],
            "n_ok": ok,
            "n_total": len(sids),
            "wall_time_s": wall,
        }

    # ------------------------------------------------------------------
    # Production branch — full pipeline
    # ------------------------------------------------------------------

    def _run_production(
        self,
        agent,
        step: int,
        out_root: str,
        pl_module: pl.LightningModule,
    ) -> Dict[str, Any]:
        """10 evenly-sampled sids, full kin → mjc → metrics → 4-panel video."""
        t_start = time.time()
        sids = self._select_sids(self._n_seqs_prod)
        device = pl_module.device

        kin_dir = osp.join(out_root, "kin")
        mjc_dir = osp.join(out_root, "mjc_rollout")
        viz_dir = osp.join(out_root, "viz")
        os.makedirs(kin_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)

        self._lazy_init_models(device)

        # Step 1-4: inference + IK refinement, save kin npz.
        ok_kin: List[str] = []
        for sid, _slen in sids:
            try:
                self._infer_and_save_kin(
                    agent, sid, device, kin_dir,
                )
                ok_kin.append(sid)
            except Exception as exc:  # noqa: BLE001
                _LOG.warning(
                    "WmmHhEvalCallback PROD: kin sid=%s FAILED: %s",
                    sid, exc,
                )

        # Step 6: spawn mujoco rollout subprocess.
        self._spawn_mujoco_rollout(kin_dir, mjc_dir)

        # Step 7-9: metrics + viz per sid.
        per_sid_metrics: Dict[str, Dict[str, float]] = {}
        for sid in ok_kin:
            try:
                m = self._score_and_viz(
                    sid, kin_dir, mjc_dir, viz_dir,
                )
                per_sid_metrics[sid] = m
            except Exception as exc:  # noqa: BLE001
                import traceback
                _LOG.warning(
                    "WmmHhEvalCallback PROD: score/viz sid=%s FAILED: %s\n%s",
                    sid, exc, traceback.format_exc(),
                )

        # Aggregate
        agg: Dict[str, float] = {}
        for k in ("SR-seq", "gerr", "lerr"):
            vals = [m[k] for m in per_sid_metrics.values() if k in m and not np.isnan(m[k])]
            agg[k] = float(np.mean(vals)) if vals else float("nan")

        # Wandb video uploads
        try:
            self._maybe_log_videos(pl_module, viz_dir, ok_kin, step)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("WmmHhEvalCallback: wandb video log failed: %s", exc)

        wall = time.time() - t_start
        return {
            "branch": "production",
            "step": step,
            "sids": [s for s, _ in sids],
            "ok_kin": ok_kin,
            "per_sid": per_sid_metrics,
            "wall_time_s": wall,
            **agg,
        }

    # ------------------------------------------------------------------
    # Inference + IK refinement (one sid)
    # ------------------------------------------------------------------

    def _infer_and_save_kin(
        self,
        agent,
        sid: str,
        device,
        kin_dir: str,
    ) -> None:
        """Run hh_joint_infer + G1I.RK + adjust, save kin npz."""
        from umm.datasets.wmm.dataset_config import G1_LINK_NAMES

        human_Q = self._build_human_q(sid).to(device)
        T_len = human_Q.shape[0]
        timestep = (
            torch.arange(T_len, device=device).float()[None] / 30.0
        )
        t0 = time.time()
        with torch.no_grad():
            gen_human_Q, gen_robot_Q, _ = agent.hh_joint_infer(
                ctx_len=self._ctx_len,
                windows_size=self._window_len,
                window_overlap=self._window_overlap,
                human_Q=human_Q,
                human_time=timestep,
                gen_steps=self._gen_steps,
                condition_timestep=self._cond_step,
                humanoid_type=self._humanoid_type,
                human_type=self._human_type,
                device=device,
                verbose=False,
                use_boost=False,
            )
        gen_robot_Q = gen_robot_Q.float()
        gen_human_Q = gen_human_Q.float()

        # FK + IK refinement (matches 1infer_ours.py)
        Q_in = torch.cat([gen_robot_Q[:, :1], gen_robot_Q], 1)
        root_pos, root_quat_xyzw, dof_pos = self._g1i.RK(
            Q_in, names=G1_LINK_NAMES,
        )
        dof_pos, _ = self._g1i.adjust(
            Q_in,
            names=G1_LINK_NAMES,
            joint_angles=dof_pos.to(device),
            lrs=[1.0, 0.1, 0.01],
            profile_flag=False,
        )
        run_time = time.time() - t0
        out_fn = osp.join(kin_dir, f"{sid}.npz")
        np.savez(
            out_fn,
            root_pos=root_pos.cpu().numpy().astype(np.float32),
            root_quat_xyzw=root_quat_xyzw.cpu().numpy().astype(np.float32),
            dof_pos=dof_pos.cpu().numpy().astype(np.float32),
            run_time=np.float64(run_time),
            fps=np.float64(T_len / max(run_time, 1e-6)),
            ctx_len=np.int64(self._ctx_len),
            window_size=np.int64(self._window_len),
            window_overlap=np.int64(self._window_overlap),
            gen_steps=np.int64(self._gen_steps),
            input_human_Q=human_Q.cpu().numpy().astype(np.float16),
            output_robot_Q=gen_robot_Q.cpu().numpy().astype(np.float16),
            output_human_Q=gen_human_Q.cpu().numpy().astype(np.float16),
        )

    # ------------------------------------------------------------------
    # Mujoco rollout subprocess
    # ------------------------------------------------------------------

    def _spawn_mujoco_rollout(self, kin_dir: str, mjc_dir: str) -> None:
        """Run scripts/2mujoco_rollout.py as a subprocess on the kin dir."""
        repo_root = _ensure_repo_root_on_path()
        script = osp.join(
            repo_root,
            "eval", "retargeting_new", "scripts", "2mujoco_rollout.py",
        )
        if not osp.exists(script):
            raise FileNotFoundError(f"mujoco rollout script missing: {script}")
        os.makedirs(mjc_dir, exist_ok=True)
        cmd = [
            sys.executable, script,
            "--src", kin_dir,
            "--dst", mjc_dir,
            "--num_rollouts", "1",
            "--thread", str(self._mjc_threads),
        ]
        _LOG.info("WmmHhEvalCallback: spawning mujoco: %s", " ".join(cmd))
        # Inherit env but ensure CUDA isn't grabbed by mujoco subproc
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""  # mujoco rollout is CPU-only
        completed = subprocess.run(
            cmd, env=env, cwd=repo_root,
            capture_output=True, text=True, timeout=self._max_budget_s,
        )
        if completed.returncode != 0:
            _LOG.warning(
                "WmmHhEvalCallback: mujoco rollout returned non-zero "
                "(rc=%d). stderr tail:\n%s",
                completed.returncode, completed.stderr[-2000:],
            )

    # ------------------------------------------------------------------
    # Per-sid metrics + 4-panel video
    # ------------------------------------------------------------------

    def _score_and_viz(
        self,
        sid: str,
        kin_dir: str,
        mjc_dir: str,
        viz_dir: str,
    ) -> Dict[str, float]:
        """Compute SR-seq / gerr / lerr and render 4-panel video for one sid."""
        repo_root = _ensure_repo_root_on_path()
        from umm.datasets.wmm.dataset_config import G1_LINK_NAMES
        kin_fn = osp.join(kin_dir, f"{sid}.npz")
        mjc_fn = osp.join(mjc_dir, f"{sid}.npz")
        human_fn = osp.join(self._human_data_root, f"{sid}.npz")
        if not (osp.exists(kin_fn) and osp.exists(mjc_fn)
                and osp.exists(human_fn)):
            raise FileNotFoundError(
                f"missing one of: kin={osp.exists(kin_fn)}, "
                f"mjc={osp.exists(mjc_fn)}, human={osp.exists(human_fn)}"
            )

        kin_data = np.load(kin_fn)
        mjc_data = np.load(mjc_fn)
        human_data = np.load(human_fn)

        # Build kinematic Q (FK from saved root_pos / quat / dof_pos).
        kin_quat = (
            kin_data["root_quat_xyzw"]
            if "root_quat_xyzw" in kin_data.files
            else kin_data["root_rot"]
        ).astype(np.float32)
        kin_Q = self._g1i.FK(
            root_pos=kin_data["root_pos"].astype(np.float32),
            root_quat_xyzw=kin_quat,
            dof_pos=kin_data["dof_pos"].astype(np.float32),
            names=self._g1i.link_names,
        )[:, 1:]  # drop root joint (matches eval.py)

        phy_Q = self._g1i.FK(
            root_pos=mjc_data["root_pos"][0].astype(np.float32),
            root_quat_xyzw=mjc_data["root_rot"][0].astype(np.float32),
            dof_pos=mjc_data["dof_pos"][0].astype(np.float32),
            names=self._g1i.link_names,
        )[:, 1:]
        # g1i.FK returns CUDA tensors (device='cuda:0'); all downstream
        # metric + viz code uses CPU numpy / matplotlib — move once here.
        if torch.is_tensor(kin_Q):
            kin_Q = kin_Q.detach().cpu()
        if torch.is_tensor(phy_Q):
            phy_Q = phy_Q.detach().cpu()
        phy_Q = _align_phy_to_kin(phy_Q, kin_Q)

        # Human FK for SR judgment.
        beta = human_data["betas"][0][0].astype(np.float32) \
            if human_data["betas"].ndim >= 3 else human_data["betas"][0].astype(np.float32)
        global_t = human_data["global_t"][0].astype(np.float32)
        pose = human_data["pose"][0].astype(np.float32)
        human_Q_fk, human_V, human_F, _ = self._hm.forward(
            shape_params=beta, pose_params=pose, transl=global_t,
        )

        standing = _fail_detect(human_Q_fk[:, 0], phy_Q[:, 0])
        gerr_all, lerr_all = _keypoint_track_err(kin_Q, phy_Q)
        # Move metrics to numpy CPU so we can index by `standing` (np.bool array).
        if torch.is_tensor(gerr_all):
            gerr_all = gerr_all.detach().cpu().numpy()
        if torch.is_tensor(lerr_all):
            lerr_all = lerr_all.detach().cpu().numpy()

        out: Dict[str, float] = {
            "SR-seq": float(bool(standing.all())),
            "gerr": float(gerr_all[standing].mean()) if standing.any()
                    else float("nan"),
            "lerr": float(lerr_all[standing].mean()) if standing.any()
                    else float("nan"),
            "n_frames": int(len(kin_Q)),
        }

        # 4-panel video. We use the same single-camera follow-pelvis layout
        # as eval/humanoid/3visualize.py but stacked into one shot per
        # frame. Panels: input human (blue) / kin robot (red) / qpos
        # projected (yellow, here = same kin since the stored kin already
        # carries the IK-refined dof) / phys (green).
        # Re-run FK with the MESH name list (45 links, subset of g1i.link_names'
        # 52 incl. imu_in_torso/toe/etc.) so kin_Q_mesh / phy_Q_mesh align with
        # self._robot_mesh.names for articulate_mesh's `assert len(T)==len(names)`.
        # The metric kin_Q / phy_Q above use full link_names; viz uses mesh names.
        try:
            # G1Interface.FK returns an EXTRA root joint at index 0
            # (regardless of `names`), so drop it with [:, 1:] to align with
            # `self._robot_mesh.names` (matches debug_1infer_ours.py + 3visualize.py).
            kin_Q_mesh = self._g1i.FK(
                root_pos=kin_data["root_pos"].astype(np.float32),
                root_quat_xyzw=kin_quat,
                dof_pos=kin_data["dof_pos"].astype(np.float32),
                names=self._robot_mesh.names,
            )[:, 1:]
            phy_Q_mesh = self._g1i.FK(
                root_pos=mjc_data["root_pos"][0].astype(np.float32),
                root_quat_xyzw=mjc_data["root_rot"][0].astype(np.float32),
                dof_pos=mjc_data["dof_pos"][0].astype(np.float32),
                names=self._robot_mesh.names,
            )[:, 1:]
            if torch.is_tensor(kin_Q_mesh):
                kin_Q_mesh = kin_Q_mesh.detach().cpu()
            if torch.is_tensor(phy_Q_mesh):
                phy_Q_mesh = phy_Q_mesh.detach().cpu()
            self._render_4panel_video(
                sid=sid,
                viz_fn=osp.join(viz_dir, f"{sid}.mp4"),
                human_V=human_V,
                human_F=human_F,
                kin_Q=kin_Q_mesh,
                phy_Q=phy_Q_mesh,
                human_Q_fk=human_Q_fk,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            _LOG.warning(
                "WmmHhEvalCallback: viz sid=%s FAILED: %s\n%s",
                sid, exc, traceback.format_exc(),
            )
        return out

    def _render_4panel_video(
        self,
        sid: str,
        viz_fn: str,
        human_V: np.ndarray,
        human_F: np.ndarray,
        kin_Q: torch.Tensor,
        phy_Q: torch.Tensor,
        human_Q_fk: torch.Tensor,
    ) -> None:
        """Render and save 4-panel mp4 (single camera following pelvis)."""
        # Optional: viz support is best-effort. If pyrender / mesh
        # rendering deps are missing, skip silently with a warning.
        try:
            import imageio
            import trimesh
            from umm.models.wmm.viz_helpers.mesh_visualizer import visualize_a_mesh
        except Exception as exc:  # noqa: BLE001
            _LOG.warning(
                "WmmHhEvalCallback: viz deps missing (%s); skipping video", exc,
            )
            return

        T_len = len(human_Q_fk)
        frames = []
        # Three panels side-by-side along x-axis.
        offsets = {
            "human": [-_SAP_L, 0.0, 0.0],
            "kin":   [0.0,     0.0, 0.0],
            "qpos":  [_SAP_L,  0.0, 0.0],
            "phy":   [2.0 * _SAP_L, 0.0, 0.0],
        }

        for t in range(0, T_len, self._viz_skip):
            mesh_pieces = []

            # Panel 1 — human (blue, alpha-blended)
            human_mesh = trimesh.Trimesh(
                vertices=human_V[t], faces=human_F, vertex_colors=_HUMAN_C,
            )
            human_mesh.apply_translation(offsets["human"])
            mesh_pieces.append(human_mesh)

            # Panel 2 — kin robot (red)
            kin_mesh = self._robot_mesh.articulate_mesh(
                kin_Q[t], self._robot_mesh.names, _KIN_C, _R_ALPHA,
            )
            kin_mesh.apply_translation(offsets["kin"])
            mesh_pieces.append(kin_mesh)

            # Panel 3 — qpos-projected (yellow)
            # In the on-the-fly callback the kin Q already reflects the
            # IK-refined dof_pos (G1I.adjust saved into kin npz), so the
            # qpos panel is equal to the kin panel modulo color/offset.
            qpos_mesh = self._robot_mesh.articulate_mesh(
                kin_Q[t], self._robot_mesh.names, _QPOS_C, _R_ALPHA,
            )
            qpos_mesh.apply_translation(offsets["qpos"])
            mesh_pieces.append(qpos_mesh)

            # Panel 4 — phys (green)
            phy_mesh = self._robot_mesh.articulate_mesh(
                phy_Q[t], self._robot_mesh.names, _PHY_C, _R_ALPHA,
            )
            phy_mesh.apply_translation(offsets["phy"])
            mesh_pieces.append(phy_mesh)

            combined = trimesh.util.concatenate(mesh_pieces)

            # Camera follows pelvis (single look-at across all 4 panels).
            look_at = human_Q_fk[t, 0, :3, 3].clone().cpu().numpy()
            look_at[2] = float(human_Q_fk[0, 0, 2, 3])
            look_at[0] += _SAP_L  # roughly center the 4-panel strip
            stand_at = look_at + np.array([0.0, -3.0, 0.0])
            stand_at[2] = 2.0
            cam_T = _get_viz_T_wc(stand_at, look_at)

            fr = visualize_a_mesh(
                cam_T_wc=cam_T.cpu().numpy(),
                mesh=combined,
                background_color=_BG_C,
                lighting=True,
                ground_color=_GD_C,
                width=_VIZ_W, height=_VIZ_H,
                ground_size=20, ground_grid_size=40,
            )
            frames.append(fr)

        imageio.mimwrite(
            viz_fn, frames, fps=30 / max(self._viz_skip, 1), codec="h264"
        )

    # ------------------------------------------------------------------
    # Wandb video upload (best effort)
    # ------------------------------------------------------------------

    def _maybe_log_videos(
        self,
        pl_module: pl.LightningModule,
        viz_dir: str,
        sids: List[str],
        step: int,
    ) -> None:
        try:
            import wandb
        except Exception:
            return
        logger = getattr(pl_module, "logger", None)
        wb_logger = getattr(logger, "wandb_logger", None) or logger
        if wb_logger is None or not hasattr(wb_logger, "experiment"):
            return
        for sid in sids:
            mp4 = osp.join(viz_dir, f"{sid}.mp4")
            if osp.exists(mp4):
                try:
                    wb_logger.experiment.log({
                        f"hh_eval/video/{sid}": wandb.Video(mp4, format="mp4"),
                    }, step=step)
                except Exception as exc:  # noqa: BLE001
                    _LOG.warning(
                        "WmmHhEvalCallback: wandb.Video upload sid=%s "
                        "failed: %s", sid, exc,
                    )

    # ------------------------------------------------------------------
    # Helpers — sid selection, model lazy init, human Q build
    # ------------------------------------------------------------------

    def _select_sids(self, n: int) -> List[Tuple[str, int]]:
        """Pick ``n`` sids by even sampling: indices 0, stride, 2*stride, ..."""
        if self._test_split is None:
            with open(self._test_split_fn) as fh:
                self._test_split = [tuple(x) for x in json.load(fh)]
        N = len(self._test_split)
        if n >= N:
            return list(self._test_split)
        idxs = [min(i * self._sampling_stride, N - 1) for i in range(n)]
        # Ensure unique indices (dedup while preserving order)
        seen = set()
        out = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                out.append(self._test_split[i])
        # Fallback: if dedup shrank below n, top up from the front.
        i = 0
        while len(out) < n and i < N:
            if i not in seen:
                seen.add(i)
                out.append(self._test_split[i])
            i += 1
        return out[:n]

    def _lazy_init_models(self, device) -> None:
        """Initialize G1Interface + HumanModel + RobotMesh on first call."""
        repo_root = _ensure_repo_root_on_path()
        if self._g1i is None:
            from lib_twist.g1_interface import G1Interface
            self._g1i = G1Interface(self._urdf_fn, device=str(device))
        if self._hm is None:
            from lib_human.human_model import HumanModel
            self._hm = HumanModel(
                model_path=self._smplx_dir,
                model_type="smplx",
                gender="neutral",
                device="cpu",
                num_betas=300,
            )
        if self._robot_mesh is None:
            self._robot_mesh = _RobotMesh(self._humanoid_mesh_fn)

    def _build_human_q(self, sid: str) -> torch.Tensor:
        """Load human npz + SMPL-X forward → ``(T, 55, 4, 4)`` Q."""
        human_fn = osp.join(self._human_data_root, f"{sid}.npz")
        if not osp.exists(human_fn):
            raise FileNotFoundError(f"human npz missing: {human_fn}")
        data = np.load(human_fn)
        beta = (
            data["betas"][0][0].astype(np.float32)
            if data["betas"].ndim >= 3
            else data["betas"][0].astype(np.float32)
        )
        global_t = data["global_t"][0].astype(np.float32)
        pose = data["pose"][0].astype(np.float32)
        Q, _, _, _ = self._hm.forward(
            shape_params=beta, pose_params=pose, transl=global_t,
        )
        return torch.as_tensor(Q)


# ---------------------------------------------------------------------------
# Robot mesh helper (ported from eval/humanoid/3visualize.py)
# ---------------------------------------------------------------------------

class _RobotMesh:
    """Articulated G1 link-mesh wrapper for 4-panel viz."""

    def __init__(self, mesh_fn: str):
        import trimesh
        self.mesh_fn = mesh_fn
        humanoid_mesh_data = np.load(mesh_fn)
        names = sorted({k[:-2] for k in humanoid_mesh_data.files})
        if "pelvis" in names:
            names.remove("pelvis")
        names = ["pelvis"] + names
        self.names = names
        self.robot_mesh = {}
        for name in names:
            self.robot_mesh[name] = trimesh.Trimesh(
                vertices=humanoid_mesh_data[f"{name}_v"],
                faces=humanoid_mesh_data[f"{name}_f"],
                vertex_colors=(
                    humanoid_mesh_data[f"{name}_c"].astype(np.float32) / 255.0
                ),
            )

    def articulate_mesh(
        self,
        T: torch.Tensor,
        names: List[str],
        color=None,
        blend_alpha: float = 1.0,
    ):
        import trimesh
        from copy import deepcopy as _dc
        assert len(T) == len(names)
        ret = []
        for _T, _n in zip(T, names):
            _m = _dc(self.robot_mesh[_n])
            _T_np = _T.cpu().numpy() if torch.is_tensor(_T) else np.asarray(_T)
            _m.apply_transform(_T_np)
            if color is not None:
                cur = _m.visual.vertex_colors
                if cur.dtype == np.uint8:
                    cur = cur.astype(np.float32) / 255.0
                tgt = np.asarray(color, dtype=np.float32)
                if tgt.max() > 1.0:
                    tgt = tgt / 255.0
                if tgt.shape[0] == 3:
                    tgt = np.concatenate([tgt, [1.0]])
                new = cur[:, :4] * (1.0 - blend_alpha) + tgt[None, :4] * blend_alpha
                _m.visual.vertex_colors = (new * 255).astype(np.uint8)
            ret.append(_m)
        return trimesh.util.concatenate(ret)
