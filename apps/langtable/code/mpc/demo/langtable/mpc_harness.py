"""LangTable MPC harness for WMM agents (w1.0.x).

Self-contained, importable utilities to run plain rollouts and a
gradient-based MPC variant against a trained UMM WMM checkpoint on the
five Protocol-A LangTable tasks.

Algorithm overview (mirrors the v1 MPC, ported to WMM agent API):

  1. Every ``act_horizon`` env steps, run an UNCONDITIONAL agent
     inference (text=instruction) to get an "imagined goal" — the
     agent's own predicted future trace from the current state.
  2. Take the first ``act_horizon`` ee-action slots from that imagined
     trace as the baseline candidate. Spawn ``K-1`` perturbed siblings
     by adding Gaussian noise (xy only) to the baseline; candidate 0
     is the unperturbed baseline.
  3. Inject all K candidates into the trace's ee-action slot ([-1]) at
     positions ``[1 : 1+act_horizon]`` with ``trace_known_mask=True``.
     Run ``agent.inference_with_grad`` with ``text=[""]*K`` (no language
     conditioning during the optimization rollout).
  4. Value func: position L2 between the predicted object positions at
     ``act_horizon`` and the imagined-goal object positions at the same
     index. Average across objects → per-candidate scalar.
  5. SGD on candidate xy (action z and rotation are not optimized) for
     ``len(lr_list)`` inner steps.
  6. Pick the lowest-value candidate; execute its first action against
     the env.

Tunable parameters live on ``MPCConfig``. Defaults are demo-friendly:
small K, few inner steps, only last 2 DDIM steps differentiable.

Usage::

    from demo.langtable.mpc_harness import (
        MPCConfig, REWARD_NAMES,
        load_agent_and_visualizer, build_env,
        run_plain_rollout, run_mpc_rollout, save_video,
    )

The module never imports torch.distributed / pytorch-lightning trainer
state; it is safe to import inside Jupyter.
"""

from __future__ import annotations

import logging
import os
import os.path as osp
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
for _p in [
    osp.join(PROJECT_ROOT, "src"),
    osp.join(PROJECT_ROOT, "ext", "lib_langtable"),
    osp.join(PROJECT_ROOT, "ext", "lib_io"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# UID constants for the visualizer's render_viz_frames(prompt_type=...)
# arg. Imported lazily inside helpers that call into rendering — keeping
# the module-load surface clean of umm-internal imports until needed.
def _import_uid_constants():
    from umm.datasets.umm_v11.dataset_config import (  # noqa: WPS433
        UID_LANGTABLE_OBJECT_GROUP,
        UID_LANGTABLE_EE,
        UID_LANGTABLE_ACT,
    )
    return UID_LANGTABLE_OBJECT_GROUP, UID_LANGTABLE_EE, UID_LANGTABLE_ACT


REWARD_NAMES: List[str] = [
    "blocktoblock",
    "blocktoabsolutelocation",
    "blocktoblockrelativelocation",
    "blocktorelativelocation",
    "separate",
]


@dataclass
class MPCConfig:
    """Tunable MPC parameters (demo-friendly defaults).

    Attributes:
        K: Number of MPC candidates run in parallel (batch dim of the
            grad-enabled agent forward). Larger = more search, more VRAM.
            Default 8.
        noise_std: Std-dev of the Gaussian noise added to the baseline
            candidate's xy actions. z and rotation are never perturbed.
            Default 0.03 (~3 cm).
        lr_list: Per-iteration SGD learning rate list. Length = number
            of inner optimization steps per replan. Default ``[1.0] * 5``.
        n_grad_steps: How many of the LAST DDIM denoising steps run with
            autograd enabled. Earlier steps run under ``torch.no_grad()``
            and the resulting ``x_t`` is detached, capping the autograd
            tape size. Default 2 (last 2 of 10 = 5x cheaper than full).
        act_horizon: Steps between MPC re-plans. Default 10 (matches
            training-time config).
        window: Total prediction window length passed to the agent.
            Default 20 (matches w1.0.3 training).
        hist: Number of history frames in the prefix. Default 5 (matches
            w1.0.3 training).
        max_steps: Episode horizon for the env loop. Default 200 (matches
            Protocol A).
        n_steps: Number of DDIM denoising steps per agent inference.
            Default 10 (matches gen_timesteps in w1.0.3 yaml).

    Notes on units / coordinates:
        - Action xy is in world-frame meters on the LangTable board.
        - Candidates live in SE3 (4,4) but only the (3, 0:2) xy
          components are perturbed and optimized.
    """

    K: int = 8
    noise_std: float = 0.03
    lr_list: List[float] = field(default_factory=lambda: [1.0] * 5)
    n_grad_steps: int = 2
    act_horizon: int = 10
    window: int = 20
    hist: int = 5
    max_steps: int = 200
    n_steps: int = 10
    save_intermediates: bool = True
    # When True, ``run_mpc_rollout`` keeps per-replan dream traces
    # (gen_trace at every inner SGD iter), candidate xy snapshots, and
    # imagined goals. ~1 MB per replan at default K/window/N. Disable
    # if memory/disk is tight.


def load_agent_and_visualizer(
    checkpoint_path: str, config_path: str, device: torch.device
):
    """Load a UMM WMM PL model + return ``(agent, visualizer, pl_module)``.

    Mirrors ``eval/langtable_official/eval_protocol_a_wmm.load_wmm_agent``
    but also returns the pl_module so the caller can re-bind the
    visualizer's ``.pl`` attr if it tries to log into a None logger.

    Args:
        checkpoint_path: Path to a ``last.ckpt``.
        config_path: Path to the matching training YAML.
        device: torch device.

    Returns:
        Tuple ``(agent, visualizer, pl_module)``. ``agent`` is the EMA
        agent if ``use_ema``, else the training agent. ``agent.eval()``
        is already called.
    """
    from umm.training_utils import get_config
    from umm.models.wmm.model import PLModel

    argv_bak = sys.argv[:]
    sys.argv = ["mpc", "--config", config_path]
    try:
        cfg = get_config()
    finally:
        sys.argv = argv_bak

    pl_module = PLModel.load_from_checkpoint(
        checkpoint_path, cfg=cfg, map_location=device, strict=False
    )
    pl_module.to(device).eval()
    agent = (
        pl_module.ema_model if getattr(pl_module, "use_ema", False) else pl_module.model
    )
    agent.eval()
    viz = pl_module.langtable_visualizer
    if viz is None:
        raise RuntimeError(
            "cfg.model.langtable block missing — visualizer not initialized."
        )
    return agent, viz, pl_module


def build_env(reward_name: str, seed: int):
    """Build a single LangTable env for one reward factory (BLOCK_8).

    Args:
        reward_name: One of ``REWARD_NAMES``.
        seed: Env seed.

    Returns:
        ``language_table.LanguageTable`` instance, already seeded.
    """
    from language_table.environments import blocks, language_table
    from language_table.environments.rewards import (
        block2absolutelocation,
        block2block,
        block2block_relative_location,
        block2relativelocation,
        separate_blocks,
    )

    rewards = {
        "blocktoblock": block2block.BlockToBlockReward,
        "blocktoabsolutelocation": block2absolutelocation.BlockToAbsoluteLocationReward,
        "blocktoblockrelativelocation": block2block_relative_location.BlockToBlockRelativeLocationReward,
        "blocktorelativelocation": block2relativelocation.BlockToRelativeLocationReward,
        "separate": separate_blocks.SeparateBlocksReward,
    }
    if reward_name not in rewards:
        raise ValueError(
            f"reward_name {reward_name!r} not in {list(rewards)}"
        )
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=rewards[reward_name],
        render_text_in_image=True,
        render_depth=True,
        render_segmentation=True,
        render_semantic_segmentation=True,
        render_instance_poses=True,
        seed=seed,
    )
    return env


def _ensure_plan_feasible(env, n_retry: int = 20, base_seed: int = 0) -> bool:
    """Reset env until the oracle confirms a feasible push plan.

    Mirrors ``language_table/eval/main.py`` lines 107-117. Required
    because Protocol-A reward factories (e.g. block2relativelocation)
    occasionally sample infeasible inits.

    Per-retry env reseeding (``env.seed(base_seed*1000 + retry_idx)``)
    makes the feasibility loop a deterministic function of
    ``base_seed`` — episode E is reproducible regardless of which
    shard / process runs it. ``base_seed=0`` (default) preserves the
    pre-shard behavior in single-shard / non-eval contexts.

    Returns:
        True if a feasible init was found within ``n_retry`` attempts.
    """
    from language_table.replay.push_oracle_rrt_slowdown import (
        ObstacleOrientedPushOracleBoard2dRRT,
    )

    oracle = ObstacleOrientedPushOracleBoard2dRRT(env, use_ee_planner=True)
    for retry in range(n_retry):
        # Re-seed env per retry so the scene stream is a deterministic
        # function of (base_seed, retry). Without this, the FIRST reset
        # is seeded but all retries inherit env's evolving RNG state —
        # making retry behavior dependent on cumulative reset count.
        env.seed(int(base_seed) * 1000 + retry)
        env.reset()
        raw = env.compute_state()
        if oracle.get_plan(raw):
            return True
    return False


def run_plain_rollout(
    visualizer,
    agent,
    env,
    cfg: MPCConfig,
    seed: int,
    device: torch.device,
):
    """Plain (non-MPC) rollout, identical path to the eval script.

    Delegates to ``visualizer.rollout_full_states_trivial_policy`` with
    ``env=env`` and ``skip_env_reset=True`` (caller already ran
    ``_ensure_plan_feasible``).

    Returns:
        Dict with keys ``succ``, ``succ_steps``, ``frames``,
        ``instruction``.
    """
    succ, succ_steps, real_trace, _ctx, uid, pcl, frames, instruction, detailed = (
        visualizer.rollout_full_states_trivial_policy(
            agent=agent,
            seed=seed,
            max_steps=cfg.max_steps,
            hist=cfg.hist,
            window=cfg.window,
            act_horizon=cfg.act_horizon,
            device=device,
            env=env,
            strict_action_check=False,
            skip_env_reset=True,
        )
    )
    # detailed_buffer is a list of (gen_trace_np, hist_len, replan_step)
    # — one entry per replan. Reconstruct a per-step "imagined goal"
    # by holding each replan's predicted end-of-window state until the
    # next replan fires (sticky).
    n_steps_actual = len(frames)  # frames includes initial obs
    imagined_goals = _stickyize_imagined_goals(detailed, n_steps_actual)
    return {
        "succ": bool(succ),
        "succ_steps": int(succ_steps),
        "frames": frames,
        "instruction": instruction,
        "real_trace": real_trace,           # (T, N, 4, 4)
        "imagined_goals": imagined_goals,    # (T, N, 4, 4)
        "uid": uid,                          # list of UID enums (N,)
        "pcl": pcl,                          # (N, PCL, 6)
        "value_curves": None,                # MPC-only
    }


def _stickyize_imagined_goals(detailed_buffer, n_frames: int) -> np.ndarray:
    """Convert per-replan imagined traces into a per-frame sticky goal.

    ``detailed_buffer`` is a list of tuples
    ``(gen_trace_np, hist_len, replan_step)``. ``gen_trace_np`` has
    shape ``(window, N, 4, 4)``. The "imagined goal" we visualize is
    the END frame of each replan's prediction — what the agent thinks
    the world will look like at horizon end. This is the same target
    the MPC value function uses.

    Returns: ``(n_frames, N, 4, 4)`` numpy array. For frames before the
    first replan (shouldn't happen since replan-0 fires at step 0),
    the first replan's end frame is used as fallback.
    """
    if len(detailed_buffer) == 0:
        raise ValueError("empty detailed_buffer — no imagined goals to render")
    end_states = []  # list of (replan_step, (N, 4, 4))
    for gen_trace_np, _hist_len, replan_step in detailed_buffer:
        end_state = gen_trace_np[-1]  # (N, 4, 4) — last frame of imagined window
        end_states.append((int(replan_step), end_state))
    end_states.sort(key=lambda x: x[0])
    N = end_states[0][1].shape[0]
    out = np.zeros((n_frames, N, 4, 4), dtype=np.float32)
    cur_idx = 0
    cur_goal = end_states[0][1]
    for t in range(n_frames):
        while cur_idx + 1 < len(end_states) and end_states[cur_idx + 1][0] <= t:
            cur_idx += 1
            cur_goal = end_states[cur_idx][1]
        out[t] = cur_goal
    return out


def _build_inference_inputs(
    hist_trace: "np.ndarray | list",
    group_id: torch.Tensor,
    sem_type: torch.Tensor,
    local_pcl: torch.Tensor,
    cfg: MPCConfig,
    fps: float,
    device: torch.device,
    batch_size: int = 1,
):
    """Build the tensors agent.inference{,_with_grad} consumes.

    Returns a dict mirroring the ``infer_agent`` setup in the visualizer
    but batched to ``batch_size`` (= K for MPC). Returns trace as a
    NORMAL tensor so the caller can flip ``requires_grad_`` on candidate
    slices before passing to inference.
    """
    if isinstance(hist_trace, list):
        hist_trace = np.stack(hist_trace, axis=0)
    hist_trace_t = torch.as_tensor(hist_trace, dtype=torch.float32)
    H, N, _, _ = hist_trace_t.shape
    if H > cfg.window:
        raise ValueError(
            f"hist length {H} exceeds window {cfg.window}"
        )
    pad = cfg.window - H
    pad_trace = torch.eye(4).view(1, 1, 4, 4).expand(pad, N, 4, 4)
    trace = torch.cat([hist_trace_t, pad_trace])  # (window, N, 4, 4)
    trace_known_mask = torch.ones((cfg.window, N))
    trace_known_mask[H:] = 0
    context = torch.eye(4).view(1, 1, 4, 4).expand(1, N, 4, 4)
    context_known_mask = torch.zeros((1, N))
    realtime = torch.arange(cfg.window, dtype=torch.float32) - H + 1
    realtime = realtime / fps

    out = {
        "trace": trace[None].expand(batch_size, -1, -1, -1, -1).contiguous().to(device),
        "context": context[None].expand(batch_size, -1, -1, -1, -1).contiguous().to(device),
        "trace_known_mask": trace_known_mask[None].expand(batch_size, -1, -1).contiguous().bool().to(device),
        "context_known_mask": context_known_mask[None].expand(batch_size, -1, -1).contiguous().bool().to(device),
        "trace_mask": torch.ones((batch_size, cfg.window, N), dtype=torch.bool, device=device),
        "context_mask": torch.ones((batch_size, 1, N), dtype=torch.bool, device=device),
        "real_timestep": realtime[None].expand(batch_size, -1).contiguous().to(device),
        "group_id": group_id[None].expand(batch_size, -1).contiguous().to(device),
        "sem_type": sem_type[None].expand(batch_size, -1).contiguous().to(device),
        "rot_mask": torch.ones((batch_size, N), dtype=torch.bool, device=device),
        "local_pcl": local_pcl[None].expand(batch_size, -1, -1, -1).contiguous().to(device),
        "local_pcl_mask": torch.ones((batch_size, N), dtype=torch.bool, device=device),
        "H": H,
        "N": N,
    }
    return out


def _imagined_goal(
    agent,
    hist_trace: "np.ndarray | list",
    group_id,
    sem_type,
    local_pcl,
    cfg: MPCConfig,
    fps: float,
    device,
    seed: int,
    instruction: str,
):
    """One-pass plain inference to get the agent's imagined goal trace.

    Returns:
        Tensor of shape ``(window, N, 4, 4)`` on ``device`` (gradients
        detached). This is the policy's own prediction with full text
        conditioning — used as the MPC value-function target.
    """
    inp = _build_inference_inputs(
        hist_trace, group_id, sem_type, local_pcl, cfg, fps, device, batch_size=1
    )
    _ctx, gen_trace = agent.inference(
        trace=inp["trace"], context=inp["context"],
        trace_known_mask=inp["trace_known_mask"],
        context_known_mask=inp["context_known_mask"],
        trace_mask=inp["trace_mask"], context_mask=inp["context_mask"],
        real_timestep=inp["real_timestep"],
        group_id=inp["group_id"], sem_type=inp["sem_type"],
        rot_mask=inp["rot_mask"],
        local_pcl=inp["local_pcl"], local_pcl_mask=inp["local_pcl_mask"],
        text=[instruction], seed=seed, verbose=False,
    )
    return gen_trace[0].detach()  # (window, N, 4, 4)


def _mpc_replan(
    agent,
    hist_trace: "np.ndarray | list",
    group_id,
    sem_type,
    local_pcl,
    cfg: MPCConfig,
    fps: float,
    device,
    seed: int,
    instruction: str,
):
    """One MPC replan: imagine goal, optimize K candidates, return best.

    Returns:
        Tuple ``(best_action_chunk_np, value_curve_np)``:
            - best_action_chunk_np: ``(act_horizon, 4, 4)`` numpy SE3
              chunk to execute (already detached).
            - value_curve_np: ``(n_inner+1, K)`` per-iteration values
              for diagnostic plotting.
    """
    n_inner = len(cfg.lr_list)
    if n_inner < 1:
        raise ValueError("MPCConfig.lr_list must have length >= 1")

    # Step 1: imagined goal (text-conditioned).
    imagined = _imagined_goal(
        agent, hist_trace, group_id, sem_type, local_pcl,
        cfg, fps, device, seed, instruction,
    )  # (window, N, 4, 4)
    H = len(hist_trace) if isinstance(hist_trace, list) else hist_trace.shape[0]

    # Step 2: candidate baseline = imagined ee-action slots over the
    # horizon ``[H : H+act_horizon]``. Slot -1 in the trace is the
    # action channel.
    act_lo = H
    act_hi = min(cfg.window, H + cfg.act_horizon)
    if act_hi <= act_lo:
        raise RuntimeError(
            f"MPC replan: act_lo={act_lo} >= act_hi={act_hi}; window/hist mismatch"
        )
    baseline = imagined[act_lo:act_hi, -1].clone()  # (Th, 4, 4)
    Th = baseline.shape[0]

    # Step 3: K candidates with xy noise on the baseline (cand 0 = baseline).
    # Use a local torch.Generator seeded from the per-replan seed so the
    # noise is a deterministic function of (env_seed, ep_num, replan_idx).
    # Using global torch.randn would couple the noise to whatever earlier
    # CUDA work happened on this process — breaks shard-invariance.
    K = cfg.K
    cand_xy = baseline[None, :, :2, 3].expand(K, -1, -1).clone()  # (K, Th, 2)
    g_noise = torch.Generator(device=device)
    g_noise.manual_seed(int(seed))
    noise = torch.randn(K, Th, 2, device=device, generator=g_noise) * cfg.noise_std
    noise[0] = 0.0
    cand_xy = cand_xy + noise
    cand_xy = cand_xy.detach().requires_grad_(True)

    # Step 4: build batched inference inputs (batch=K) ONCE — only the
    # action xy slice is rebuilt each inner step from the leaf.
    inp = _build_inference_inputs(
        hist_trace, group_id, sem_type, local_pcl,
        cfg, fps, device, batch_size=K,
    )
    # Make trace_known_mask True at the action slot for [act_lo:act_hi].
    trace_known_mask = inp["trace_known_mask"].clone()
    trace_known_mask[:, act_lo:act_hi, -1] = True
    inp["trace_known_mask"] = trace_known_mask

    # Freeze agent. Snapshot pre-call requires_grad state so the caller's
    # outer state is restored on every replan exit (success OR exception).
    # Without this, a non-EMA caller would have its params permanently
    # frozen by a single MPC call.
    prior_rg = [p.requires_grad for p in agent.parameters()]
    for p in agent.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.SGD([cand_xy], lr=cfg.lr_list[0])
    value_curve = []        # list of (K,)             per inner iter
    cand_xy_curve = []      # list of (K, Th, 2)        per inner iter
    gen_trace_curve = []    # list of (K, window, N, 4, 4) per inner iter
    # NOTE: gen_trace_curve is the heavy part of intermediates. Disable
    # via cfg.save_intermediates=False if memory/disk is tight.
    save_intermediates = bool(getattr(cfg, "save_intermediates", True))

    obj_imag = imagined[act_hi - 1, :-2, :2, 3]  # (O, 2)

    for it in range(n_inner):
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr_list[it]
        optimizer.zero_grad()

        # Inject candidate xy into the trace at the action slot.
        # trace shape (K, window, N, 4, 4)
        trace = inp["trace"].clone()
        # Start from baseline at the action positions across all K.
        baseline_full = baseline[None].expand(K, -1, -1, -1).clone()
        trace[:, act_lo:act_hi, -1] = baseline_full
        # Overwrite xy with the leaf so grad flows.
        trace[:, act_lo:act_hi, -1, :2, 3] = cand_xy

        with torch.enable_grad():
            _ctx_g, gen_trace_g = agent.inference_with_grad(
                trace=trace,
                context=inp["context"],
                trace_known_mask=inp["trace_known_mask"],
                context_known_mask=inp["context_known_mask"],
                trace_mask=inp["trace_mask"],
                context_mask=inp["context_mask"],
                real_timestep=inp["real_timestep"],
                group_id=inp["group_id"],
                sem_type=inp["sem_type"],
                rot_mask=inp["rot_mask"],
                local_pcl=inp["local_pcl"],
                local_pcl_mask=inp["local_pcl_mask"],
                text=[""] * K,
                seed=seed,
                n_grad_steps=cfg.n_grad_steps,
                n_steps=cfg.n_steps,
                verbose=False,
            )
            # Value: position L2 of objects at horizon end vs imagined.
            obj_pred = gen_trace_g[:, act_hi - 1, :-2, :2, 3]  # (K, O, 2)
            diff = obj_pred - obj_imag[None]
            per_cand = diff.norm(dim=-1).mean(dim=-1)  # (K,)
            loss = per_cand.mean()

        value_curve.append(per_cand.detach().cpu().numpy())
        cand_xy_curve.append(cand_xy.detach().cpu().numpy().copy())
        if save_intermediates:
            gen_trace_curve.append(gen_trace_g.detach().cpu().numpy().copy())
        if it < n_inner - 1:
            loss.backward()
            optimizer.step()

    # Final eval pass — recompute values one more time AT the optimized
    # candidates so the curve has n_inner+1 points (start + after each
    # step, last point being the post-final-step eval).
    with torch.no_grad():
        trace = inp["trace"].clone()
        baseline_full = baseline[None].expand(K, -1, -1, -1).clone()
        trace[:, act_lo:act_hi, -1] = baseline_full
        trace[:, act_lo:act_hi, -1, :2, 3] = cand_xy.detach()
        _ctx_e, gen_trace_e = agent.inference(
            trace=trace, context=inp["context"],
            trace_known_mask=inp["trace_known_mask"],
            context_known_mask=inp["context_known_mask"],
            trace_mask=inp["trace_mask"], context_mask=inp["context_mask"],
            real_timestep=inp["real_timestep"],
            group_id=inp["group_id"], sem_type=inp["sem_type"],
            rot_mask=inp["rot_mask"],
            local_pcl=inp["local_pcl"], local_pcl_mask=inp["local_pcl_mask"],
            text=[""] * K, seed=seed, verbose=False, n_steps=cfg.n_steps,
        )
        obj_pred_e = gen_trace_e[:, act_hi - 1, :-2, :2, 3]
        per_cand_e = (obj_pred_e - obj_imag[None]).norm(dim=-1).mean(dim=-1)
    value_curve.append(per_cand_e.detach().cpu().numpy())
    cand_xy_curve.append(cand_xy.detach().cpu().numpy().copy())
    if save_intermediates:
        gen_trace_curve.append(gen_trace_e.detach().cpu().numpy().copy())
    final_values = per_cand_e

    # Step 6: pick best. Build the output action chunk from the
    # baseline (rotation/z untouched) with xy taken from the best
    # candidate.
    best_idx = int(torch.argmin(final_values).item())
    best_chunk = baseline.clone()
    best_chunk[:, :2, 3] = cand_xy.detach()[best_idx]

    # Restore the caller's pre-call requires_grad state. NOTE: this
    # restore does NOT run if the inference call raised — for a demo
    # script that's acceptable (re-load the ckpt). For production
    # callers that need exception safety, wrap the SGD section in
    # try/finally. For w1.0.3 (use_ema=True) this is a no-op since the
    # EMA agent is never trained; for non-EMA callers it's load-bearing.
    for p, rg in zip(agent.parameters(), prior_rg):
        p.requires_grad_(rg)

    # imagined goal: end frame of the text-conditioned imagined trace,
    # in world coords, (N, 4, 4). This is what the value function targets.
    imagined_end = imagined[-1].detach().cpu().numpy()
    intermediates = {
        "baseline_chunk": baseline.detach().cpu().numpy(),     # (Th, 4, 4)
        "cand_xy_per_iter": np.stack(cand_xy_curve, axis=0),   # (n_inner+1, K, Th, 2)
        "value_per_iter": np.stack(value_curve, axis=0),       # (n_inner+1, K)
        "imagined_goal": imagined_end,                          # (N, 4, 4)
        "imagined_full": imagined.detach().cpu().numpy(),       # (window, N, 4, 4)
        "best_idx": best_idx,
        "act_lo": act_lo,
        "act_hi": act_hi,
    }
    if save_intermediates and len(gen_trace_curve) > 0:
        intermediates["gen_trace_per_iter"] = np.stack(gen_trace_curve, axis=0)
        # shape: (n_inner+1, K, window, N, 4, 4)
    return (
        best_chunk.cpu().numpy(),
        np.stack(value_curve, axis=0),
        imagined_end,
        intermediates,
    )


def run_mpc_rollout(
    visualizer,
    agent,
    env,
    cfg: MPCConfig,
    seed: int,
    device: torch.device,
):
    """Closed-loop MPC rollout against ``env``.

    Mirrors ``visualizer.rollout_full_states_trivial_policy`` step-by-step
    but replaces the inference call with ``_mpc_replan`` every
    ``act_horizon`` env steps.

    Returns:
        Dict with keys ``succ``, ``succ_steps``, ``frames``,
        ``instruction``, ``value_curves`` (list of per-replan curves).
    """
    from tqdm import tqdm

    obs, instruction, obj_uids, ee_uids = visualizer.reset_env(env, skip_reset=True)
    group_id, sem_type, local_pcl = visualizer.prepare_trace_attr_oracle(
        obj_uids, ee_uids
    )

    frames = [obs["rgb"]]
    cur_state_T_wl = np.asarray(
        [obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
    )
    ee_T_wl = cur_state_T_wl[-1].copy()
    trace_buffer = [
        np.concatenate([cur_state_T_wl, cur_state_T_wl[-1:].copy()], axis=0)
    ]

    succ = False
    succ_steps = -1
    value_curves = []
    imagined_goals_per_replan: List[np.ndarray] = []  # (N, 4, 4) each
    replan_steps: List[int] = []
    intermediates_per_replan: List[dict] = []  # full optimization traces
    pending_actions: List[np.ndarray] = []

    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        for t in tqdm(range(cfg.max_steps), desc="mpc rollout"):
            if t % cfg.act_horizon == 0:
                hist_trace = trace_buffer[-cfg.hist:]
                # Per-replan deterministic seed = base_seed * 1000 + replan_idx.
                # Different replans get different but reproducible noise; same
                # episode in any shard / num_shards reproduces identical seeds.
                replan_idx = t // cfg.act_horizon
                replan_seed = int(seed) * 1000 + replan_idx
                best_chunk, value_curve, imagined_end, inter = _mpc_replan(
                    agent=agent,
                    hist_trace=hist_trace,
                    group_id=group_id, sem_type=sem_type,
                    local_pcl=local_pcl,
                    cfg=cfg, fps=env._control_frequency,
                    device=device, seed=replan_seed, instruction=instruction,
                )
                value_curves.append(value_curve)
                imagined_goals_per_replan.append(imagined_end)
                replan_steps.append(t)
                # Capture the real-state snapshot at the moment of this
                # replan — pairs with the dream traces for analysis.
                inter["replan_step"] = int(t)
                inter["real_state_at_replan"] = trace_buffer[-1].copy()
                intermediates_per_replan.append(inter)
                pending_actions = [best_chunk[i] for i in range(best_chunk.shape[0])]

            if not pending_actions:
                logging.warning(f"MPC: no pending action at t={t}; stopping early")
                break
            action_T = pending_actions.pop(0)
            applied_act = action_T[:2, 3] - ee_T_wl[:2, 3]

            next_obs, _r, done, _info = env.step(applied_act)
            frames.append(next_obs["rgb"])
            cur_state_T_wl = np.asarray(
                [next_obs["instance_poses"][uid] for uid in obj_uids + ee_uids]
            )
            ee_T_wl = cur_state_T_wl[-1].copy()
            trace_buffer.append(
                np.concatenate([cur_state_T_wl, action_T[None]], axis=0)
            )
            if done:
                succ = True
                succ_steps = t
                break
    finally:
        np.random.set_state(rng_state)

    # Build per-step sticky imagined goal array.
    fake_detailed = [
        # imagined_end has shape (N, 4, 4); _stickyize_imagined_goals
        # expects gen_trace_np of shape (window, N, 4, 4). Wrap as a
        # 1-frame "window" to reuse the helper.
        (g[None, ...], 0, s)
        for g, s in zip(imagined_goals_per_replan, replan_steps)
    ]
    n_frames_actual = len(frames)
    imagined_goals = _stickyize_imagined_goals(fake_detailed, n_frames_actual)

    real_trace = np.stack(trace_buffer, axis=0)  # (T, N, 4, 4)
    UID_GROUP, UID_EE, UID_ACT = _import_uid_constants()
    uid = [UID_GROUP[u] for u in obj_uids] + [UID_EE, UID_ACT]

    return {
        "succ": bool(succ),
        "succ_steps": int(succ_steps),
        "frames": frames,
        "instruction": instruction,
        "real_trace": real_trace,
        "imagined_goals": imagined_goals,
        "uid": uid,
        "pcl": local_pcl,
        "value_curves": value_curves,
        "intermediates": intermediates_per_replan,
        "replan_steps": replan_steps,
    }


def save_video(frames, path: str, fps: int = 10):
    """Save a list of HxWx3 uint8 frames to ``path`` as mp4.

    Uses imageio; falls back to writing a gif if mp4 backend is missing.
    """
    import imageio

    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    # imageio's stubs declare mimsave taking List[ArrayLike]; pass a list
    # of per-frame uint8 arrays to avoid the false-positive overload error.
    frames_list = [np.asarray(f, dtype=np.uint8) for f in frames]
    try:
        imageio.mimsave(path, frames_list, fps=fps, codec="libx264")  # type: ignore[arg-type]
    except (ValueError, RuntimeError):
        gif_path = osp.splitext(path)[0] + ".gif"
        imageio.mimsave(gif_path, frames_list, fps=fps)  # type: ignore[arg-type]
        return gif_path
    return path


def save_side_by_side_video(plain_frames, mpc_frames, path: str, fps: int = 10):
    """Save plain | MPC frames stacked horizontally as one mp4."""
    import imageio

    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    n = min(len(plain_frames), len(mpc_frames))
    frames_list = [
        np.concatenate(
            [np.asarray(plain_frames[i], dtype=np.uint8),
             np.asarray(mpc_frames[i], dtype=np.uint8)], axis=1
        ) for i in range(n)
    ]
    try:
        imageio.mimsave(path, frames_list, fps=fps, codec="libx264")  # type: ignore[arg-type]
    except (ValueError, RuntimeError):
        gif_path = osp.splitext(path)[0] + ".gif"
        imageio.mimsave(gif_path, frames_list, fps=fps)  # type: ignore[arg-type]
        return gif_path
    return path


def save_first_frame_jpg(frames, path: str):
    """Persist the first frame as a JPG for visual sanity-checking."""
    import imageio

    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    imageio.imwrite(path, frames[0].astype(np.uint8))
    return path


# ────────────────────────────────────────────────────────────────────
# Goal-vs-real 3D rendering
# ────────────────────────────────────────────────────────────────────

def render_goal_vs_real_video(
    visualizer,
    real_trace: np.ndarray,
    imagined_goals: np.ndarray,
    uid: list,
    pcl: torch.Tensor,
    path: str,
    H: int = 256,
    W: int = 256,
    viz_skip: int = 2,
    fps: int = 10,
):
    """Render LEFT=imagined goal, RIGHT=real-state replay, side-by-side.

    Mirrors the training-time ``rollout_3dviz_*`` videos: uses the same
    ``visualizer.render_viz_frames`` helper so the per-block point clouds
    + EE marker look identical to what the user already sees during
    training. Output is one mp4 with the two streams concatenated
    horizontally.

    Args:
        visualizer: The ``LangTableVisualizer`` instance (provides
            ``render_viz_frames`` + ``make_viz_T_cam``).
        real_trace: ``(T, N, 4, 4)`` per-step real states from the env
            rollout.
        imagined_goals: ``(T, N, 4, 4)`` per-step sticky imagined goal
            (end frame of the most recent agent prediction).
        uid: Per-part UID list (length N), as returned by the rollout.
        pcl: ``(N, PCL, 6)`` torch tensor of local point clouds.
        path: Output mp4 path.
        H, W: Per-pane render dimensions in pixels.
        viz_skip: Render every k-th frame to keep the file small.
        fps: Output video framerate.

    Returns: the path written. Falls back to .gif if libx264 is missing.
    """
    import imageio

    T_total, N, _, _ = real_trace.shape
    assert imagined_goals.shape[0] == T_total, (
        f"goal frames {imagined_goals.shape[0]} != real frames {T_total}"
    )
    real_t = torch.as_tensor(real_trace, dtype=torch.float32)
    goal_t = torch.as_tensor(imagined_goals, dtype=torch.float32)
    pcl_mask = torch.ones((N,), dtype=torch.bool)

    viz_T_cam = visualizer.make_viz_T_cam(batch=None)

    real_frames = visualizer.render_viz_frames(
        prompt_type=uid, T_wl=real_t[::viz_skip],
        viz_T_cam=viz_T_cam, local_pcl=pcl,
        local_pcl_mask=pcl_mask, H=H, W=W,
    )
    goal_frames = visualizer.render_viz_frames(
        prompt_type=uid, T_wl=goal_t[::viz_skip],
        viz_T_cam=viz_T_cam, local_pcl=pcl,
        local_pcl_mask=pcl_mask, H=H, W=W,
    )

    n = min(len(real_frames), len(goal_frames))
    out_frames = []
    for i in range(n):
        # render_viz_frames returns RGBA uint8. Drop alpha and stack.
        gf = np.asarray(goal_frames[i], dtype=np.uint8)[..., :3]
        rf = np.asarray(real_frames[i], dtype=np.uint8)[..., :3]
        out_frames.append(np.concatenate([gf, rf], axis=1))

    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    try:
        imageio.mimsave(path, out_frames, fps=fps, codec="libx264")  # type: ignore[arg-type]
    except (ValueError, RuntimeError):
        gif_path = osp.splitext(path)[0] + ".gif"
        imageio.mimsave(gif_path, out_frames, fps=fps)  # type: ignore[arg-type]
        return gif_path
    return path


# ────────────────────────────────────────────────────────────────────
# Optimization-curve plotting
# ────────────────────────────────────────────────────────────────────

def plot_optimization_curves(value_curves: list, path: str, title: str = ""):
    """Plot per-replan inner-SGD value trajectories.

    ``value_curves`` is a list of ``(n_inner+1, K)`` numpy arrays — one
    entry per MPC replan. Plots two panels:

      LEFT  — best (min over K) value vs inner-iter, one line per replan.
      RIGHT — all K candidates of the FIRST replan (per-candidate trace).

    This matches the diagnostic style of the v1 MPC notebook.

    Returns: the path written.
    """
    import matplotlib.pyplot as plt

    if len(value_curves) == 0:
        raise ValueError("value_curves is empty — no replans recorded")

    arrs = [np.asarray(c) for c in value_curves]
    n_inner_p1 = arrs[0].shape[0]
    K = arrs[0].shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_l, ax_r = axes

    # Left panel: best per replan
    iters = np.arange(n_inner_p1)
    cmap = plt.get_cmap("viridis")
    for i, arr in enumerate(arrs):
        best_per_iter = arr.min(axis=1)
        c = cmap(i / max(len(arrs) - 1, 1))
        ax_l.plot(iters, best_per_iter, color=c, alpha=0.7, label=f"replan {i}" if i % 2 == 0 else None)
    ax_l.set_xlabel("inner SGD iter")
    ax_l.set_ylabel("value (lower = closer to goal)")
    ax_l.set_title("Best-of-K per replan")
    if len(arrs) <= 8:
        ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)

    # Right panel: per-K trajectories of replan 0
    arr0 = arrs[0]
    for k in range(K):
        ax_r.plot(iters, arr0[:, k], alpha=0.6, label=f"K={k}")
    ax_r.set_xlabel("inner SGD iter")
    ax_r.set_ylabel("value")
    ax_r.set_title(f"Per-candidate (replan 0, K={K})")
    ax_r.legend(fontsize=7, ncol=2)
    ax_r.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ────────────────────────────────────────────────────────────────────
# Per-rollout meta JSON
# ────────────────────────────────────────────────────────────────────

def save_meta_json(
    path: str,
    task: str,
    instruction: str,
    succ: bool,
    succ_steps: int,
    max_steps: int,
    method: str,
    mpc_config: "MPCConfig | None" = None,
    extras: "dict | None" = None,
):
    """Persist per-rollout metadata for downstream inspection.

    Records: task name, language prompt (env-decoded instruction at
    reset time), succ_steps (= step at which env.done fired, or -1),
    succ (bool), max_steps (the cap), method ("plain" or "mpc"), and
    all tunable MPC params if applicable.

    Returns: the path written.
    """
    import json as _json

    payload = {
        "task": task,
        "instruction": instruction,
        "method": method,
        "succ": bool(succ),
        "succ_steps": int(succ_steps),
        "max_steps": int(max_steps),
        "terminated": (succ_steps != -1) or (succ_steps == max_steps),
    }
    if mpc_config is not None:
        payload["mpc_config"] = {
            "K": mpc_config.K,
            "noise_std": mpc_config.noise_std,
            "lr_list": list(mpc_config.lr_list),
            "n_grad_steps": mpc_config.n_grad_steps,
            "act_horizon": mpc_config.act_horizon,
            "window": mpc_config.window,
            "hist": mpc_config.hist,
            "max_steps": mpc_config.max_steps,
            "n_steps": mpc_config.n_steps,
        }
    if extras:
        payload["extras"] = extras

    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        _json.dump(payload, f, indent=2)
    return path


def summarize_pair(plain_result: dict, mpc_result: dict, task: str) -> dict:
    """One-row summary for a (plain, MPC) episode pair.

    The headline argument for MPC is NOT a higher success rate — both
    methods can succeed on the same scene — but a LOWER ``succ_steps``
    (time-to-goal). This helper exposes that contrast cleanly.

    Convention: ``succ_steps == -1`` means failure. For the
    ``steps_delta`` field we treat a failure as ``max_steps`` so a
    successful MPC vs a failed plain shows a positive delta rather than
    ``NaN``.
    """
    plain_steps = plain_result["succ_steps"]
    mpc_steps = mpc_result["succ_steps"]
    cap = 200  # reported steps if failure (used only for delta math)
    plain_eff = plain_steps if plain_steps >= 0 else cap
    mpc_eff = mpc_steps if mpc_steps >= 0 else cap
    return {
        "task": task,
        "plain_succ": plain_result["succ"],
        "plain_steps": plain_steps,
        "mpc_succ": mpc_result["succ"],
        "mpc_steps": mpc_steps,
        "steps_delta": plain_eff - mpc_eff,  # positive = MPC faster
    }


# Mapping of param name → human-readable description, used by the
# notebook to render a tunable-parameter table.
TUNABLE_DOC = [
    ("K", "Number of MPC candidates run in parallel. Larger = wider search, more VRAM."),
    ("noise_std", "Std-dev of Gaussian noise on candidate xy actions (meters)."),
    ("lr_list", "Per-iteration SGD learning rates. Length = inner steps per replan."),
    ("n_grad_steps", "Last N DDIM steps run with autograd; earlier steps detached."),
    ("act_horizon", "Steps between MPC re-plans."),
    ("window", "Total prediction window length passed to the agent."),
    ("hist", "Number of history frames in the prefix."),
    ("max_steps", "Episode horizon for the env loop."),
    ("n_steps", "DDIM denoising steps per agent inference."),
    ("save_intermediates", "If True, keep per-replan dream traces and candidate snapshots."),
]


# ────────────────────────────────────────────────────────────────────
# Intermediates I/O + analysis helpers
# ────────────────────────────────────────────────────────────────────

def save_mpc_intermediates(intermediates: list, path: str):
    """Save per-replan optimization intermediates to a single .npz file.

    ``intermediates`` is the list returned in ``run_mpc_rollout(...)['intermediates']``
    — one dict per replan with keys: ``replan_step``, ``baseline_chunk``,
    ``cand_xy_per_iter``, ``value_per_iter``, ``imagined_goal``,
    ``imagined_full``, ``best_idx``, ``act_lo``, ``act_hi``,
    ``real_state_at_replan``, and (if save_intermediates=True)
    ``gen_trace_per_iter``.

    Storage layout: each replan's arrays are flattened with a
    ``rep<i>__<key>`` prefix so a single np.savez compressed call
    captures everything. Load with::

        d = np.load(path); for k in d.files: print(k, d[k].shape)
    """
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    flat = {"n_replans": np.array(len(intermediates))}
    for i, rep in enumerate(intermediates):
        for k, v in rep.items():
            flat[f"rep{i:03d}__{k}"] = np.asarray(v)
    np.savez_compressed(path, **flat)
    return path


def plot_optimization_curves_with_envstep(
    intermediates: list, path: str, title: str = "",
):
    """Per-replan curve plot annotated with the env step.

    LEFT: best-of-K value vs inner SGD iter, color-coded by env step.
    RIGHT: env-step (x) vs final-best value (y). Shows whether the
    optimizer's residual error grows / shrinks across the rollout.

    Use alongside the simpler ``plot_optimization_curves`` for full
    diagnostic context.
    """
    import matplotlib.pyplot as plt

    if len(intermediates) == 0:
        raise ValueError("empty intermediates")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_l, ax_r = axes

    cmap = plt.get_cmap("viridis")
    n = len(intermediates)
    final_bests = []
    env_steps = []
    for i, rep in enumerate(intermediates):
        v = np.asarray(rep["value_per_iter"])  # (n_inner+1, K)
        best = v.min(axis=1)
        c = cmap(i / max(n - 1, 1))
        env_step = int(rep.get("replan_step", i * 10))
        env_steps.append(env_step)
        final_bests.append(float(best[-1]))
        label = f"step {env_step}" if (i % max(1, n // 6) == 0) else None
        ax_l.plot(np.arange(len(best)), best, color=c, alpha=0.8, label=label)

    ax_l.set_xlabel("inner SGD iter")
    ax_l.set_ylabel("best-of-K value (lower = closer to imagined goal)")
    ax_l.set_title("Optimization per replan (color = env step)")
    if any(l is not None for l in ax_l.get_legend_handles_labels()[1]):
        ax_l.legend(fontsize=8, loc="best")
    ax_l.grid(True, alpha=0.3)

    ax_r.plot(env_steps, final_bests, "o-", color="C1")
    ax_r.set_xlabel("env step at replan")
    ax_r.set_ylabel("final best-of-K value")
    ax_r.set_title("Residual error across the rollout")
    ax_r.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_per_replan_curves(intermediates: list, path: str, title: str = "",
                           ncols: int = 4):
    """One matplotlib figure, one subplot per replan, all K candidates.

    For each replan, plot the per-candidate value trajectory across
    inner SGD iters (K lines). The selected candidate (``best_idx``)
    is drawn with a thicker red line so it pops vs the rest.

    Layout: ``ncols`` columns × ``ceil(n / ncols)`` rows. Useful when
    you want to scan optimization behavior replan-by-replan instead
    of summary curves.
    """
    import math
    import matplotlib.pyplot as plt

    if len(intermediates) == 0:
        raise ValueError("empty intermediates")
    n = len(intermediates)
    nrows = max(1, math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.0 * ncols, 2.4 * nrows),
        squeeze=False, sharex=True,
    )
    cmap = plt.get_cmap("tab10")
    for i, rep in enumerate(intermediates):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        v = np.asarray(rep["value_per_iter"])  # (n_inner+1, K)
        K = v.shape[1]
        iters = np.arange(v.shape[0])
        best_idx = int(rep.get("best_idx", -1))
        for k in range(K):
            color = cmap(k % 10)
            if k == best_idx:
                ax.plot(iters, v[:, k], color="red", linewidth=2.0,
                        alpha=0.95, label=f"best (K={k})", zorder=5)
            else:
                ax.plot(iters, v[:, k], color=color, alpha=0.55,
                        linewidth=1.0)
        env_step = int(rep.get("replan_step", -1))
        ax.set_title(f"replan {i} — env step {env_step}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if r == nrows - 1:
            ax.set_xlabel("inner SGD iter", fontsize=8)
        if c == 0:
            ax.set_ylabel("value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="upper right")

    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def render_dream_traces_video(
    visualizer, intermediates: list, uid: list, pcl: torch.Tensor, path: str,
    H: int = 200, W: int = 200, fps: int = 4,
    iters_to_show: tuple = (0, -1),  # start + final by default; ('mid', also OK)
    max_K: int = 4,
):
    """Render per-replan dream traces — start vs final SGD iter, side-by-side.

    For each replan, takes the predicted ``gen_trace`` at iter 0 and the
    final iter (or any indices in ``iters_to_show``), renders the K
    candidates' end frames in a grid, and concatenates the panels into
    a single video. Mirrors the v1 MPC notebook's "dream traces"
    visualization.

    Requires ``intermediates[i]['gen_trace_per_iter']`` to be present
    (i.e. ``cfg.save_intermediates=True``).
    """
    import imageio

    if len(intermediates) == 0:
        raise ValueError("empty intermediates")
    has_dream = "gen_trace_per_iter" in intermediates[0]
    if not has_dream:
        raise ValueError(
            "intermediates lack 'gen_trace_per_iter'; "
            "set MPCConfig.save_intermediates=True before the rollout"
        )
    pcl_mask = torch.ones((pcl.shape[0],), dtype=torch.bool)
    viz_T_cam = visualizer.make_viz_T_cam(batch=None)

    out_frames = []
    for rep in intermediates:
        gen_per_iter = np.asarray(rep["gen_trace_per_iter"])  # (n_iter, K, T, N, 4, 4)
        n_iter, K = gen_per_iter.shape[:2]
        K_show = min(K, max_K)

        # Resolve iter indices ('mid' → middle).
        idxs = []
        for v in iters_to_show:
            if v == "mid":
                idxs.append(n_iter // 2)
            elif isinstance(v, int):
                idxs.append(v if v >= 0 else n_iter + v)
        idxs = [max(0, min(n_iter - 1, i)) for i in idxs]

        panels = []
        for it in idxs:
            row_imgs = []
            for k in range(K_show):
                # Render the END frame of the predicted trace for cand k.
                T_wl = torch.as_tensor(gen_per_iter[it, k, -1:], dtype=torch.float32)
                f = visualizer.render_viz_frames(
                    prompt_type=uid, T_wl=T_wl,
                    viz_T_cam=viz_T_cam, local_pcl=pcl,
                    local_pcl_mask=pcl_mask, H=H, W=W,
                )[0]
                row_imgs.append(np.asarray(f, dtype=np.uint8)[..., :3])
            panels.append(np.concatenate(row_imgs, axis=1))  # horizontal: K cands
        # Stack panels vertically (one row per iter).
        replan_img = np.concatenate(panels, axis=0)
        out_frames.append(replan_img)

    os.makedirs(osp.dirname(path) or ".", exist_ok=True)
    try:
        imageio.mimsave(path, out_frames, fps=fps, codec="libx264")  # type: ignore[arg-type]
    except (ValueError, RuntimeError):
        gif_path = osp.splitext(path)[0] + ".gif"
        imageio.mimsave(gif_path, out_frames, fps=fps)  # type: ignore[arg-type]
        return gif_path
    return path
