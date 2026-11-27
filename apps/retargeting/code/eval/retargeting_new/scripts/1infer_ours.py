"""Stage 1: WMM retargeting inference on the test clips (one output .npz per clip).

The three settings of the paper table are produced by varying `--cond_step` and `--skip`:
  30FPS clean: --cond_step 0 --skip 1 --human_data_root <clean>
  3FPS Filling: --cond_step 0 --skip 10 --human_data_root <clean>
  q0.1_t0.1 CT=10: --cond_step 10 --skip 1 --human_data_root <noised>

CT (`condition_timestep`) is the diffusion timestep at which the observed human pose is fed in as
conditioning (the config has `train_diff_timesteps=100`); CT=10 is the noised-condition robustness mode used
for the noisy (q0.1_t0.1) setting.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch
from tqdm import tqdm


PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Training config YAML")
    p.add_argument("--ckpt", required=True, help="Checkpoint .ckpt")
    p.add_argument("--save_dir", required=True,
                   help="Output dir under logs/eval/humanoid/<NAME>/ — REQUIRED")
    p.add_argument("--human_data_root", default="./logs/eval/humanoid/amass_human_test/",
                   help="Frozen GT human npz dir (clean OR _q0.1_t0.1)")
    p.add_argument("--test_split_fn", default="./data_splits/amass_humanoid_v2/full_test.json")
    p.add_argument("--ckpt_prefix", default="ema_model.")
    # input_flag: default to None so we can distinguish "user didn't set"
    # from "user set False". Auto-defaults to (cfg.model.name != "gemm_v1")
    # below — gemm_v1 ckpts have no input_layer state, all others do.
    # Mismatch silently randomizes the InputLayer embeddings → catastrophic eval.
    p.add_argument("--input_flag", action=argparse.BooleanOptionalAction, default=None,
                   help="Load input_layer from ckpt. Auto-defaults from cfg.model.name "
                        "(gemm_v1→False, wmm/etc→True). Override at your own risk.")
    p.add_argument("--cond_step", type=int, default=0,
                   help="condition_timestep — 0=clean, 10=noisy (q0.1_t0.1) setting")
    p.add_argument("--skip", type=int, default=1,
                   help="Subsample stride — 1=30FPS, 10=3FPS Filling")
    p.add_argument("--ctx_len", type=int, default=9)
    p.add_argument("--window_len", type=int, default=30)
    p.add_argument("--window_overlap", type=int, default=5)
    p.add_argument("--gen_steps", type=int, default=10)
    p.add_argument("--n_first", type=int, default=None,
                   help="Process only first N split entries (None=all 2014)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", action="store_true",
                   help="Run a warmup pass (notebook does this; first call carries compile cost)")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, PROJECT_ROOT)

    from umm.training_utils import get_config, init_setup, get_logger
    from lib_twist.g1_interface import G1Interface
    from lib_human.human_model import HumanModel

    config_fn = args.config if osp.isabs(args.config) else osp.join(PROJECT_ROOT, args.config)

    # ──────────────────────────────────────────────────────────────────
    # Cross-arch eval — auto-default input_flag based on cfg.model.name.
    # gemm_v1 has no input_layer in state_dict → False.
    # wmm / future archs use InputLayerV11Super → True.
    # Mismatch silently randomizes the input embeddings → CATASTROPHIC
    # (random outputs). The loud log line below MUST be sanity-checked
    # on every eval run.
    # ──────────────────────────────────────────────────────────────────
    import yaml as _yaml
    with open(config_fn) as _fh:
        _cfg_peek = _yaml.safe_load(_fh)
    _arch = (_cfg_peek.get("model", {}) or {}).get("name", None)
    if _arch is None:
        raise ValueError(
            f"cfg.model.name missing in {config_fn} — required for "
            f"input_flag auto-default. Set `model.name` in the YAML "
            f"(e.g. 'gemm_v1' or 'wmm') so the cross-arch eval guard "
            f"can pick the right input_flag."
        )
    if args.input_flag is None:
        args.input_flag = (_arch != "gemm_v1")
    print("=" * 78)
    print(
        f"[infer_ours] arch={_arch}  ckpt_prefix={args.ckpt_prefix}  "
        f"input_flag={args.input_flag}"
    )
    print(
        "[infer_ours] >>> SANITY CHECK: gemm_v1 → input_flag=False; "
        "wmm/other → True. Mismatch = random embeddings = bogus eval."
    )
    print("=" * 78)
    ckpt_path = args.ckpt if osp.isabs(args.ckpt) else osp.join(PROJECT_ROOT, args.ckpt)
    save_dir = args.save_dir if osp.isabs(args.save_dir) else osp.join(PROJECT_ROOT, args.save_dir)
    human_data_root = args.human_data_root if osp.isabs(args.human_data_root) else osp.join(PROJECT_ROOT, args.human_data_root)
    test_split_fn = args.test_split_fn if osp.isabs(args.test_split_fn) else osp.join(PROJECT_ROOT, args.test_split_fn)

    print(f"[infer_ours] config       = {config_fn}")
    print(f"[infer_ours] ckpt         = {ckpt_path}")
    print(f"[infer_ours] save_dir     = {save_dir}")
    print(f"[infer_ours] human_root   = {human_data_root}")
    print(f"[infer_ours] split        = {test_split_fn}")
    print(f"[infer_ours] cond_step    = {args.cond_step}  (CT)")
    print(f"[infer_ours] skip         = {args.skip}")
    print(f"[infer_ours] ctx/W/ovlp/gs= {args.ctx_len}/{args.window_len}/{args.window_overlap}/{args.gen_steps}")

    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not osp.isdir(human_data_root):
        raise FileNotFoundError(f"human_data_root not found: {human_data_root}")
    os.makedirs(save_dir, exist_ok=True)

    cfg = get_config(f"--config {config_fn} --expname retarget --use_wandb=False")
    cfg.log_dir = osp.join(PROJECT_ROOT, cfg.log_dir)
    cfg.model.human.model_path = osp.join(PROJECT_ROOT, cfg.model.human.model_path)
    for k in cfg.model.humanoid.mesh_fn:
        cfg.model.humanoid.mesh_fn[k] = osp.join(PROJECT_ROOT, cfg.model.humanoid.mesh_fn[k])
    get_logger(cfg)
    init_setup(cfg)

    # Dispatch PLModel via cfg.model.name (e.g. "gemm_v1" for historical
    # g7.17.2 ckpt). Uses get_model registry so any future arch with a
    # PLModel.get_agent classmethod works without script changes.
    import importlib
    plmod = importlib.import_module(f"umm.models.{cfg.model.name}.model")
    PLModel = getattr(plmod, "PLModel")
    agent = PLModel.get_agent(cfg, init_fn=ckpt_path, ckpt_prefix=args.ckpt_prefix, input_flag=args.input_flag)
    if hasattr(agent, "summarize_ctx_record"):
        agent.summarize_ctx_record()
    agent = agent.eval().to(args.device)
    # gemm_v1 historical inference uses hh_boost cache pre-computation
    if hasattr(agent, "hh_boost"):
        agent.hh_boost(C=args.ctx_len, T=args.window_len)

    urdf_file = osp.join(PROJECT_ROOT, "ext/lib_twist/src/lib_twist/assets/g1/g1_custom_collision_with_fixed_hand.urdf")
    G1I = G1Interface(urdf_file, device=args.device)  # noqa: F841 — kept for legacy compat (FK side-effects in some agents)

    HM = HumanModel(
        model_path=osp.join(PROJECT_ROOT, "data/smpls"),
        model_type="smplx",
        gender="neutral",
        device="cpu",
        num_betas=300,
    )

    with open(test_split_fn, "r") as f:
        test_split = json.load(f)
    if args.n_first is not None:
        test_split = test_split[: args.n_first]

    def load_human_data(human_fn):
        data = np.load(human_fn)
        beta = data["betas"][0][0].astype(np.float32)
        global_t = data["global_t"][0].astype(np.float32)
        pose = data["pose"][0].astype(np.float32)
        return pose, global_t, beta

    def build_mask(T_len: int):
        if args.skip <= 1:
            return None
        index = np.arange(0, T_len, args.skip).tolist()
        if T_len - 1 not in index:
            index.append(T_len - 1)
        m = np.zeros(T_len, dtype=bool)
        m[np.array(index)] = True
        return torch.as_tensor(m).bool()

    def retarget_one(human_Q: torch.Tensor, save_npz_fn: str | None):
        # NOTE: NOT @torch.inference_mode() — G1I.adjust does IK refinement
        # via chain.batch_finetune which calls loss.backward(). Use no_grad
        # only around the agent forward.
        human_Q = torch.as_tensor(human_Q).to(args.device)
        T_len = human_Q.shape[0]
        timestep = (torch.arange(T_len).float()[None] / 30.0).to(args.device)

        mask = build_mask(T_len)
        if mask is not None:
            mask_full = mask[:, None].expand(-1, human_Q.shape[1])
            human_Q[~mask_full] = 0.0
            human_visible = mask_full.bool().to(args.device)
        else:
            human_visible = torch.ones_like(human_Q[:, :, 0, 0]).bool()

        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            gen_human_Q, gen_robot_Q, gen_time = agent.hh_joint_infer(
                ctx_len=args.ctx_len,
                windows_size=args.window_len,
                window_overlap=args.window_overlap,
                human_Q=human_Q,
                human_time=timestep,
                human_visible=human_visible,
                gen_steps=args.gen_steps,
                condition_timestep=args.cond_step,
                verbose=False,
            )
        gen_robot_Q = gen_robot_Q.float()
        gen_human_Q = gen_human_Q.float()

        from umm.datasets.umm_v1.dataset_config import G1_LINK_NAMES

        root_pos, root_quat_xyzw, dof_pos = G1I.RK(
            torch.cat([gen_robot_Q[:, :1], gen_robot_Q], 1),
            names=G1_LINK_NAMES,
        )
        dof_pos, _ = G1I.adjust(
            torch.cat([gen_robot_Q[:, :1], gen_robot_Q], 1),
            names=G1_LINK_NAMES,
            joint_angles=dof_pos.cuda(),
            lrs=[1.0, 0.1, 0.01],
            profile_flag=False,
        )
        run_time = time.time() - t0
        fps = T_len / run_time

        if save_npz_fn is not None:
            np.savez(
                save_npz_fn,
                root_pos=root_pos.cpu().numpy().astype(np.float32),
                root_quat_xyzw=root_quat_xyzw.cpu().numpy().astype(np.float32),
                dof_pos=dof_pos.cpu().numpy().astype(np.float32),
                run_time=np.float64(run_time),
                fps=np.float64(fps),
                ctx_len=np.int64(args.ctx_len),
                window_size=np.int64(args.window_len),
                window_overlap=np.int64(args.window_overlap),
                gen_steps=np.int64(args.gen_steps),
                input_human_Q=human_Q.cpu().numpy().astype(np.float16),
                output_robot_Q=gen_robot_Q.cpu().numpy().astype(np.float16),
                output_human_Q=gen_human_Q.cpu().numpy().astype(np.float16),
            )
        return fps

    if args.warmup and len(test_split) > 11:
        sid_w = test_split[11][0]
        pose, transl, beta = load_human_data(osp.join(human_data_root, sid_w + ".npz"))
        Qw, _, _, _ = HM.forward(shape_params=beta, pose_params=pose, transl=transl)
        retarget_one(Qw, None)

    pbar = tqdm(test_split)
    for sid, s_len in pbar:
        out_fn = osp.join(save_dir, sid + ".npz")
        if osp.exists(out_fn):
            continue
        pose, transl, beta = load_human_data(osp.join(human_data_root, sid + ".npz"))
        Q, _, _, _ = HM.forward(shape_params=beta, pose_params=pose, transl=transl)
        assert len(Q) == int(s_len), f"len mismatch: {len(Q)} vs {int(s_len)} for {sid}"
        fps = retarget_one(Q, out_fn)
        pbar.set_postfix(fps=f"{fps:.1f}")


if __name__ == "__main__":
    main()
