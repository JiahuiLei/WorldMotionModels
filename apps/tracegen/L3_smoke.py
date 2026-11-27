"""Layer 3 smoke test: load the released TraceGen training code, config and data, draw a few training
batches and build the model (on the CPU, without training).

    python apps/tracegen/L3_smoke.py [--run epic_k128] [--batches 3]

Needs no GPU. It prints the shape of each tensor in the first batches and the model size, and checks
that every imported module comes from apps/tracegen/code_train/ or the Python environment itself.
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
CODE = APP / "code_train"
RUNS = ("droid_k128", "droid_k400", "epic_k128", "epic_k400", "droidepic_k128", "droidepic_k400")


def describe(x):
    if hasattr(x, "shape"):
        return f"{tuple(x.shape)} {x.dtype}"
    if isinstance(x, dict):
        return {k: describe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    return type(x).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="epic_k128", choices=RUNS)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--data", default=os.environ.get("WMM_DATA", str(APP.parent.parent / "release_data")),
                        help="release_data root (holds tracegen/L3_train and tracegen/checkpoints)")
    args = parser.parse_args()

    data = Path(args.data).resolve() / "tracegen"
    sys.path[:0] = [str(CODE), str(CODE / "src"), str(CODE / "ext" / "lib_io"), str(CODE / "ext" / "lib_mani"),
                    str(CODE / "ext" / "pytorch_kinematics" / "src")]
    os.environ.setdefault("HF_HOME", str(data / "checkpoints" / "hf_home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("WMM_DINOV3_REPO", str(CODE / "ext" / "dinov3"))
    os.environ.setdefault("WMM_DINOV3_CKPT_DIR", str(data / "checkpoints" / "dinov3"))
    # A shared dataset module imports a ManiSkill helper that creates a SAPIEN renderer at import time;
    # without a GPU that crashes unless SAPIEN renders on the CPU. The smoke test renders nothing.
    os.environ.setdefault("SAPIEN_RENDER_DEVICE", "cpu")
    os.environ.setdefault("SAPIEN_DISABLE_RAY_TRACING", "1")
    # The config names ./data_wmm/... relative to the working directory.
    os.chdir(data / "L3_train")
    config = CODE / "configs" / "wmm" / "tg" / f"w1.6.0_{args.run}_tpsqspt.yaml"

    t0 = time.time()
    from umm.training_utils import get_config, get_logger
    from umm.datasets import get_data_module
    from umm.models import get_model

    # Logs of this smoke test go to a temporary folder; nothing is written next to the data.
    os.environ["WANDB_MODE"] = "disabled"
    log_dir = tempfile.mkdtemp(prefix="tg_l3_smoke_")
    sys.argv = ["train.py", "--config", str(config), "--workers", "0", "--log_dir", log_dir]
    cfg = get_config()
    dm = get_data_module(cfg)
    dm.setup("fit")
    loader = dm.train_dataloader()
    print(f"train batch size {cfg.train_batchsize}; setup took {time.time() - t0:.0f} s")
    for i, batch in enumerate(loader):
        items = batch.items() if isinstance(batch, dict) else enumerate(batch)
        print(f"batch {i}:", {k: describe(v) for k, v in items})
        if i + 1 >= args.batches:
            break

    model = get_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"model {type(model).__module__}.{type(model).__name__}: {n / 1e6:.1f} M parameters "
          "(training model + EMA copy + frozen T5 and DINOv3)")

    # The Trainer itself needs a GPU; build the callbacks it would get (printing, visualization, the
    # in-training TraceGen evaluation) so their code is loaded and checked as well.
    from umm.sims.robot_sim_callbacks import get_eval_callback
    from umm.training_utils.callbacks import PrintingCallback, VizCallback
    get_logger(cfg)
    callbacks = [PrintingCallback(cfg), VizCallback(cfg), *get_eval_callback(cfg, os.path.join(log_dir, "rollout_videos"))]
    print("training callbacks:", [type(c).__name__ for c in callbacks])

    allowed = [CODE.resolve()] + [Path(p).resolve() for p in (sys.prefix, sys.base_prefix)]
    outside = sorted({str(f) for f in (Path(m.__file__).resolve() for m in list(sys.modules.values())
                                       if getattr(m, "__file__", None))
                      if f != Path(__file__).resolve() and not any(f.is_relative_to(a) for a in allowed)})
    if outside:
        raise SystemExit(f"modules loaded from outside {CODE} and the Python environment: {outside}")
    print(f"OK: {args.batches} batches in {time.time() - t0:.0f} s; every module comes from {CODE} or {sys.prefix}")


if __name__ == "__main__":
    main()
