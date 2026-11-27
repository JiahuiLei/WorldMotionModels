"""Give the EMA model of a trained checkpoint the context buffer recorded during training.

    python apps/retargeting/L3_share_ctx_buffer.py <trained.ckpt> <out.ckpt>

During training the model records context frames into `model.ctx_record_buffer` (4 buckets x 16,384 frames of
shape (1, 83, 4, 4), float32). Inference loads the EMA weights (`ema_model.*`) and needs that buffer, but only the
in-training check (`callbacks.hh_eval`) hands it to the EMA model, by sharing the same object. A run trained with
`--callbacks.hh_eval.enabled=False` therefore saves an empty `ema_model.ctx_record_buffer`. This script sets it to
the recorded buffer, which is what the sharing gives: in the paper checkpoint the two buffers are equal. Weights
are not touched; the input file is not modified.
"""

import shutil
import sys

import torch


def buffer_size(buf: dict) -> tuple:
    return len(buf), sum(len(frames) for frames in buf.values())


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    recorded = sd["model.ctx_record_buffer"]
    if not recorded:
        raise SystemExit("model.ctx_record_buffer is empty: nothing was recorded during training")
    ema = sd["ema_model.ctx_record_buffer"]
    if ema:
        shutil.copyfile(src, dst)
        print(f"ema_model.ctx_record_buffer already holds {buffer_size(ema)} (buckets, frames); copied {src} unchanged")
        return
    sd["ema_model.ctx_record_buffer"] = recorded
    torch.save(ckpt, dst)
    print(f"step {ckpt['global_step']}: ema_model.ctx_record_buffer <- model.ctx_record_buffer "
          f"({buffer_size(recorded)[0]} buckets, {buffer_size(recorded)[1]} frames); wrote {dst}")


if __name__ == "__main__":
    main()
