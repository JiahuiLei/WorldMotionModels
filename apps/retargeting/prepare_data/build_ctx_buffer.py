"""Fill the context buffer of a WMM retargeting checkpoint from your own training data.

    python apps/retargeting/prepare_data/build_ctx_buffer.py <weights.ckpt> <out.ckpt> \
        [--data_root release_data/retargeting/L3_train] [--seed 42] [--workers 12]

At inference WMM conditions on context frames of paired human/robot poses. During training the model
records the context frames of every training batch into `ctx_record_buffer`: one first-in-first-out queue
of 16,384 frames per set of body parts (human joints, G1 links, H1-2 links), each frame of shape
(1, P, 4, 4) float32. The published checkpoint does not contain these frames because they are AMASS- and
Embody3D-derived motion. This script draws training batches with the training data loader (training config
`w1.0.1`, fixed seed), records their context frames with the model's own recording function
(`AgentV11.record_ctx`), stops once every queue holds 16,384 frames and no new body-part set has appeared for
`--settle` batches, and writes the buffer into `model.ctx_record_buffer` and `ema_model.ctx_record_buffer`.
Weights are not changed.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

APP = Path(__file__).resolve().parent.parent
CODE = APP / 'code' / 'train'
CONFIG = CODE / 'configs' / 'wmm' / 'hh' / 'w1.0.1.yaml'
BUFFER_SIZE = 16384


def code_paths(data_root: Path):
    paths = [CODE, CODE / 'src']
    for lib in sorted((CODE / 'ext').iterdir()) + sorted((data_root / 'ext').iterdir()):
        paths.append(lib / 'src' if (lib / 'src').is_dir() else lib)
    return [str(p) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ckpt_in')
    parser.add_argument('ckpt_out')
    parser.add_argument('--data_root', default=str(APP.parent.parent / 'release_data' / 'retargeting' / 'L3_train'),
                        help='folder holding data/, data_splits/ and ext/ (the training config paths are relative to it)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--settle', type=int, default=2000,
                        help='batches without a new body-part set required before stopping')
    parser.add_argument('--max_batches', type=int, default=100000)
    args = parser.parse_args()

    ckpt_in, ckpt_out = Path(args.ckpt_in).resolve(), Path(args.ckpt_out).resolve()
    data_root = Path(args.data_root).resolve()
    sys.path[:0] = code_paths(data_root)
    os.chdir(data_root)
    os.environ.setdefault('SAPIEN_RENDER_DEVICE', 'cpu')
    os.environ.setdefault('SAPIEN_DISABLE_RAY_TRACING', '1')

    import pytorch_lightning as pl
    import torch
    from umm.datasets import get_data_module
    from umm.models.wmm.agent import AgentV11
    from umm.training_utils import get_config

    pl.seed_everything(args.seed, workers=True)
    cfg = get_config(['--config', str(CONFIG), '--workers', str(args.workers)])
    dm = get_data_module(cfg)
    dm.setup('fit')

    recorder = SimpleNamespace(ctx_record_flag=True, ctx_record_buffer={}, ctx_record_buffer_size=BUFFER_SIZE)
    buf = recorder.ctx_record_buffer
    t0, last_new, n = time.time(), 0, 0
    for n, batch in enumerate(dm.train_dataloader(), start=1):
        before = len(buf)
        AgentV11.record_ctx(recorder, batch['context'], batch['context_mask'], batch['trace_uid'])
        if len(buf) > before:
            last_new = n
        full = all(len(q) == BUFFER_SIZE for q in buf.values())
        if n % 500 == 0:
            print(f'batch {n}: ' + ', '.join(f'{len(k)} parts: {len(q)}' for k, q in buf.items())
                  + f' ({time.time() - t0:.0f} s)', flush=True)
        if full and n - last_new >= args.settle:
            break
        if n >= args.max_batches:
            raise SystemExit(f'buffer not full after {n} batches: {[len(q) for q in buf.values()]}')

    ckpt = torch.load(ckpt_in, map_location='cpu', weights_only=False)
    for prefix in ('model.', 'ema_model.'):
        if prefix + 'ctx_record_buffer' in ckpt['state_dict']:
            raise SystemExit(f'{ckpt_in} already has {prefix}ctx_record_buffer')
        ckpt['state_dict'][prefix + 'ctx_record_buffer'] = buf
    torch.save(ckpt, ckpt_out)
    print(f'wrote {ckpt_out}: {len(buf)} body-part sets x {BUFFER_SIZE} frames from {n} batches (seed {args.seed})')


if __name__ == '__main__':
    main()
