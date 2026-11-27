"""Export a weights-only WMM checkpoint from a full PyTorch Lightning training checkpoint.

    python apps/retargeting/export_weights.py <full.ckpt> <weights.ckpt> [--keep_ctx_buffer]

Keeps `epoch`, `global_step`, `pytorch-lightning_version` and every `model.*` and `ema_model.*` tensor of
`state_dict`. Drops optimizer, scheduler, loop, callback, data-module and EMA-helper state, and the context
buffer recorded from training data (`*.ctx_record_buffer`) unless `--keep_ctx_buffer` is given; rebuild it
with `prepare_data/build_ctx_buffer.py`. Then checks that every kept tensor equals the original.
"""

import argparse

import torch

KEEP = ('model.', 'ema_model.')
BUFFER = 'ctx_record_buffer'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('src')
    parser.add_argument('dst')
    parser.add_argument('--keep_ctx_buffer', action='store_true')
    args = parser.parse_args()

    full = torch.load(args.src, map_location='cpu', mmap=True, weights_only=False)
    state = {k: v for k, v in full['state_dict'].items()
             if k.startswith(KEEP) and (args.keep_ctx_buffer or not k.endswith(BUFFER))}
    if not any(k.startswith('ema_model.') for k in state):
        raise ValueError(f'{args.src} has no ema_model.* weights')
    torch.save({
        'epoch': full['epoch'],
        'global_step': full['global_step'],
        'pytorch-lightning_version': full['pytorch-lightning_version'],
        'state_dict': state,
    }, args.dst)

    out = torch.load(args.dst, map_location='cpu', mmap=True, weights_only=False)['state_dict']
    if out.keys() != state.keys():
        raise RuntimeError('key set changed while writing')
    n = 0
    for k, v in state.items():
        if torch.is_tensor(v):
            if not torch.equal(out[k], v):
                raise RuntimeError(f'{k} differs')
            n += 1
        else:
            for bucket, frames in v.items():
                if len(out[k][bucket]) != len(frames) or not all(torch.equal(a, b) for a, b in zip(out[k][bucket], frames)):
                    raise RuntimeError(f'{k}[{bucket}] differs')
                n += len(frames)
    print(f'wrote {args.dst}: {len(state)} entries, {n} tensors equal to {args.src}, global_step {full["global_step"]}')


if __name__ == '__main__':
    main()
