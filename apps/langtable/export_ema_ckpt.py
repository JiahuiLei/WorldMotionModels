"""Export the EMA-only WMM checkpoint used by Layer 2 from a full PyTorch Lightning training checkpoint.

    python apps/langtable/export_ema_ckpt.py <full.ckpt> <ema.ckpt>

Keeps epoch, global_step, pytorch-lightning_version and the `ema_model.*` entries of `state_dict`
(the weights the evaluation runs); drops the training model, optimizer, scheduler and EMA shadow state.
"""

import sys

import torch


def main(src: str, dst: str) -> None:
    ckpt = torch.load(src, map_location='cpu', mmap=True, weights_only=False)
    slim = {
        'epoch': ckpt['epoch'],
        'global_step': ckpt['global_step'],
        'pytorch-lightning_version': ckpt['pytorch-lightning_version'],
        'state_dict': {k: v for k, v in ckpt['state_dict'].items() if k.startswith('ema_model.')},
    }
    if not slim['state_dict']:
        raise ValueError(f'{src} has no ema_model.* weights')
    torch.save(slim, dst)
    print(f'wrote {dst}: {len(slim["state_dict"])} tensors, global_step {slim["global_step"]}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
