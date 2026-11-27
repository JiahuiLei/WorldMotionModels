"""Build the noisy test inputs of the paper (noise sigma 0.1) from your clean test clips and the published noise.

    python apps/retargeting/prepare_data/add_test_noise.py \
        --clean release_data/retargeting/L2_eval/amass_human_test \
        --noise release_data/retargeting/L2_eval/amass_human_test_q0.1_t0.1_noise \
        --out   release_data/retargeting/L2_eval/amass_human_test_q0.1_t0.1

The paper's noisy clips are the clean clips plus one draw of Gaussian noise (std 0.1 rad on the 165 SMPL-X
axis-angle values, 0.1 m on the root position). The noise is published as `<clip>.npz` holding
`pose_noise` (1, T, 165) and `global_t_noise` (1, T, 3), float64. For each clip this script writes
`pose = float64(clean pose) + pose_noise`, `global_t = float64(clean global_t) + global_t_noise`, and copies
`betas` and `fps`; with the paper's clean clips this reproduces the paper's noisy clips bit for bit.
"""

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--clean', required=True, type=Path)
    parser.add_argument('--noise', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in args.noise.glob('*.npz'))
    if not names:
        raise SystemExit(f'no noise files in {args.noise}')
    for name in names:
        clean = np.load(args.clean / name)
        noise = np.load(args.noise / name)
        np.savez(args.out / name,
                 pose=clean['pose'].astype(np.float64) + noise['pose_noise'],
                 global_t=clean['global_t'].astype(np.float64) + noise['global_t_noise'],
                 betas=clean['betas'], fps=clean['fps'])
    print(f'wrote {len(names)} clips to {args.out}')


if __name__ == '__main__':
    main()
