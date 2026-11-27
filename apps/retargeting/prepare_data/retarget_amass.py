"""Step 2 for AMASS: curate the human motions and retarget them to a humanoid robot with PHUMA.

    python apps/retargeting/prepare_data/retarget_amass.py --src_root <amass_twist>/human_states \
        --save_root release_data/retargeting/L3_train/data/amass_phuma_v2/g1 --robot_name g1
    python apps/retargeting/prepare_data/retarget_amass.py --src_root <amass_twist>/human_states \
        --save_root release_data/retargeting/L3_train/data/amass_phuma_v2/h12 --robot_name h1_2

Only the human chunks of the test motions, for the Layer-2 test clips (a few CPU-core hours):
    python apps/retargeting/prepare_data/retarget_amass.py --src_root <amass_twist>/human_states --human_only \
        --split release_data/retargeting/L1_results/splits/amass_humanoid_v2_full_test.json \
        --save_root release_data/retargeting/L3_train/data/amass_phuma_v2/g1

Input: the per-motion files written by amass_to_twist_human.py. Output: see phuma_datagen.py.
`g1/human_states` and `h12/human_states` hold the same human chunks. 3001 optimisation steps per chunk.
"""

import argparse
import functools
import json
from pathlib import Path

import phuma_datagen


def process(wrapper, path: str, num_iter: int) -> None:
    seq_id = Path(path).stem
    wrapper.process_seq(seq_id, *wrapper.load_amass_saved_human_npz(path), num_iter=num_iter, print_every=1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src_root', required=True, type=Path)
    parser.add_argument('--num_iter', type=int, default=3001)
    parser.add_argument('--ids', nargs='*', help='process only these motion ids (file stems)')
    parser.add_argument('--split', type=Path, help='process only the motions of the clips in this split file')
    phuma_datagen.add_common_args(parser)
    args = parser.parse_args()
    files = sorted(args.src_root.glob('*.npz'))
    if args.ids:
        files = [f for f in files if f.stem in set(args.ids)]
    if args.split:
        motions = {entry[0].rsplit('_', 2)[0] for entry in json.loads(args.split.read_text())}
        files = [f for f in files if f.stem in motions]
        if len(files) != len(motions):
            raise SystemExit(f'{len(motions) - len(files)} of {len(motions)} split motions missing in {args.src_root}')
    if not files:
        raise SystemExit(f'no input motions in {args.src_root}')
    phuma_datagen.run_jobs(args, [str(f) for f in files], functools.partial(process, num_iter=args.num_iter))


if __name__ == '__main__':
    main()
