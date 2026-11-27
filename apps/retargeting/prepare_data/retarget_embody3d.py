"""Embody 3D: curate the SMPL-X fits of each person and retarget them to a humanoid robot with PHUMA.

    python apps/retargeting/prepare_data/retarget_embody3d.py --embody3d_root <embody3d> \
        --save_root release_data/retargeting/L3_train/data/embody3d_phuma_processed/g1 --robot_name g1
    python apps/retargeting/prepare_data/retarget_embody3d.py --embody3d_root <embody3d> \
        --indices prepare_data/embody3d_indices_hand_duo.json \
        --save_root release_data/retargeting/L3_train/data/embody3d_phuma_processed/hand_duo_subset

`<embody3d>` is the folder into which the `embody_3d__20251016__*_smplx.zip` archives were extracted
(`<category>/<capture>/<person>/smplx_mesh_*/<capture>.npy`). The index files map `<category>/<capture>` to the
persons used for training: `embody3d_indices.json` (26,258 persons in 9,005 captures, robot data for the G1
folder) and `embody3d_indices_hand_duo.json` (299 Hand_Duo persons; training reads only their `human_states`).
Output file names end in `__________<person>`; see phuma_datagen.py for the content. 1001 optimisation steps
per chunk.
"""

import argparse
import functools
import json
from pathlib import Path

import phuma_datagen

HERE = Path(__file__).resolve().parent


def process(wrapper, job, num_iter: int) -> None:
    capture, person, path = job
    wrapper.process_seq(capture, *wrapper.load_embody3d_dir(path), num_iter=num_iter, print_every=1000,
                        seq_id_postfix=f'__________{person}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--embody3d_root', required=True, type=Path)
    parser.add_argument('--indices', type=Path, default=HERE / 'embody3d_indices.json')
    parser.add_argument('--num_iter', type=int, default=1001)
    parser.add_argument('--ids', nargs='*', help='process only these "<capture>|<person>" pairs')
    phuma_datagen.add_common_args(parser)
    args = parser.parse_args()
    index = json.loads(args.indices.read_text())
    jobs = [(key.split('/')[1], person, str(args.embody3d_root / key / person))
            for key, persons in index.items() for person in persons]
    if args.ids:
        jobs = [j for j in jobs if f'{j[0]}|{j[1]}' in set(args.ids)]
    missing = [j for j in jobs if not Path(j[2]).is_dir()]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(jobs)} person folders missing under {args.embody3d_root}, '
                         f'e.g. {missing[0][2]}')
    phuma_datagen.run_jobs(args, jobs, functools.partial(process, num_iter=args.num_iter))


if __name__ == '__main__':
    main()
