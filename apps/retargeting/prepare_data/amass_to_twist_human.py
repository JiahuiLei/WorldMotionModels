"""Step 1 for AMASS: select the AMASS motions of the TWIST motion dataset and resample them to 30 FPS.

    python apps/retargeting/prepare_data/amass_to_twist_human.py --amass_root <amass> --out <amass_twist>

`<amass>` holds the AMASS "SMPL-X G" archives of the 20 subsets below, each extracted into a folder named as in
`SUBSETS` (e.g. `<amass>/ACCAD/...`, `<amass>/MPI_MoSh/...`); give `--subset_dir NAME=PATH` for a subset that
lives elsewhere or under another name. Every AMASS file whose name matches a motion of the TWIST motion
dataset (YanjieZe/TWIST; names and frame counts in twist_motion_index.json) is kept. Per subset, the frame
step is the rounded mean ratio between AMASS and TWIST frame counts; a motion whose resampled length differs
from TWIST's is skipped.

Output `<out>/human_states/<subset>__<twist name>.npz` (input of retarget_amass.py), float16:
`pose` (1, N, 165) SMPL-X axis-angle at the AMASS frame rate, `global_t` (1, N/step, 3) root translation
resampled to 30 FPS, `betas` (1, 16), `valid_mask` (1, N/step) bool, `fps` 30.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _clean(name: str) -> str:
    return name.split('.npz')[0].replace('-', '_').replace('(', '_').replace(')', '_')


def _strip_stage(name: str) -> str:
    return name.split('_stageiii')[0].split('_stageii')[0].split('_stagei')[0]


def key_by_file(path: Path) -> str:
    return _strip_stage(_clean(path.name))


def key_by_folder_and_file(path: Path) -> str:
    return path.parent.name + '_' + _strip_stage(_clean(path.name)).split('_poses')[0]


def key_by_folder_and_file_keep_stage(path: Path) -> str:
    return path.parent.name + '_' + _clean(path.name).split('_poses')[0]


SUBSETS = [
    ('ACCAD', key_by_file), ('BMLmovi', key_by_file), ('BMLhandball', key_by_folder_and_file),
    ('CNRS', key_by_folder_and_file_keep_stage), ('DanceDB', key_by_file), ('Eyes_Japan', key_by_file),
    ('GRAB', key_by_folder_and_file_keep_stage), ('HDM05', key_by_file), ('HUMAN4D', key_by_folder_and_file),
    ('HumanEva', key_by_folder_and_file), ('MPI_MoSh', key_by_folder_and_file),
    ('MPI_Limits', key_by_folder_and_file), ('SFU', key_by_file), ('SSM_synced', key_by_file),
    ('TCD_HandMoCap', key_by_file), ('TotalCapture', key_by_folder_and_file), ('Transitions', key_by_file),
    ('DFaust', key_by_file), ('EKUT', key_by_folder_and_file), ('KIT', key_by_file),
]


def match_subset(subset_dir: Path, key_fn, twist: dict):
    """Return ([(seq_key, amass_file, twist_len)], frame step) for one subset."""
    matches, ratios = [], []
    for path in sorted(p.resolve() for p in subset_dir.glob('**/*.npz')):
        key = key_fn(path)
        if key not in twist:
            continue
        try:
            n = np.load(path)['poses'].shape[0]
        except Exception as e:  # noqa: BLE001 - unreadable or pose-less AMASS files are skipped, as listed below
            print(f'skip {path}: {e!r}')
            continue
        ratios.append(n / twist[key])
        matches.append((key, path, twist[key]))
    step = int(np.round(np.mean(ratios))) if ratios else 0
    return matches, step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--amass_root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--twist_index', type=Path, default=HERE / 'twist_motion_index.json')
    parser.add_argument('--subset_dir', action='append', default=[], metavar='NAME=PATH')
    args = parser.parse_args()

    twist_index = json.loads(args.twist_index.read_text())
    subset_dirs = {name: args.amass_root / name for name, _ in SUBSETS}
    subset_dirs.update(dict(item.split('=', 1) for item in args.subset_dir))
    out = args.out / 'human_states'
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for name, key_fn in SUBSETS:
        subset_dir = Path(subset_dirs[name])
        if not subset_dir.is_dir():
            raise SystemExit(f'AMASS subset {name} not found at {subset_dir}')
        matches, step = match_subset(subset_dir, key_fn, twist_index[name.lower()])
        written = 0
        for key, path, twist_len in matches:
            data = np.load(path)
            if not {'poses', 'trans', 'betas'} <= set(data.files):
                print(f'skip {path}: no poses/trans/betas')
                continue
            trans = data['trans'][::step]
            if len(data['poses'][::step]) != twist_len:
                print(f'skip {path}: {len(trans)} frames at step {step}, TWIST has {twist_len}')
                continue
            np.savez(out / f'{name.lower()}__{key}.npz',
                     valid_mask=np.ones((1, twist_len), dtype=bool),
                     global_t=trans.astype(np.float16)[None],
                     pose=data['poses'].astype(np.float16)[None],
                     betas=data['betas'].astype(np.float16)[None],
                     fps=30)
            written += 1
        total += written
        print(f'{name}: {len(matches)} matched, frame step {step}, {written} written')
    print(f'{total} motions in {out}')


if __name__ == '__main__':
    main()
