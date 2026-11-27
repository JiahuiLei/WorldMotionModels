"""Collect the clean test clips of the retargeting benchmark from the AMASS human chunks.

    python apps/retargeting/prepare_data/make_test_clips.py \
        --human_states release_data/retargeting/L3_train/data/amass_phuma_v2/g1/human_states \
        --split release_data/retargeting/L1_results/splits/amass_humanoid_v2_full_test.json \
        --out release_data/retargeting/L2_eval/amass_human_test

Copies `<clip>.npz` of every split entry (2014 clips; the table scores the first 2000).
"""

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--human_states', required=True, type=Path)
    parser.add_argument('--split', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()

    clips = [entry[0] for entry in json.loads(args.split.read_text())]
    missing = [c for c in clips if not (args.human_states / f'{c}.npz').is_file()]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(clips)} clips missing in {args.human_states}, e.g. {missing[:3]}')
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        shutil.copyfile(args.human_states / f'{clip}.npz', args.out / f'{clip}.npz')
    print(f'copied {len(clips)} clips to {args.out}')


if __name__ == '__main__':
    main()
