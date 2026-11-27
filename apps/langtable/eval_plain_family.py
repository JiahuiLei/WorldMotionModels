"""Run the plain Language Table evaluation for some task families only.

    python apps/langtable/eval_plain_family.py --families blocktoblock[,separate,...] \
        --checkpoint <ckpt> --config <yaml> --output <new folder> [--smoke_episodes N]

Running one family per process lets the plain row use several GPUs at once (`L2_eval_queue.py`).
The protocol of each family is unchanged: `code/plain/eval/langtable_official/eval_and_save_wmm.py`
builds one simulator per family with seed 0 and seeds every episode (simulator, NumPy and the
model's sampling noise) with its episode number, so a family's episodes do not depend on which
families ran before it in the same process. The one exception is the wording of `separate`
instructions, which the upstream reward draws from NumPy's unseeded global generator and which is
random per run in any case.

`--smoke_episodes N` runs only the first N episodes of each family (quick check of the setup);
the run then records `smoke_override_active: true` in its `rollout_config.json`.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / 'code' / 'plain' / 'eval' / 'langtable_official' / 'eval_and_save_wmm.py'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--families', required=True, help='comma-separated reward-factory folder names')
    parser.add_argument('--smoke_episodes', type=int, default=None, help='run only the first N episodes per family')
    args, rest = parser.parse_known_args()

    spec = importlib.util.spec_from_file_location('eval_and_save_wmm', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wanted = args.families.split(',')
    unknown = [f for f in wanted if f not in module.REWARDS]
    if unknown:
        raise SystemExit(f'unknown families {unknown}; expected some of {list(module.REWARDS)}')
    module.REWARDS = {name: cls for name, cls in module.REWARDS.items() if name in wanted}
    if args.smoke_episodes is not None:
        module._NUM_ROLLOUTS_SMOKE = args.smoke_episodes

    sys.argv = [str(SCRIPT)] + rest
    module.main()


if __name__ == '__main__':
    main()
