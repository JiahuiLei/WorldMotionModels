"""Layer 3 smoke test: load the released training code, config and data, draw a few training batches,
and build the model (on the CPU, without training).

    python apps/langtable/L3_smoke.py [--data_root release_data/langtable/L3_train] [--batches 3]

Needs no GPU. It prints the shape of each tensor in the first batches and the model size, and checks
that every project module is imported from apps/langtable/code/train/.
"""

import argparse
import os
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
CODE = APP / 'code' / 'train'
CONFIG = CODE / 'configs' / 'wmm' / 'lt' / 'w1.0.3_b8all.yaml'
PROJECT_PACKAGES = ('umm', 'language_table', 'lib_io', 'lib_mani', 'lib_human', 'pytorch_kinematics',
                    'prepare_data', 'eval')


def code_paths():
    paths = [CODE, CODE / 'src']
    for lib in sorted((CODE / 'ext').iterdir()):
        paths.append(lib / 'src' if (lib / 'src').is_dir() else lib)
    return [str(p) for p in paths]


def describe(x):
    if hasattr(x, 'shape'):
        return f'{tuple(x.shape)} {x.dtype}'
    if isinstance(x, (list, tuple)):
        return f'{type(x).__name__}[{len(x)}]'
    return type(x).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data_root', default=str(APP.parent.parent / 'release_data' / 'langtable' / 'L3_train'),
                        help='folder holding data_wmm/ and data_splits/ (the paths in the training config are relative to it)')
    parser.add_argument('--batches', type=int, default=3)
    args = parser.parse_args()

    sys.path[:0] = code_paths()
    data_root = Path(args.data_root).resolve()
    os.chdir(data_root)
    # The dataset module imports a ManiSkill helper that creates a SAPIEN renderer at import time;
    # without a GPU that crashes unless SAPIEN is told to render on the CPU. The smoke test renders nothing.
    os.environ.setdefault('SAPIEN_RENDER_DEVICE', 'cpu')
    os.environ.setdefault('SAPIEN_DISABLE_RAY_TRACING', '1')
    # The model's T5 text encoder comes from the Layer 2 copy, as in L3_train.sh.
    os.environ.setdefault('HF_HOME', str(data_root.parent / 'L2_eval' / 'hf_home'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    t0 = time.time()
    from umm.training_utils import get_config
    from umm.datasets import get_data_module
    from umm.models import get_model

    cfg = get_config(['--config', str(CONFIG), '--workers', '0'])
    dm = get_data_module(cfg)
    dm.setup('fit')
    loader = dm.train_dataloader()
    print(f'train batch size {cfg.train_batchsize}; setup took {time.time() - t0:.0f} s')
    for i, batch in enumerate(loader):
        items = batch.items() if isinstance(batch, dict) else enumerate(batch)
        print(f'batch {i}:', {k: describe(v) for k, v in items})
        if i + 1 >= args.batches:
            break

    model = get_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f'model {type(model).__module__}.{type(model).__name__}: {n / 1e6:.1f} M parameters (training model + EMA copy + frozen T5)')

    outside = sorted({m.__file__ for name, m in list(sys.modules.items())
                      if name.split('.')[0] in PROJECT_PACKAGES and getattr(m, '__file__', None)
                      and not Path(m.__file__).resolve().is_relative_to(CODE)})
    if outside:
        raise SystemExit(f'project modules loaded from outside {CODE}: {outside}')
    print(f'OK: {args.batches} batches in {time.time() - t0:.0f} s; all project modules come from {CODE}')


if __name__ == '__main__':
    main()
