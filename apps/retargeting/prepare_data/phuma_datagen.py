"""PHUMA retargeting as used to build the WMM training data (shared by retarget_amass.py and retarget_embody3d.py).

Each motion is cut into 20 s chunks with 1 s overlap and physically curated by PHUMA. For every chunk the
human motion is written to `<save_root>/human_states/<id>_<start>_<end>.npz` (SMPL-X neutral, 300 betas,
float16: `Q` (1, T, 55, 4, 4) joint poses, `global_t` (1, T, 3), `pose` (1, T, 165) axis-angle, `betas`
(1, 1, 300), `valid_mask` (1, T), `fps` 30). Chunks that pass curation are retargeted to the robot by PHUMA's
optimisation and written to `<save_root>/<robot>_states/<id>_<start>_<end>.npz` (`Q` (T, L, 4, 4) float16
link poses, `names` (L-1,), `fps` 30) and `<save_root>/<robot>_motion_pkl/*.pkl` (root pose and joint
angles). `<save_root>/robot_meshes.npz` holds the robot link meshes.

The PHUMA library comes from the third-party download (`code_ext/baselines/data_processing/lib_phuma`). The
training data were built with `PHUMA/src/retarget/phuma_processor_datagen.py`, which weights all body
keypoints equally; `phuma_processor.py` next to it (used for the PHUMA baseline row) weights hands and feet
20 times higher.

With `--human_only` the robot optimisation is skipped and only `human_states` is written (minutes instead of
hours; the human files are the same as in a full run). This is all the Layer-2 test clips need.

Every worker runs single-threaded on one pinned CPU, so a chunk's result does not depend on how many workers
run or in which order the chunks are processed.
"""

import multiprocessing as mp
import os
import sys
from pathlib import Path

for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_var] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

APP = Path(__file__).resolve().parent.parent
ROBOT_DIRS = {'g1': 'g1', 'h1_2': 'h12'}


def add_common_args(parser) -> None:
    default_ext = APP.parent.parent / 'release_data' / 'retargeting' / 'code_ext'
    parser.add_argument('--save_root', required=True, type=Path)
    parser.add_argument('--robot_name', choices=sorted(ROBOT_DIRS), default='g1')
    parser.add_argument('--code_ext', type=Path, default=default_ext,
                        help='third-party code folder (release_data/retargeting/code_ext)')
    parser.add_argument('--smplx_dir', type=Path,
                        default=APP.parent.parent / 'release_data' / 'retargeting' / 'L2_eval' / 'smpls' / 'smplx',
                        help='folder holding SMPLX_NEUTRAL.npz (SMPL-X download)')
    parser.add_argument('--workers', type=int, default=max(1, len(os.sched_getaffinity(0)) - 2))
    parser.add_argument('--resume', action='store_true', help='skip chunks whose robot output exists')
    parser.add_argument('--human_only', action='store_true', help='write human_states only, no robot retargeting')


class _NoRetarget:
    def retarget(self, **kwargs):
        return None


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def setup_paths(code_ext: Path, smplx_dir: Path) -> None:
    """Lay out the SMPL-X model where the PHUMA code looks for it and put the library on sys.path."""
    root = code_ext.resolve() / 'baselines'
    if not (smplx_dir / 'SMPLX_NEUTRAL.npz').is_file():
        raise SystemExit(f'SMPLX_NEUTRAL.npz not found in {smplx_dir}')
    lib = root / 'data_processing' / 'lib_phuma'
    _link(smplx_dir.parent, root / 'data' / 'smpls')
    _link(smplx_dir, lib / 'PHUMA' / 'asset' / 'human_model' / 'smplx')
    sys.path[:0] = [str(lib), str(root / 'envs' / 'lib_twist' / 'pytorch_kinematics' / 'src')]


def make_wrapper(args):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from phuma_wrapper import PhumaWrapper
    from PHUMA.src.retarget.phuma_processor_datagen import PHUMA_Processor

    robot = ROBOT_DIRS[args.robot_name]
    wrapper = PhumaWrapper(
        root_dir=str(args.save_root), human_dir_name='human_states', robot_dir_name=f'{robot}_states',
        robot_pkl_name=f'{robot}_motion_pkl', fps=30, robot_name=args.robot_name, resume=args.resume,
        curation_chunk_len_seconds=20.0, curation_chunk_overlap_seconds=1.0)
    wrapper.retargetor = PHUMA_Processor(device='cpu', fps=30)
    if args.human_only:
        wrapper.retargetor = _NoRetarget()
        wrapper.__save_robot__ = lambda *a, **k: None
    return wrapper


_STATE = {}


def _init(args, process_job, counter):
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    cpus = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpus[index % len(cpus)]})
    setup_paths(args.code_ext, args.smplx_dir)
    _STATE.update(wrapper=make_wrapper(args), process_job=process_job)


def _run(job):
    try:
        _STATE['process_job'](_STATE['wrapper'], job)
        return job, None
    except Exception as e:  # noqa: BLE001 - one failing motion must not stop the others; it is reported below
        return job, repr(e)


def run_jobs(args, jobs, process_job) -> None:
    """Run process_job(wrapper, job) for every job on args.workers single-threaded processes.

    PHUMA rejects a few motions (e.g. an assertion inside its curation); they produce no output, as in the
    released training data. They are listed in `<save_root>/failed.txt`.
    """
    args.save_root.mkdir(parents=True, exist_ok=True)
    counter = mp.Value('i', 0)
    failed = []
    with mp.get_context('fork').Pool(args.workers, initializer=_init, initargs=(args, process_job, counter)) as pool:
        for i, (job, err) in enumerate(pool.imap_unordered(_run, jobs), start=1):
            if err:
                failed.append(f'{job}\t{err}')
                print(f'FAILED {job}: {err}', flush=True)
            if i % 100 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} motions done', flush=True)
    (args.save_root / 'failed.txt').write_text(''.join(f'{line}\n' for line in failed))
    print(f'done: {len(jobs) - len(failed)} motions processed, {len(failed)} failed '
          f'(listed in {args.save_root / "failed.txt"})')
