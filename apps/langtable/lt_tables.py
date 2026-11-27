"""Shared table code for the Language Table notebooks (L1_tables.ipynb and L2_compare.ipynb)."""

import json
import os
import re
import statistics
from pathlib import Path

import pandas as pd

FAMILIES = {  # paper column -> reward-factory folder name
    'B2B': 'blocktoblock',
    'B2AL': 'blocktoabsolutelocation',
    'B2BRL': 'blocktoblockrelativelocation',
    'B2RL': 'blocktorelativelocation',
    'Sep': 'separate',
}
COLUMNS = list(FAMILIES) + ['Avg']
N_EPISODES = 50
EPISODE_GLOB = {'plain': '*/meta.json', 'mpc': 'ep_*/meta.json'}

PAPER_TABLE1 = {  # transcribed from the paper, Table 1 (Language Table)
    'Scratch':         [4.0, 6.0, 10.0, 6.0, 52.0, 15.6],
    'UniPi':           [14.0, 4.0, 12.0, 10.0, 72.0, 22.4],
    'VPT':             [36.0, 38.0, 48.0, 26.0, 70.0, 43.6],
    'LAPA':            [58.0, 56.0, 52.0, 48.0, 96.0, 62.0],
    'ActionVLA':       [76.0, 72.0, 76.0, 70.0, 90.0, 76.8],
    'LCB':             [90.0, 71.0, 66.0, 73.0, 99.0, 80.0],
    'LangTable orig.': [88.0, 65.0, 70.0, 68.0, 94.0, 77.0],
    'WMM (ours)':      [90.0, 76.0, 80.0, 90.0, 100.0, 87.2],
}
QUOTED_FROM = {
    'Scratch': 'LAPA paper', 'UniPi': 'LAPA paper', 'VPT': 'LAPA paper',
    'LAPA': 'LAPA paper', 'ActionVLA': 'LAPA paper',
    'LCB': 'LCB paper', 'LangTable orig.': 'LCB paper',
}

STATS = ['Succ.', 'Mean steps', 'Med. steps', 'Min steps', 'Max steps']
PAPER_TABLE2 = {  # transcribed from the paper, Table 2 (Plain Policy versus MPC)
    ('B2B', 'Plain'):  [90.0, 79.5, 73.0, 26, 198],
    ('B2B', 'MPC'):    [100.0, 79.1, 68.5, 16, 175],
    ('B2AL', 'Plain'): [76.0, 92.7, 87.0, 14, 187],
    ('B2AL', 'MPC'):   [80.0, 87.2, 85.5, 2, 171],
}
DIGITS = {'Succ.': 1, 'Mean steps': 1, 'Med. steps': 1, 'Min steps': 0, 'Max steps': 0}


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / 'apps' / 'langtable').is_dir():
            return p
    raise FileNotFoundError(f'no folder containing apps/langtable above {start}')


def langtable_data(repo_root: Path) -> Path:
    """`$WMM_DATA/langtable`, or `<repo>/release_data/langtable` when WMM_DATA is unset."""
    root = Path(os.environ['WMM_DATA']) if 'WMM_DATA' in os.environ else repo_root / 'release_data'
    return root / 'langtable'


def load_episodes(run_dir: Path, mode: str) -> pd.DataFrame:
    """One row per episode of a WMM run (`mode` is 'plain' or 'mpc'), read from its meta.json files."""
    rows = []
    for col, folder in FAMILIES.items():
        files = sorted((run_dir / folder).glob(EPISODE_GLOB[mode]))
        assert len(files) == N_EPISODES, (str(run_dir), folder, len(files))
        for f in files:
            m = json.loads(f.read_text())
            rows.append({'mode': mode, 'family': col, 'ep': m['ep_num'],
                         'success': bool(m['succ']), 'steps': int(m['succ_steps']),
                         'instruction': m['instruction']})
    df = pd.DataFrame(rows)
    for col in FAMILIES:
        assert sorted(df[df.family == col].ep) == list(range(N_EPISODES)), (str(run_dir), col)
    assert ((df.steps == -1) == ~df.success).all(), 'a failed episode must have steps == -1'
    return df


def check_results_json(episodes: pd.DataFrame, run_dir: Path, mode: str) -> None:
    """Assert that the run's own results.json agrees with its per-episode files."""
    summary = json.loads((run_dir / 'results.json').read_text())
    ep_key, ok_key, st_key = ('ep', 'success', 'steps') if mode == 'plain' else ('ep_num', 'succ', 'succ_steps')
    for col, folder in FAMILIES.items():
        mine = episodes[episodes.family == col].set_index('ep')
        theirs = summary['rewards'][folder]
        assert int(mine.success.sum()) == theirs['successes'] and len(mine) == theirs['total'], (mode, col)
        for e in theirs['episodes']:
            row = mine.loc[e[ep_key]]
            assert row.success == e[ok_key] and row.steps == e[st_key], (mode, col, e)


def load_orig_episodes(run_dir: Path) -> pd.DataFrame:
    """Episodes of the original language_table/eval/main.py, read from its video names
    `videos/<reward>_<ep>_<success|failure>.mp4` (the script saves nothing else per episode)."""
    folder_to_col = {v: k for k, v in FAMILIES.items()}
    pat = re.compile(r'^(?P<reward>[a-z]+)_(?P<ep>\d+)_(?P<res>success|failure)\.mp4$')
    rows = []
    for f in sorted((run_dir / 'videos').glob('*.mp4')):
        m = pat.match(f.name)
        assert m, f.name
        rows.append({'family': folder_to_col[m['reward']], 'ep': int(m['ep']), 'success': m['res'] == 'success'})
    df = pd.DataFrame(rows)
    for col in FAMILIES:
        assert sorted(df[df.family == col].ep) == list(range(N_EPISODES)), (str(run_dir), col)
    return df


def success_rates(episodes: pd.DataFrame) -> dict:
    """Success rate (%) per family plus their plain mean `Avg`."""
    rates = {col: 100.0 * episodes[episodes.family == col].success.mean() for col in FAMILIES}
    rates['Avg'] = sum(rates[col] for col in FAMILIES) / len(FAMILIES)
    return rates


def policy_stats(episodes: pd.DataFrame) -> list:
    """[success %, mean, median, min, max steps over successful episodes] for one family of one run."""
    ok = episodes[episodes.success].steps.tolist()
    return [100.0 * episodes.success.mean(), statistics.mean(ok), statistics.median(ok), min(ok), max(ok)]
