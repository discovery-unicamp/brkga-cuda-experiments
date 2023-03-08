import logging
from pathlib import Path
import sys
import time
from typing import List
import pandas as pd

from shell import shell


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)8s]"
           " %(filename)s:%(lineno)s: %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logging.Formatter.converter = time.gmtime

CATEGORIES = ['test_time', 'commit', 'tool', 'problem', 'instance',
              'decoder', 'system', 'cpu', 'gpu', 'nvcc', 'g++']
COMMIT_FILE = Path('.commit')

INFO_SCRIPTS = {
    'system': 'uname -v',
    'cpu': 'cat /proc/cpuinfo | grep "model name" | uniq | cut -d" " -f 3-',
    'cpu-memory': "grep MemTotal /proc/meminfo | awk '{print $2 / 1024}'",
    'gpu': "nvidia-smi -L | grep -oP 'NVIDIA.*(?= \(UUID)'"
           ' | sed -n "$(([DEVICE] + 1)) p"',
    'gpu-memory': "nvidia-smi -i [DEVICE] | grep -m1 -oP '[0-9]*(?=MiB)'"
                  " | tail -n1",
    'nvcc': 'nvcc --version | grep "release" | grep -o "V.*"',
    'g++': 'g++ --version | grep "g++"',
}

PARAMS = {
    'tool',
    'problem',
    'instance',
    'seed',
    'threads',
    'pop-count',
    'pop-size',
    'rhoe-function',
    'parents',
    'elite-parents',
    'elite',
    'mutant',
    'exchange-interval',
    'exchange-count',
    'pr-interval',
    'pr-pairs',
    'pr-block-factor',
    'pr-min-diff',
    'pr-max-time',
    'pr-select',
    'prune-interval',
    'prune-threshold',
}


def save_results(
        df: pd.DataFrame,
        path: Path,
        system: List[str] = [],
        device: int = 1,
):
    dest_file = path.with_suffix('.zip')
    logging.info("Saving results to the file %s (%s)", dest_file, path.name)

    if path.suffix != '.tsv':
        raise ValueError("Only .tsv files are supported")

    logging.debug("Add system info")
    for info in system:
        if info == 'commit':
            # FIXME this was applied due to permission error on git
            df.loc[:, info] = COMMIT_FILE.read_text().split('\n')[0]
        else:
            script = INFO_SCRIPTS[info].replace('[DEVICE]', str(device))
            df.loc[:, info] = shell(script)

    logging.debug("Sort values and columns to ease view")
    df = df.sort_values(by=['tool', 'problem', 'instance'])

    columns = ['start_time', 'tool', 'problem', 'instance', 'seed', 'ans',
               'elapsed']
    columns += [c for c in df.columns if c not in columns and c not in system]
    columns += system
    df = df.loc[:, columns]

    logging.debug("Saving compressed results to %s", dest_file)
    df.to_csv(dest_file, sep='\t', index=False,
              compression={'method': 'zip', 'archive_name': path.name})


def compress_convergence(convergence: str) -> str:
    compressed = []
    previous = ''
    begin = None
    ignored = False
    fitness, elapsed, generation = None, None, None
    for i, symbol in enumerate(convergence):
        if symbol == '(':
            begin = i + 1
        elif symbol == ')':
            data = convergence[begin: i].split(',')
            assert len(data) == 3

            fitness = float(data[0])
            elapsed = float(data[1])
            generation = int(data[2])

            begin = None
            ignored = fitness == previous
            if not ignored:
                compressed.append(f"({fitness:g},{elapsed:g},{generation})")
                previous = fitness

    if ignored:
        # Save the last result to keep the time elapsed/last generation
        compressed.append(f"({fitness:g},{elapsed:g},{generation})")

    return f"[{','.join(compressed)}]"


def read_results(path: Path) -> pd.DataFrame:
    logging.info("Reading result file %s", path)
    types = {col: 'category' for col in CATEGORIES}
    types['ans'] = 'float'
    types['elapsed'] = 'float'

    results_df = pd.read_csv(path, sep='\t', dtype=types)
    logging.debug("Found %d entries", len(results_df))

    results_df = results_df.rename(columns={
        'decode': 'decoder',
    })

    results_df.loc[:, 'convergence'] = (
        results_df['convergence']
        .str
        .replace('inf', "float('inf')")
        .apply(eval)
    )

    missing_params = PARAMS - set(results_df.columns)
    for p in missing_params:
        results_df[p] = 0
    return results_df
