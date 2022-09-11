import logging
from pathlib import Path
import time
import pandas as pd


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s.%(msecs)03d] [%(levelname)8s]"
           " %(filename)s:%(lineno)s: %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S',
)
logging.Formatter.converter = time.gmtime

CATEGORIES = ['test_time', 'commit', 'tool', 'problem', 'instance',
              'decoder', 'system', 'cpu', 'gpu', 'nvcc', 'g++']


def save_results(df: pd.DataFrame, path: Path):
    dest_file = path.with_suffix('.zip')
    logging.info("Saving results to the file %s (%s)", dest_file, path.name)

    if path.suffix != '.tsv':
        raise ValueError("Only .tsv files are supported")

    logging.debug("Compress convergence to save")
    df.loc[:, 'convergence'] = df['convergence'].apply(__compress_convergence)

    logging.debug("Saving compressed results to %s", dest_file)
    df.to_csv(dest_file, sep='\t', index=False,
              compression={'method': 'zip', 'archive_name': path.name})


def __compress_convergence(convergence: str) -> str:
    compressed = []
    previous = ''
    begin = None
    gen = 0
    for i in range(len(convergence)):
        if convergence[i] == '(':
            begin = i + 1
        elif convergence[i] == ')':
            gen += 1
            fitness, elapsed = convergence[begin: i].split(',')
            if fitness != previous:
                compressed.append(f"({float(fitness):g},{float(elapsed):g},{gen})")
                previous = fitness

    return '[' + ','.join(compressed) + ']'


def read_results(path: Path) -> pd.DataFrame:
    logging.info("Reading result file %s", path)
    types = {col: 'category' for col in CATEGORIES}
    types['ans'] = 'float'
    types['elapsed'] = 'float'
    return pd.read_csv(path, sep='\t', dtype=types)
