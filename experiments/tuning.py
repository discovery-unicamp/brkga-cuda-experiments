from bisect import bisect_right
from typing import Iterable, List, Tuple
import pandas as pd

from pathlib import Path
from result import read_results


first_tuning = pd.concat(list(map(read_results, [
    Path('results', '2022-10-04T15:08:00.zip'),
])))
first_tuning.loc[:, 'convergence'] = first_tuning['convergence'].apply(eval)


def first_phase():
    params = [
        'tool', 'problem', 'instance', 'decoder', 'omp-threads', 'threads',
        'generations', 'max-time', 'pop-count', 'pop-size', 'rhoe', 'elite',
        'mutant', 'exchange-interval', 'exchange-count', 'pr-interval',
        'pr-pairs', 'pr-block-factor', 'similarity-threshold', 'prune-interval',
        'log-step', 'commit',
    ]
    df = (
        first_tuning
        .groupby('combination_id')
        .agg({
            **{p: 'first' for p in params},
            'ans': 'mean',
            'elapsed': 'median',
            'seed': 'count',
            'convergence': __apc,
        })
    )
    breakpoint()


def __apc(
        values: Iterable[List[Tuple[float, float, int]]],
        by_time: bool = True,
):
    """Average Performance Curve"""
    # breakpoint()
    if by_time:
        id_column = 1
    else:
        id_column = 2

    def bs(conv, elapsed):
        k = bisect_right(conv, elapsed, key=lambda c: c[id_column])
        return float('inf') if k == 0 else conv[k - 1][0]

    values = list(values)
    indexes = sorted(set(v[id_column] for line in values for v in line))

    n = len(values)
    apc = []
    for i in indexes:
        ans = [bs(values[k], i) for k in range(n)]
        avg_ans = sum(ans) / n
        if not apc or abs(avg_ans - apc[-1][0]) > 1e-3:
            apc.append((avg_ans, i))

    return apc


if __name__ == '__main__':
    first_phase()
