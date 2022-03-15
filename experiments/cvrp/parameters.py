from asyncio.log import logger
import datetime
from functools import reduce
import itertools
import logging
from operator import mul
from pathlib import Path
from typing import Dict, Iterable, List, TypeVar, Union

import pandas as pd
from experiment import run_experiment


T = TypeVar('T')

TEST_COUNT = 10
OUTPUT_PATH = Path('experiments', 'params')


def define_parameters(
        application: str,
        instances: List[str],
        params_possibilities: Dict[str, List[Union[str, float, int]]],
        ):
    test_date = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    results = (
        run_experiment(application, params, instances, TEST_COUNT)
        for params in __all_combinations(params_possibilities)
    )
    results = (r for r in results if r is not None and not r.empty)
    results = pd.concat(results)

    output = OUTPUT_PATH.joinpath(application)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output.joinpath(f'{test_date}.tsv'),
                index=False, sep='\t')


def __all_combinations(
        possibilities: Dict[str, List[T]],
        ) -> Iterable[Dict[str, T]]:
    keys = possibilities.keys()
    values = possibilities.values()
    logging.warning(f'Number of possibilities: {reduce(mul, map(len, values))}')
    return (dict(zip(keys, comb)) for comb in itertools.product(*values))


if __name__ == '__main__':
    def cl_range(steps, max, start=None):
        def build():
            x = start or steps
            while x <= max:
                yield x
                x += steps
        return list(build())

    define_parameters(
        'cvrp',
        ['X-n101-k25', 'X-n502-k39', 'X-n1001-k43'],
        {
            'threads': [256],
            'generations': [1000],
            'exchange-interval': [25],
            'exchange-count': [2],
            'pop_count': [4],
            'pop_size': [256],
            'elite': [.1],
            'mutant': [.1],
            'rho': cl_range(.025, .9, start=.6),
            'decode': ['host-sorted'],
            'tool': ['brkga-cuda'],
            'log-step': [25],
        },
    )
