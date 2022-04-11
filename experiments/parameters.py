import datetime
from functools import reduce
import itertools
import logging
from operator import mul
from pathlib import Path
from typing import Dict, Iterable, List, TypeVar, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment import run_experiment
from instance import get_bks_value
import pandas_util as pdu


T = TypeVar('T')

TEST_COUNT = 10
OUTPUT_PATH = Path('experiments', 'params')

PLT_HD = (12.8, 7.2)
PLT_4K = (38.4, 21.6)


def define_parameters(
        problem: str,
        instances: List[str],
        params_possibilities: Dict[str, List[Union[str, float, int]]],
) -> Path:
    test_date = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    results = (
        run_experiment(problem, params, instances, TEST_COUNT)
        for params in __all_combinations(params_possibilities)
    )
    results = (r for r in results if r is not None and not r.empty)
    results = pd.concat(results)

    output = OUTPUT_PATH.joinpath(problem)
    output.mkdir(parents=True, exist_ok=True)
    file = output.joinpath(f'{test_date}.tsv')
    results.to_csv(file, index=False, sep='\t')
    return file


def __all_combinations(
        possibilities: Dict[str, List[T]],
) -> Iterable[Dict[str, T]]:
    keys = possibilities.keys()
    values = possibilities.values()
    logging.warning(f'#combinations: {reduce(mul, map(len, values))}')
    return (dict(zip(keys, comb)) for comb in itertools.product(*values))


def plot_results(cols: Union[str, List[str]], problem: str, file: Path):
    if isinstance(cols, str):
        cols = [cols]

    df = pd.read_csv(file, sep='\t')
    df['test_num'] = list(range(len(df)))

    # Converts the convergence to a list of ints
    df['convergence'] = [np.asarray(list(map(int, conv[1:-1].split(','))))
                         for conv in df['convergence']]

    # Normalize the values
    for instance in df['instance'].unique():
        bks = get_bks_value(problem, instance)
        df.loc[df['instance'] == instance, ['ans', 'convergence']] /= bks

        df.loc[df['instance'] == instance, 'elapsed'] /= (
            df.loc[df['instance'] == instance, 'elapsed'].min()
        )

    summary = (
        df
        .groupby(['instance'] + cols)
        .agg({
            'ans': ['min', 'max', 'mean', 'median', 'std'],
            'elapsed': ['min', 'max', 'mean', 'median', 'std'],
        })
        .reset_index()
    )

    base_path = OUTPUT_PATH.joinpath(problem)

    logging.info('Generating the answer view')
    plt.figure(figsize=PLT_HD)
    for instance in summary['instance'].unique():
        values = summary[summary['instance'] == instance]

        x = values[cols].astype(str).agg('#'.join, axis=1)
        y = values[('ans', 'mean')]
        e = values[('ans', 'std')]
        plt.errorbar(x, y, e, marker='o', label=instance)

    plt.xlabel(', '.join(cols))
    plt.ylabel('fitness / bks')
    plt.legend()
    plt.grid()
    plt.savefig(base_path.joinpath(f'{"#".join(cols)}-answer.png'))

    logging.info('Generating the time elapsed view')
    plt.figure(figsize=PLT_HD)
    for instance in summary['instance'].unique():
        values = summary[summary['instance'] == instance]

        x = values[cols].astype(str).agg('#'.join, axis=1)
        y = values[('elapsed', 'median')]
        e = values[('elapsed', 'std')]
        plt.errorbar(x, y, e, marker='o', label=instance)

    plt.xlabel(', '.join(cols))
    plt.ylabel('elapsed')
    plt.legend()
    plt.grid()
    plt.savefig(base_path.joinpath(f'{"#".join(cols)}-elapsed.png'))

    logging.info('Generating the convergence view')
    convergence = df.explode('convergence').reset_index(drop=True)
    convergence['iteration'] = 25 * convergence.groupby('test_num').cumcount()
    convergence = convergence[convergence['iteration'] <= 2000]
    for instance in df['instance'].unique():
        plt.figure(figsize=PLT_4K)
        for col_value in df[cols].drop_duplicates():
            convergence['iteration'] += 2
            values = (
                convergence
                .loc[(convergence['instance'] == instance) | (convergence[cols] == col_value)]
                .groupby('iteration')
                .agg({'convergence': ['mean', 'std']})
                .reset_index()
            )

            x = values[cols].astype(str).agg('#'.join, axis=1)
            y = values[('convergence', 'mean')]
            e = values[('convergence', 'std')]
            plt.errorbar(x, y, e, marker='.', label=f'{instance}={col_value}')

        plt.legend()
        plt.grid()
        plt.xlabel(', '.join(cols))
        plt.ylabel('convergence')
        plt.savefig(base_path.joinpath(
            f'{"#".join(cols)}-convergence-{instance}.png'))


def run_parameters(instances: List[str], params: Dict[str, Union[str, int, float]]):
    results = []
    tool = 'brkga-cuda'
    for params in experiment_params:
        params['threads'] = 256
        params['exchange-count'] = 1
        params['generations'] = 1000
        params['decode'] = 'host'
        params['tool'] = tool
        params['log-step'] = 50
        results.append(run_experiment('cvrp', params, instances, test_count=20))

    results = pd.concat(results)

    output = OUTPUT_PATH.joinpath('cvrp')
    output.mkdir(parents=True, exist_ok=True)

    test_date = results.iloc[0]['test_date']
    results.to_csv(output.joinpath(f'fixed-params-{test_date}.tsv'),
                   index=False, sep='\t')


def plot_crossed(problem: str, file: Path, params: List[str]):
    full_data = pd.read_csv(file, sep='\t')
    full_data = __normalize_results(full_data, params)

    # for instance in df['instance'].unique():
    #     bks = get_bks_value(problem, instance)
    #     df.loc[df['instance'] == instance, 'ans'] /= bks

    for tool, instance in pdu.tuples(full_data[['tool', 'instance']].drop_duplicates()):
        if instance != 'X-n1001-k43':
            continue
        data = (
            full_data
            .loc[(full_data['instance'] == instance) & (full_data['tool'] == tool)]
            .groupby(params)
            .agg({
                'ans': ['mean', 'std'],
                'elapsed': ['median', 'std'],
            })
        )

        x = data[('elapsed', 'median')]
        x_err = data[('elapsed', 'std')]
        y = data[('ans', 'mean')]
        y_err = data[('ans', 'std')]
        labels = data.reset_index()[params].astype(str).agg(' '.join, axis=1)

        plt.figure(figsize=PLT_HD)
        for xx, yy, xe, ye, lbl in zip(x, y, x_err, y_err, labels):
            plt.errorbar(xx, yy, xerr=xe, yerr=ye, linestyle='None', label=lbl, marker='o')
        plt.xlabel('time elapsed')
        plt.ylabel('fitness / bks')
        plt.title('Slowest/avg/fastest params + GPU-BRKGA params')
        plt.legend()
        plt.grid()
        print(f'analysis-{instance}.png')
        plt.savefig(OUTPUT_PATH.joinpath(problem, f'analysis-{instance}.png'))
        plt.close()


def __normalize_results(df: pd.DataFrame, params: List[str]) -> pd.DataFrame:
    df = (
        df
        .groupby(['tool', 'instance'] + params)
        .agg({
            'ans': 'mean',
            'elapsed': 'median',
        })
        .reset_index()
    )

    normalization = (
        df
        .groupby(['tool', 'instance'])
        .agg({
            'ans': ['min', 'max'],
            'elapsed': ['min', 'max'],
        })
        .reset_index()
    )

    df = df.merge(normalization, on=['tool', 'instance'])
    df['ans'] = (df['ans'] - df[('ans', 'min')]) / (df[('ans', 'max')] - df[('ans', 'min')])
    df['elapsed'] = (df['elapsed'] - df[('elapsed', 'min')]) / (df[('elapsed', 'max')] - df[('elapsed', 'min')])
    df = df.drop(columns=[('ans', 'min'), ('ans', 'max'), ('elapsed', 'min'), ('elapsed', 'max')])
    return df


if __name__ == '__main__':
    # plot_results('rho', 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-21T20:10:01.tsv'))
    # plot_results('exchange-interval', 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-22T00:21:10.tsv'))
    # plot_results('exchange-count', 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-22T16:59:29.tsv'))
    # plot_results('pop-count', 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-22T20:55:08.tsv'))
    # plot_results('elite', 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-23T20:12:13.tsv'))
    # plot_results('mutant', 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-23T23:34:24.tsv'))
    # plot_results(['pop-size', 'threads'], 'cvrp', OUTPUT_PATH.joinpath('cvrp', '2022-03-24T19:23:40.tsv'))

    plot_crossed(
        'cvrp',
        # OUTPUT_PATH.joinpath('cvrp', '2022-03-25T23:47:27.tsv'),
        OUTPUT_PATH.joinpath('cvrp', 'fixed-params-2022-04-01T09:16:51.tsv'),
        params=[
            'threads',
            'exchange-interval',
            'exchange-count',
            'pop-count',
            'pop-size',
            'elite',
            'mutant',
            'rho',
        ],
    )
    exit()

    # def cl_range(steps, max, start=None):
    #     def build():
    #         x = start or steps
    #         while x <= max:
    #             yield x
    #             x = round(x + steps, 3)
    #     return list(build())

    instances = ['X-n101-k25', 'X-n502-k39', 'X-n1001-k43']
    # define_parameters(
    #     'cvrp',
    #     instances,
    #     {
    #         'threads': [256],  # cl_range(128, 512),
    #         'generations': [10000],
    #         'exchange-interval': [25, 50],  # cl_range(5, 50, start=10),
    #         'exchange-count': [1],  # cl_range(1, 5),
    #         'pop-count': [3],  # cl_range(1, 8, 2),
    #         'pop-size': [128],  # cl_range(128, 512),
    #         'elite': [.05, .1],  # cl_range(.02, .2),
    #         'mutant': [.1, .15],  # cl_range(.02, .3),
    #         'rho': [.7, .75, .8],  # cl_range(.05, .95, start=.55),
    #         'decode': ['host-sorted'],
    #         'tool': ['brkga-cuda'],
    #         'log-step': [50],
    #     },
    # )

    experiment_params = [
        {'exchange-interval': 25, 'pop-count': 3, 'pop-size': 128, 'elite': 0.10, 'mutant': 0.10, 'rho': 0.75,},
        {'exchange-interval': 25, 'pop-count': 3, 'pop-size': 128, 'elite': 0.05, 'mutant': 0.10, 'rho': 0.75,},
        {'exchange-interval': 50, 'pop-count': 3, 'pop-size': 128, 'elite': 0.10, 'mutant': 0.15, 'rho': 0.80,},
        {'exchange-interval': 25, 'pop-count': 3, 'pop-size': 128, 'elite': 0.05, 'mutant': 0.15, 'rho': 0.80,},
        {'exchange-interval': 50, 'pop-count': 1, 'pop-size': 256, 'elite': 0.15625, 'mutant': 0.15625, 'rho': 0.70,},
        {'exchange-interval': 50, 'pop-count': 1, 'pop-size': 512, 'elite': 0.15625, 'mutant': 0.15625, 'rho': 0.70,},
        {'exchange-interval': 50, 'pop-count': 1, 'pop-size': 1024, 'elite': 0.15625, 'mutant': 0.15625, 'rho': 0.70,},
    ]

    run_parameters(instances, experiment_params)
