from bisect import bisect_right
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar
import matplotlib.pyplot as plt

import pandas as pd

from result import PARAMS, read_results


T = TypeVar('T')


FILES = [
    'BRKGA-MP-IPR-old.zip',
    'BRKGA-CUDA+PR+pruning.zip',
    # 'BRKGA-MP-IPR+PR.zip',
    'BRKGA-CUDA+pruning.zip',
    'BRKGA-CUDA.zip',
]
DATA = {file: read_results(Path(f'results/{file}')) for file in FILES}

COLORS = ['red', 'green', 'blue', 'purple']

FITNESS_ELAPSED = [1, 2, 3, 5, 15, 30, 60, 120, 180]
FITNESS_VAR = [
    'rhoe',
    'elite',
    'similarity-threshold',
]
SMOOTH_PLOT = True
PLOT_GENERATIONS = False


def plot_convergence():
    df = DATA[FILES[0]]
    instances = df[['problem', 'instance']].drop_duplicates()
    # instances = instances[instances['instance'].isin(['X-n219-k73'])]
    color_label = {}

    plot_params = list(PARAMS - {'problem', 'instance', 'seed'})

    for _, inst in instances.iterrows():
        plt.figure(figsize=(10.80, 7.20))
        problem = inst['problem']
        instance = inst['instance']

        for input_file in FILES:
            df = DATA[input_file]
            df = df.loc[(df['instance'] == instance)]
            df = df.loc[(df['problem'] == problem)]
            df = build_apc(df)

            for _, row in df.iterrows():
                fitness, elapsed, generation = zip(*row['convergence'])
                if not SMOOTH_PLOT:
                    fitness = [y
                               for i, x in enumerate(fitness)
                               for y in (1 if i == len(fitness) - 1 else 2)
                                        * [x]]
                    elapsed = [y
                               for i, x in enumerate(elapsed)
                               for y in (1 if i == 0 else 2) * [x]]
                    generation = [y
                                  for i, x in enumerate(generation)
                                  for y in (1 if i == 0 else 2) * [x]]

                # if not PLOT_GENERATIONS and elapsed[-1] < row['max-time']:
                #     elapsed.append(row['max-time'])
                #     fitness.append(fitness[-1])

                label = '_'.join(map(str, row[plot_params]))
                if label not in color_label:
                    assert COLORS
                    color_label[label] = COLORS.pop()

                plt.plot(
                    (generation if PLOT_GENERATIONS else elapsed),
                    fitness,
                    label=label,
                    color=color_label[label],
                )

        name = f"{problem}-{instance}"

        if problem == 'scp':
            plt.ylim((500, 1500))

        plt.title(f"{name} convergence")
        plt.xlabel("Generation" if PLOT_GENERATIONS else "Time (s)")
        plt.ylabel("Fitness")
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.minorticks_on()
        plt.grid(axis='y', which='minor', linestyle=':')
        plt.grid(axis='y', which='major', linestyle='-')
        plt.grid(axis='x', which='major', linestyle=':')

        outfile = Path('convergence', name + ".png")
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close()


def build_apc(df: pd.DataFrame) -> pd.DataFrame:
    def eval_apc(df: pd.DataFrame):
        last_result = df['convergence'].apply(lambda conv: conv[-1])
        max_elapsed = float(max(x[1] for x in last_result))
        max_generation = int(max(x[2] for x in last_result))
        step = max_elapsed / min(400, max_generation)
        apc = []
        i = 0
        while i <= max_elapsed + 1e-7:
            convergence = [find_conv(conv, i) for conv in df['convergence']]
            apc.append((
                mean(c[0] for c in convergence),  # fitness
                i,  # elapsed
                mean(c[2] for c in convergence),  # generation
            ))
            i += step

        return apc

    df = df.groupby(list(PARAMS - {'seed'})).apply(eval_apc).reset_index()
    df = df.rename(columns={0: 'convergence'})
    return df


def find_conv(conv: List[Tuple[float, float, int]], elapsed: float):
    assert len(conv) > 0
    k = bisect_right(conv, elapsed, key=lambda c: c[1])
    if k == 0:
        # No value found before elapsed
        return float('inf'), float(0), int(0)
    assert conv[k - 1][1] <= elapsed
    assert k == len(conv) or conv[k][1] > elapsed
    return conv[k - 1]


def mean(iterable: Iterable[T]) -> T:
    lst = list(iterable)
    return sum(lst) / len(lst)


if __name__ == '__main__':
    plot_convergence()
    # fitness()
