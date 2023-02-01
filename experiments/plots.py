from bisect import bisect_right
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar
import matplotlib.pyplot as plt

import pandas as pd

from result import PARAMS, read_results


T = TypeVar('T')


# FILES = [
#     # 'BRKGA-MP-IPR+PR.tsv',
#     # 'BRKGA-CUDA+PR+pruning.zip',
#     # 'BRKGA-MP-IPR+PR.zip',
#     # 'BRKGA-CUDA+pruning.zip',
#     # 'BRKGA-CUDA.zip',
#     # 'BRKGA-CUDA+CEA-LS.tsv',
# ]
# data = pd.concat((read_results(Path(f'results/{file}')) for file in FILES),
#                  ignore_index=True)
data = pd.read_csv('results/v5.tsv', sep='\t')

COLORS = ['red', 'green', 'blue', 'purple', 'orange']

FITNESS_ELAPSED = [1, 2, 3, 5, 15, 30, 60, 120, 180]
FITNESS_VAR = [
    'rhoe',
    'elite',
    'similarity-threshold',
]
SMOOTH_PLOT = False
PLOT_GENERATIONS = False


def plot_convergence():
    plot_by = ['problem', 'instance']
    plot_params = [p for p in PARAMS if p not in plot_by + ['seed']]
    label_color = {}
    for _, plot_condition in data[plot_by].drop_duplicates().iterrows():
        plt.figure(figsize=(10.80, 7.20))

        mask = True
        for p in plot_by:
            mask = mask & (data[p] == plot_condition[p])
        df = build_apc(data.loc[mask])

        for _, row in df.iterrows():
            fitness, elapsed, generation = zip(*row['convergence'])
            if not SMOOTH_PLOT:
                fitness = [y
                           for i, x in enumerate(fitness)
                           for y in (1 if i == len(fitness) - 1 else 2) * [x]]
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
            if label not in label_color:
                assert COLORS
                label_color[label] = COLORS.pop()

            plt.plot(
                (generation if PLOT_GENERATIONS else elapsed),
                fitness,
                label=label,
                color=label_color[label],
            )

        name = '-'.join(plot_condition[p] for p in plot_by)

        plt.title(f"{name} convergence")
        plt.xlabel("Generation" if PLOT_GENERATIONS else "Time (s)")
        plt.ylabel("Fitness")
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.minorticks_on()
        plt.grid(axis='y', which='minor', linestyle=':')
        plt.grid(axis='y', which='major', linestyle='-')
        plt.grid(axis='x', which='major', linestyle='--')
        plt.grid(axis='x', which='minor', linestyle=':')

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

    assert not df.empty  # Avoids breaking inside `eval_apc`
    return (
        df
        .groupby([p for p in PARAMS if p != 'seed'])
        .apply(eval_apc)
        .rename('convergence')
        .reset_index()
    )


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


def greedy_vs_optimal_cvrp():
    df = data.loc[data['problem'].isin(['cvrp', 'cvrp_greedy'])]
    assert len(df['problem'].unique() == 2)

    markers = ['^', 'v', '<', '>', 'o', 's', 'p', 'P', '*', 'D', ]

    df = df.groupby(['tool', 'problem', 'instance'])['ans'].mean().reset_index()

    instances =  sorted(df['instance'].unique(),
                        key=lambda x: int(x[3:].split('-')[0]))
    instances_dict = {name: k for k, name in enumerate(instances)}
    df['x'] = df['instance'].apply(lambda x: instances_dict[x])
    df = df.rename(columns={'ans': 'y'})

    fig = plt.figure(figsize=(10.80, 7.20))
    markers = ['o', '.']
    colors = ['grey', 'black']
    for i, problem in enumerate(df['problem'].unique()):
        x = df.loc[df['problem'] == problem, 'x']
        y = df.loc[df['problem'] == problem, 'y']
        label = 'Optimal' if problem == 'cvrp' else 'Greedy'
        label = f'{label} decoder'
        plt.scatter(x, y, marker=markers[i], c=colors[i], label=label)

    plt.legend()
    plt.xticks(range(len(instances)), instances, rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.2)
    plt.title('CVRP: greedy vs. optimal decoder')
    plt.xlabel('Instance')
    plt.ylabel('Fitness')
    plt.grid(axis='x', which='major', linestyle='--')

    fig_file = Path('plots/greedy-vs-optimal-decoder.eps')
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_file, bbox_inches='tight')
    plt.savefig(fig_file.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # plot_convergence()
    # fitness()
    greedy_vs_optimal_cvrp()
