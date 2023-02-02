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

FIG_SIZE = (10.80, 7.20)
COMPARE_TO = 'brkga-api (cpu)'
COLORS = ['red', 'green', 'blue', 'purple', 'orange']

FITNESS_ELAPSED = [1, 2, 3, 5, 15, 30, 60, 120, 180]
FITNESS_VAR = [
    'rhoe',
    'elite',
    'similarity-threshold',
]
SMOOTH_PLOT = False
PLOT_GENERATIONS = False

PLOTS = {
    'brkga-api (cpu)':
        ('BRKGA-API (CPU)', 'black', '.', 0),
    'gpu-brkga (cpu)':
        ('GPU-BRKGA (CPU)', 'blue', '.', 1),
    'gpu-brkga-fix (cpu)':
        ('GPU-BRKGA-Fixed (CPU)', 'cyan', '.', 2),
    'brkga-cuda-1.0 (cpu)':
        ('BrkgaCuda 1.0 (CPU)', 'pink', '*', 3),
    'brkga-cuda-2.0 (cpu)':
        ('BrkgaCuda 2.0 (CPU)', 'green', (3, 2, 0), 4),
    'brkga-cuda-2.0 (all-cpu)':
        ('BrkgaCuda 2.0 (All-CPU)', 'coral', (3, 2, 180), 5),
    'brkga-cuda-2.0 (cpu-permutation)':
        ('BrkgaCuda 2.0 (CPU-permutation)', 'red', (3, 2, 180), 6),
    'brkga-cuda-2.0 (all-cpu-permutation)':
        ('BrkgaCuda 2.0 (All-CPU-permutation)', 'lime', (3, 2, 180), 7),
    'gpu-brkga (gpu)':
        ('GPU-BRKGA (GPU)', 'purple', '.', 8),
    'gpu-brkga-fix (gpu)':
        ('GPU-BRKGA-Fixed (GPU)', 'magenta', '.', 9),
    'brkga-cuda-1.0 (gpu)':
        ('BrkgaCuda 1.0 (All-GPU)', 'cadetblue', '*', 10),
    'brkga-cuda-1.0 (gpu-permutation)':
        ('BrkgaCuda 1.0 (All-GPU-permutation)', 'orange', '*', 11),
    'brkga-cuda-2.0 (gpu)':
        ('BrkgaCuda 2.0 (GPU)', 'brown', (3, 2, 90), 12),
    'brkga-cuda-2.0 (all-gpu)':
        ('BrkgaCuda 2.0 (All-GPU)', 'peru', (3, 2, 90), 13),
    'brkga-cuda-2.0 (gpu-permutation)':
        ('BrkgaCuda 2.0 (GPU-permutation)', 'sienna', (3, 2, 270), 14),
    'brkga-cuda-2.0 (all-gpu-permutation)':
        ('BrkgaCuda 2.0 (All-GPU-permutation)', 'violet', (3, 2, 270), 15),
}

LABELS = {k: p[0] for k, p in PLOTS.items()}
COLORS = {k: p[1] for k, p in PLOTS.items()}
MARKERS = {k: p[2] for k, p in PLOTS.items()}
LABEL_ORDER = {k: p[3] for k, p in PLOTS.items()}


def plot_convergence():
    plot_by = ['problem', 'instance']
    plot_params = [p for p in PARAMS if p not in plot_by + ['seed']]
    label_color = {}
    colors = COLORS.copy()
    for _, plot_condition in data[plot_by].drop_duplicates().iterrows():
        plt.figure(figsize=FIG_SIZE)

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
                assert colors
                label_color[label] = colors.pop()

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

    fig = plt.figure(figsize=FIG_SIZE)
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
    plt.xlabel('Instance')
    plt.ylabel('Fitness')
    plt.grid(axis='x', which='major', linestyle='--')

    fig_file = Path('plots/greedy-vs-optimal-decoder.eps')
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_file, bbox_inches='tight')
    plt.savefig(fig_file.with_suffix('.png'), bbox_inches='tight')
    plt.close(fig)


def time_box_plot():
    # The columns with the results
    ans = 'ans'
    elapsed = 'elapsed'

    groups = data.groupby('problem').groups.items()
    groups = sorted(groups, key=lambda x: x[0][1])  # permutation by problem
    for problem, rows in groups:
        compare_results = data[
            (data['tool'] == COMPARE_TO.split()[0])
            & (data['problem'] == problem)
        ]

        rows = list(set(rows).union(set(compare_results.index)))
        plot = data.iloc[rows].pivot_table(
            values=[ans, elapsed],
            index=['instance', 'seed'],
            columns=['tool', 'decode'],
        )

        plot.columns = pd.MultiIndex.from_tuples(
            [(col[0], f'{col[1]} ({col[2]})') for col in plot.columns])

        # Calculate the average of the solution of the algorithm in COMPARE_TO
        #  and normalize the other solutions.
        tmp = plot[(ans, COMPARE_TO)].reset_index()
        tmp.columns = tmp.columns.droplevel(1)
        tmp = tmp.groupby('instance')[ans].mean()
        norm = plot[(ans, COMPARE_TO)].copy()
        for instance in tmp.index:
            norm.loc[instance] = tmp.loc[instance]
        for algorithm in set(col for _, col in plot.columns):
            plot[(ans, algorithm)] /= norm

        # Calculate the median of time of the algorithm in COMPARE_TO
        #  and normalize the other times.
        tmp = plot[(elapsed, COMPARE_TO)].reset_index()
        tmp.columns = tmp.columns.droplevel(1)
        tmp = tmp.groupby('instance')[elapsed].median()
        norm = plot[(elapsed, COMPARE_TO)].copy()
        for instance in tmp.index:
            norm.loc[instance] = tmp.loc[instance]
        for algorithm in set(col for _, col in plot.columns):
            plot[(elapsed, algorithm)] = norm / plot[(elapsed, algorithm)]

        # Build the figure
        plot.columns = plot.columns.swaplevel(0, 1)
        plot = plot.sort_index(axis=1, level=0)

        fig = plt.figure(figsize=FIG_SIZE)

        algos = sorted(set(col for col, _ in plot.columns), key=LABEL_ORDER.get)
        points = [plot[(algorithm, elapsed)] for algorithm in algos]
        bp = plt.boxplot(points, sym='k.', vert=True, patch_artist=True, whis=1.5)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')

        # Format and save the figure
        labels = [LABELS[a] for a in algos]
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel('Speedup', fontsize=11)
        plt.minorticks_on()
        plt.grid(axis='y', which='minor', linestyle=':')
        plt.grid(axis='y', which='major', linestyle='-')
        plt.grid(axis='x', which='major', linestyle=':')

        fig.autofmt_xdate(rotation=45)
        plt.subplots_adjust(bottom=0.3)
        plt.tight_layout()
        plt.savefig(f'plots/time-box-plot-{problem}.eps')
        plt.savefig(f'plots/time-box-plot-{problem}.png')
        plt.close(fig)


if __name__ == '__main__':
    # plot_convergence()
    # fitness()
    # greedy_vs_optimal_cvrp()
    time_box_plot()
