import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


COMPARE_TO = 'brkga-api (cpu)'

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

FIG_NUMBER = {
    'quality': 3,
    'box-tsp': 4,
    'box-scp': 5,
    'box-cvrp': 6,
    'box-cvrp_greedy': 7,
}


def make_plot(ax: plt.Axes, df: pd.DataFrame, problem: str):
    # The columns with the results
    ans = 'ans'
    elapsed = 'elapsed'

    plot = df.pivot_table(
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
    algos = sorted(set(col for col, _ in plot.columns), key=LABEL_ORDER.get)
    data = [plot[(algorithm, ans)] for algorithm in algos]
    bp = ax.boxplot(data, sym='k.', vert=True, patch_artist=True, whis=1.5)
    for algo, patch in zip(algos, bp['boxes']):
        if 'gpu-brkga ' in algo:
            patch.set_facecolor('red')
        else:
            patch.set_facecolor('lightblue')

    # Format the figure
    ax.set_title(problem.upper().replace('_', '-'))

    labels = [LABELS[a] for a in algos]
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Fitness ratio')

    if problem == 'scp':
        ax.set_ylim([0.65, 1.75])
    else:
        ax.set_ylim([0.89, 1.25])

    # ax.minorticks_on()
    # ax.tick_params(axis='both', labelsize=6)
    ax.grid(zorder=0)
    # ax.grid(which='minor', linestyle=':')


def make_boxplot(df: pd.DataFrame, name: str):
    # The columns with the results
    ans = 'ans'
    elapsed = 'elapsed'

    plot = df.pivot_table(
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

    fig = plt.figure(figsize=(10*2/3*0.8, 8*2/3*0.8))

    algos = sorted(set(col for col, _ in plot.columns), key=LABEL_ORDER.get)
    data = [plot[(algorithm, elapsed)] for algorithm in algos]
    bp = plt.boxplot(data, sym='k.', vert=True, patch_artist=True, whis=1.5)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    # Format and save the figure
    labels = [LABELS[a] for a in algos]
    plt.xticks(range(1, len(labels) + 1), labels, size='x-small', fontsize=11)
    plt.ylabel('Speedup', fontsize=11)
    plt.minorticks_on()
    plt.grid(axis='y', which='minor', linestyle=':')
    plt.grid(axis='y', which='major', linestyle='-')
    plt.grid(axis='x', which='major', linestyle=':')

    fig.autofmt_xdate(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.tight_layout()
    plt.savefig(f'figs/{name}.eps')
    plt.savefig(f'figs/{name}.png')
    plt.close(fig)


def build_fitness_plot(df: pd.DataFrame):
    groups = df.groupby('problem').groups.items()
    groups = sorted(groups, key=lambda x: x[0][1])  # permutation by problem

    fig, axes = plt.subplots(2, 2, figsize=(10*.9, 8*.9))
    axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for idx, group in enumerate(groups):
        problem = group[0]
        compare_results = df[
            (df['tool'] == COMPARE_TO.split()[0])
            & (df['problem'] == problem)
        ]

        rows = list(set(group[1]).union(set(compare_results.index)))
        make_plot(axes[idx], df.iloc[rows], problem)

    fig.tight_layout()
    fig.savefig('figs/solution-quality.eps')
    fig.savefig('figs/solution-quality.png')
    plt.close(fig)


def build_time_plot(df: pd.DataFrame):
    groups = df.groupby('problem').groups.items()
    groups = sorted(groups, key=lambda x: x[0][1])  # permutation by problem

    for group in groups:
        problem = group[0]
        compare_results = df[
            (df['tool'] == COMPARE_TO.split()[0])
            & (df['problem'] == problem)
        ]

        rows = list(set(group[1]).union(set(compare_results.index)))
        make_boxplot(df.iloc[rows], name='box-' + problem)


def main():
    results = pd.read_csv(Path(sys.argv[1]), sep='\t')
    build_fitness_plot(results)
    build_time_plot(results)


if __name__ == '__main__':
    main()
