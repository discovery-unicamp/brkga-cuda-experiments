from bisect import bisect_right
from datetime import datetime
import logging
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

import pandas as pd

from result import read_results


FILES = [
    # 'remove-duplicated-dl3.zip',
    # 'remove-duplicated-dl4.zip',
    # 'brkga-mp-ipr.zip',
    # 'pr-test-v1.zip',
    '2022-10-18T11:23:30.zip',
    # '2022-11-02T15:51:17.zip',
    '.backup2.tsv.zip',
]
DATA = {file: read_results(Path(f'results/{file}')) for file in FILES}

TOOL_COLOR = {
    'brkga-cuda-2.0': 'green',
    'brkga-mp-ipr': 'red',
}

PARAMS = [
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
    'prune-interval',
    'prune-threshold',
]

logging.info("Converting convergence to lists...")
for file in FILES:
    DATA[file].loc[:, 'convergence'] = DATA[file]['convergence'].apply(eval)
    DATA[file] = DATA[file].loc[DATA[file]['start_time'] > '2022-10-16T03:09:51']
    for p in PARAMS:
        if p not in DATA[file].columns:
            DATA[file].loc[:, p] = None

logging.info("Done.")

FITNESS_ELAPSED = [1, 2, 3, 5, 15, 30, 60, 120, 180]
FITNESS_VAR = [
    'rhoe',
    'elite',
    'similarity-threshold',
]
PLOT_GENERATIONS = False


def convergence():
    df = DATA[FILES[0]]
    instances = df[['problem', 'instance']].drop_duplicates()
    # instances = instances[instances['instance'].isin(['X-n219-k73'])]

    for _, inst in instances.iterrows():
        plt.figure(figsize=(10.80, 7.20))
        problem = inst['problem']
        instance = inst['instance']

        for input_file in FILES:
            df = DATA[input_file]
            df = df.loc[(df['instance'] == instance)]
            df = df.loc[(df['problem'] == problem)]
            # df = df.loc[(df['similarity-threshold'] == .9)]
            # df = df.loc[(df['elite'] == .1)]

            for _, row in df.iterrows():
                fitness, elapsed, generation = zip(*row['convergence'])
                fitness = [y
                           for i, x in enumerate(fitness)
                           for y in (1 if i == len(fitness) - 1 else 2) * [x]]
                elapsed = [y
                           for i, x in enumerate(elapsed)
                           for y in (1 if i == 0 else 2) * [x]]
                generation = [y
                              for i, x in enumerate(generation)
                              for y in (1 if i == 0 else 2) * [x]]

                if elapsed[-1] < row['max-time'] and not PLOT_GENERATIONS:
                    elapsed.append(row['max-time'])
                    fitness.append(fitness[-1])

                plt.plot(
                    (generation if PLOT_GENERATIONS else elapsed),
                    fitness,
                    label='_'.join(map(str, row[PARAMS])),
                    color=TOOL_COLOR[row['tool']],
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


def fitness():
    def bb(conv: List[Tuple[float, float, float]], elapsed: float):
        assert len(conv) > 0
        k = bisect_right(conv, elapsed, key=lambda c: c[1])
        assert k > 0
        assert conv[k - 1][1] <= elapsed
        assert k == len(conv) or conv[k][1] > elapsed
        return conv[k - 1][0]

    instances = DATA[FILES[0]][['problem', 'instance']].drop_duplicates()
    for _, ins in instances.iterrows():
        instance = ins['instance']
        problem = ins['problem']
        for var in FITNESS_VAR:
            logging.info("Creating figure for %s:%s with var %s",
                         problem, instance, var)
            df = pd.concat((
                df[(df['instance'] == instance) & (df['problem'] == problem)]
                for df in DATA.values()
            ))

            fig, ax = plt.subplots(3, 3, figsize=(2*10.80, 2*7.20))
            axes = [ax[i, j] for i in range(3) for j in range(3)]
            for k, elapsed in enumerate(FITNESS_ELAPSED):
                df['y'] = df['convergence'].map(lambda conv: bb(conv, elapsed))
                df.boxplot('y', by=var, ax=axes[k])
                axes[k].set_title(f"elapsed={elapsed}s")
                axes[k].grid()

            name = f"{problem}-{instance}-{var}"
            outfile = Path('fitness', name + ".png")
            outfile.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(outfile.absolute()))
            plt.close(fig)


if __name__ == '__main__':
    convergence()
    # fitness()
