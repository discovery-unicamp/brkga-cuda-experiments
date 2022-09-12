from bisect import bisect_right
import logging
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt

import pandas as pd

from result import read_results


files = [
    'remove-duplicated-dl3.zip',
    'remove-duplicated-dl4.zip',
]

PARAMS = [
    'decoder',
    'threads',
    'generations',
    'seed',
    'pop-count',
    'pop-size',
    'rhoe',
    'elite',
    'mutant',
    'exchange-interval',
    'exchange-count',
    'similarity-threshold',
]

FITNESS_ELAPSED = [1, 2, 3, 5, 15, 30, 60, 120, 180]
FITNESS_VAR = [
    'rhoe',
    'elite',
    'similarity-threshold',
]


def convergence():
    df = read_results(Path(f'results/{files[0]}'))
    instances = df[['problem', 'instance']].drop_duplicates()

    for _, instance in instances.iterrows():
        plt.figure(figsize=(10.80, 7.20))

        for infile in files:
            df = read_results(Path(f'results/{infile}'))
            df = df.loc[(df['instance'] == instance['instance'])]
            df = df.loc[(df['problem'] == instance['problem'])]
            df.loc[:, 'convergence'] = df['convergence'].apply(eval)

            for _, row in df.iterrows():
                fitness, elapsed, generation = zip(*row['convergence'])
                plt.plot(
                    elapsed,
                    fitness,
                    label='_'.join(map(str, row[PARAMS])),
                )

        name = f"{instance['problem']}-{instance['instance']}"

        if instance['problem'] == 'scp':
            plt.ylim((500, 1500))

        plt.title(f"{name} convergence")
        plt.xlabel("Time (s)")
        plt.ylabel("Fitness / BKS")
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.minorticks_on()
        plt.grid(axis='y', which='minor', linestyle=':')
        plt.grid(axis='y', which='major', linestyle='-')
        plt.grid(axis='x', which='major', linestyle=':')

        outfile = Path('convergence', name + ".png")
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, bbox_extra_artists=(legend,), bbox_inches='tight')


def fitness():
    def bb(conv: List[Tuple[float, float, float]], elapsed: float):
        assert len(conv) > 0
        k = bisect_right(conv, elapsed, key=lambda c: c[1])
        assert conv[k - 1][1] <= elapsed
        assert k == len(conv) or conv[k][1] > elapsed
        return conv[k - 1][0]

    data = {file: read_results(Path(f'results/{file}')) for file in files}

    for file in files:
        data[file].loc[:, 'convergence'] = data[file]['convergence'].apply(eval)

    instances = data[files[0]][['problem', 'instance']].drop_duplicates()
    for _, ins in instances.iterrows():
        instance = ins['instance']
        problem = ins['problem']
        for var in FITNESS_VAR:
            logging.info("Creating figure for %s:%s with var %s",
                         problem, instance, var)
            df = pd.concat((
                df[(df['instance'] == instance) & (df['problem'] == problem)]
                for df in data.values()
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
            fig.savefig(outfile)
            plt.close(fig)


if __name__ == '__main__':
    # convergence()
    fitness()
