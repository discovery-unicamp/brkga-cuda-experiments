from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


TOOLS = ['brkga-cuda-2.0', 'brkga-mp-ipr']
TUNING_PATH = Path('tuning')


def main():
    result_df = pd.concat(__read_output(), ignore_index=True)
    __plot_tuning_convergence(result_df)


def __read_output():
    instance_fix = {
        'i1': 'X-n219-k73',
        'i2': 'X-n599-k92',
        'i3': 'X-n1001-k43',
    }

    for tool in TOOLS:
        problem = 'cvrp'
        decoder = 'cpu'

        results = []
        for file in __output_files(tool, problem, decoder):
            lines = [line for line in file.read_text().split('\n') if line]
            if len(lines) == 1:
                instance_re = re.search(r"-(i.*?)-", file.name)
                output_line = 0
            elif len(lines) == 2:
                instance_re = re.search(r"--instance [\w/-]*/(.*?)\.[a-z]+ ",
                                        lines[0])
                output_line = 1
            else:
                raise RuntimeError(
                    f"Invalid number of lines ({len(lines)}) on {file}")

            assert instance_re is not None
            instance = instance_re.group(1)
            instance = instance_fix.get(instance, instance)

            output = dict(tuple(r.split('='))
                          for r in lines[output_line].split(' '))

            results.append((
                instance,
                float(output['ans']),
                float(output['elapsed']),
            ))

        result_df = pd.DataFrame(results,
                                 columns=['instance', 'ans', 'elapsed'])
        result_df['tool'] = tool
        result_df['problem'] = problem
        result_df['decoder'] = decoder
        result_df = result_df.astype({
            'tool': 'category',
            'problem': 'category',
            'decoder': 'category',
            'instance': 'category',
        })
        yield result_df


def __output_files(tool: str, problem: str, decoder: str):
    logs_path = TUNING_PATH.joinpath(f"{tool}_{problem}_{decoder}", 'logs')
    return logs_path.glob('*.stdout')


def __plot_tuning_convergence(result_df: pd.DataFrame):
    def plot(df: pd.DataFrame):
        df = df.sort_values(by=['ans'], ascending=False)

        problem = df.iloc[0]['problem']
        instance = df.iloc[0]['instance']
        label = f"{problem}_{instance}"

        plt.figure(figsize=(10.80, 7.20))
        for tool, decoder in (df[['tool', 'decoder']]
                              .drop_duplicates()
                              .itertuples(index=False, name=None)):
            df2 = df[(df['tool'] == tool) & (df['decoder'] == decoder)]
            x = list(range(len(df2)))
            y = list(df2['ans'])
            plt.plot(x, y, label=f"{tool}_{decoder}")

        plt.title("Tuning convergence")
        plt.xlabel("Configuration")
        plt.ylabel("Fitness")
        legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.minorticks_on()
        plt.grid(axis='y', which='minor', linestyle=':')
        plt.grid(axis='y', which='major', linestyle='-')
        plt.grid(axis='x', which='major', linestyle=':')

        outfile = Path('tuning-convergence', f"{label}.png")
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, bbox_extra_artists=(legend,), bbox_inches='tight')
        plt.close()

    result_df.groupby(['problem', 'instance']).apply(plot)


if __name__ == '__main__':
    main()
