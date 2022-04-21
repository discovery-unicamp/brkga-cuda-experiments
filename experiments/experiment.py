import datetime
import logging
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Union
import pandas as pd

from instance import get_instance_path


# docker build -t brkga -f experiments/Dockerfile . && docker run -v ~/brkga-cuda/experiments/results/:/main/results/ --rm --gpus all brkga

#             threads exchange-interval exchange-count pop-count pop-size elite   mutant  rhoe
# best time:  256     25                1              3         128      0.10    0.10    0.75
# avg-1 time: 256     25                1              3         128      0.05    0.10    0.75
# avg-2 time: 256     50                1              3         128      0.10    0.15    0.80
# worst time: 256     25                1              3         128      0.05    0.15    0.80
# GPU-BRKGA:  -       -                 -              1         256      0.15625 0.15625 0.70
# GPU-BRKGA:  -       -                 -              1         512      0.15625 0.15625 0.70
# GPU-BRKGA:  -       -                 -              1         1024     0.15625 0.15625 0.70

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)8s]"
           " %(filename)s:%(lineno)s: %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S',
)

failures = []  # FIXME

SOURCE_PATH = Path('applications')
INSTANCES_PATH = Path('instances')
OUTPUT_PATH = Path('results')

EXECUTABLES = {
    'cvrp': Path('brkga-cuda'),
    'tsp': Path('brkga-cuda'),
    'scp': Path('brkga-cuda'),
}

INSTANCES = {
    'cvrp': [
        'X-n219-k73',
        'X-n266-k58',
        'X-n317-k53',
        'X-n336-k84',
        'X-n376-k94',
        'X-n384-k52',
        'X-n420-k130',
        'X-n429-k61',
        'X-n469-k138',
        'X-n480-k70',
        'X-n548-k50',
        'X-n586-k159',
        'X-n599-k92',
        'X-n655-k131',
        # The following doesn't work with the original GPU-BRKGA code
        'X-n733-k159',
        'X-n749-k98',
        'X-n819-k171',
        'X-n837-k142',
        'X-n856-k95',
        'X-n916-k207',
        'X-n957-k87',
        'X-n979-k58',
        'X-n1001-k43',
    ],
    'scp': [
        'scp41',
        'scp42',
        'scp43',
        'scp44',
        'scp45',
        'scp46',
        'scp47',
        'scp48',
        'scp49',
        # Missing instances:
        # 'scp51',
        # 'scp52',
        # 'scp53',
        # 'scp54',
        # 'scp55',
        # 'scp56',
        # 'scp57',
        # 'scp58',
        # 'scp59',
        # 'scp61',
        # 'scp62',
        # 'scp63',
        # 'scp64',
        # 'scp65',
    ],
    'tsp': [
        'zi929',
        'lu980',
        'rw1621',
        'mu1979',
        'nu3496',
        'ca4663',
        'tz6117',
        'eg7146',
        'ym7663',
        'pm8079',
        'ei8246',
        'ar9152',
        'ja9847',
        'gr9882',
        'kz9976',
        'fi10639',
        'mo14185',
        'ho14473',
        'it16862',
        'vm22775',
        'sw24978',
        'bm33708',
    ]
}


def run_experiment(
        problem: str,
        params: Dict[str, Union[str, float, int]],
        instances: List[str],
        test_count: int,
) -> Optional[pd.DataFrame]:
    if test_count == 0:
        raise ValueError('Test count is zero')

    # Take info here to avoid changes by the user
    info = __get_system_info()

    results = []
    for instance in instances:
        logging.info(f'[{problem}] Testing {instance}')
        try:
            tmp = []
            for test in range(test_count):
                seed = test + 1
                tmp.append(__run_test(problem, params, instance, seed))
            results += tmp
        except (KeyboardInterrupt, AssertionError):
            raise
        except:
            logging.exception(f'Failed to run instance {instance} ({problem})')
            failures.append(f'{problem} - {instance}')

    results = pd.DataFrame([{**r, **info} for r in results])
    results['tool'] = params.get('tool', 'default')

    logging.info('Experiment finished')
    return results


def __get_system_info() -> Dict[str, str]:
    with open('info.txt') as f:
        data = f.read()
    return dict(line.split(': ') for line in data.split('\n') if line)


def __shell(cmd: str, get: bool = True) -> str:
    logging.debug(f'Execute command `{cmd}`')
    try:
        stdout = subprocess.PIPE if get else None
        process = subprocess.run(
            cmd, stdout=stdout, text=True, shell=True, check=True)
    except subprocess.CalledProcessError as error:
        output = error.stdout.strip() if get else ''
        if output:
            logging.info(f'Script output before error:\n'
                         f'--- begin ---\n{output}\n==== end ====')
        raise

    if not get:
        return ''

    output = process.stdout.strip()
    logging.debug(f'Script output:\n--- begin ---\n{output}\n==== end ====')
    return output


def __run_test(
        problem: str,
        params: Dict[str, Union[str, float, int]],
        instance: str,
        seed: int,
) -> Dict[str, str]:
    executable = EXECUTABLES[problem]

    logging.info(f'Test instance {instance} of {problem} ({str(executable)})')
    logging.info(f'Test with seed {seed} and params {params}')

    instance_path = get_instance_path(problem, instance)

    parsed_params = {key: __parse_param(value)
                     for key, value in params.items()}
    parsed_params['instance'] = str(instance_path.absolute())
    parsed_params['seed'] = str(seed)

    cmd = str(executable.absolute())
    cmd += ''.join(f' --{arg} {value}' for arg, value in parsed_params.items())

    result = dict(tuple(r.split('=')) for r in __shell(cmd).split())
    if 'convergence' not in result or result['convergence'] == '[]':
        result['convergence'] = '?'

    return {
        **parsed_params,
        'seed': str(seed),
        'instance': instance,
        'ans': result['ans'],
        'elapsed': result['elapsed'],
        'convergence': result.get('convergence', '?'),
    }


def __parse_param(value: Union[int, float, str]) -> str:
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def test_all():
    results = []
    for problem in ['scp', 'cvrp', 'tsp']:
        for tool in ['brkga-cuda', 'gpu-brkga', 'brkga-api']:
            if tool == 'gpu-brkga' and problem == 'tsp':
                logging.warning('GPU-BRKGA doesn\'t support the TSP instance')
                continue

            params = {
                'threads': 256,
                'generations': 1000,
                'exchange-interval': 50,
                'exchange-count': 2,
                'pop-count': 3,
                'pop-size': 256,
                'elite': .1,
                'mutant': .1,
                'rhoe': .75,
                'decode': 'host',
                'tool': tool,
                'problem': problem,
                'log-step': 25,
            }

            results.append(run_experiment(
                problem, params, INSTANCES[problem], test_count=10))

    if not results:
        logging.warning('All tests failed')
        return

    test_time = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    results = pd.concat(results)
    results['test_time'] = test_time

    # Define the first columns of the .tsv
    # The others are still written to the file after these ones
    first_columns = ['test_time', 'commit', 'tool',
                     'problem', 'instance', 'ans', 'elapsed', 'seed']
    other_columns = [c for c in results.columns if c not in first_columns]
    results = results[first_columns + other_columns]

    output = OUTPUT_PATH.joinpath('all')
    output.mkdir(parents=True, exist_ok=True)
    output = output.joinpath(f'{test_time}.tsv')
    results.to_csv(output, index=False, sep='\t')

    if failures:
        format_failures = "".join("\n - " + f for f in failures)
        logging.error(f'Failures:{format_failures}')


if __name__ == '__main__':
    test_all()
