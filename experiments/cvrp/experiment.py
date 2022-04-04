import datetime
from enum import Enum
import logging
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Union
import pandas as pd

from instance import get_instance_path


#             threads exchange-interval exchange-count pop-count pop-size elite   mutant  rho
# best time:  256     25                1              3         128      0.10    0.10    0.75
# avg-1 time: 256     25                1              3         128      0.05    0.10    0.75
# avg-2 time: 256     50                1              3         128      0.10    0.15    0.80
# worst time: 256     25                1              3         128      0.05    0.15    0.80
# GPU-BRKGA:  -       -                 -              1         256      0.15625 0.15625 0.70
# GPU-BRKGA:  -       -                 -              1         512      0.15625 0.15625 0.70
# GPU-BRKGA:  -       -                 -              1         1024     0.15625 0.15625 0.70

SOURCE_PATH = Path('applications')
INSTANCES_PATH = Path('instances')
OUTPUT_PATH = Path('experiments', 'results')

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

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)8s]"
           " %(filename)s:%(lineno)s: %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S',
)


class ToolName(Enum):
    YELMEWAD2021_CUSTOMER = 'yelmewad2021', 'yelmewad2021-customer'

    def __new__(cls, *args, **kw):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, sources: str, target: str):
        self.sources = sources
        self.target = target


def run_tool(tool: ToolName, instances: List[str]):
    build_folder = f'build-{tool.sources}'
    load = f'cmake -DCMAKE_BUILD_TYPE=release -B{build_folder} {tool.sources}'
    build = f'cmake --build {build_folder}'
    __shell(load, get=False)
    __shell(build, get=False)

    data = []
    # Only instances X of CVRP
    for instance in instances:
        instance_path = get_instance_path('cvrp', instance)
        cmd = f'{str(Path(build_folder, tool.target).absolute())} {str(instance_path)}'
        results = __shell(cmd).split()

        fitness = float(results[3])
        elapsed = float(results[6])
        data.append((instance, elapsed, fitness))
    return pd.DataFrame(data, columns=['instance', 'elapsed', 'fitness'])


def run_experiment(
        problem: str,
        params: Dict[str, Union[str, float, int]],
        instances: List[str],
        test_count: int,
        mode: str = 'release',
) -> Optional[pd.DataFrame]:
    default_columns = ['test_date', 'tool', 'instance',
                       'ans', 'elapsed', 'seed']
    for p in default_columns:
        if p in params and p != 'tool':
            raise ValueError(f'Parameter cannot be named `{p}`')

    executable = __compile(problem, mode)
    if test_count == 0:
        logging.warning('Test count is zero; ignoring the execution')
        return

    # Take info here to avoid changes by the user
    test_date = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    info = __get_system_info()

    results = []
    for instance in instances:
        logging.info(f'[{problem}] Testing {instance}')
        for test in range(test_count):
            seed = test + 1
            ans = __run_test(problem, executable, params, instance, seed)
            results.append(ans)

    results = [{**r, **info} for r in results]
    results = pd.DataFrame(results)
    results['test_date'] = test_date
    results['tool'] = params.get('tool', 'default')

    columns = [c for c in results.columns if c not in default_columns]
    results = results[default_columns + columns]

    logging.info('Experiment finished')
    return results


def __compile(problem: str, mode: str) -> Path:
    logging.info(f'Compiling {problem} with {mode} mode')
    mode = mode.lower()
    folder = f'build-{mode}'
    target = f'brkga-cvrp'

    load = f'cmake -DCMAKE_BUILD_TYPE={mode} -B{folder} {str(SOURCE_PATH)}'
    build = f'cmake --build {folder} --target {target}'
    __shell(load, get=False)
    __shell(build, get=False)

    return Path(folder, 'cvrp', target)


def __get_system_info() -> Dict[str, str]:
    return {
        'commit': __shell('git log --format="%H" -n 1'),
        'system': __shell('uname -v'),
        'cpu': __shell('cat /proc/cpuinfo | grep "model name"'
                       ' | uniq | cut -d" " -f 3-'),
        'cpu-cores': __shell('nproc'),
        'host-memory': __shell('grep MemTotal /proc/meminfo'
                               " | awk '{print $2 / 1024}'") + 'MiB',
        'gpu': __shell('lspci | grep " VGA " | cut -d" " -f 5-'),
        'gpu-cores': 'unknown',
        'gpu-memory': __shell('nvidia-smi -q | grep -m1 Total'
                              " | awk '{print $3}'") + 'MiB',
        'nvcc': __shell('nvcc --version | grep "release" | grep -o "V.*"'),
        'g++': __shell('g++ --version | grep "g++"'),
    }


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
        executable: Path,
        params: Dict[str, Union[str, float, int]],
        instance: str,
        seed: int,
) -> Dict[str, str]:
    logging.info(f'Test instance {instance} of {problem}'
                 f' ({str(executable)}) with params {params} and seed {seed}')
    instance_path = get_instance_path(problem, instance)

    cmd = str(executable.absolute())
    cmd += ''.join(f' --{arg} {value}' for arg, value in params.items())
    cmd += f' --instance {str(instance_path.absolute())} --seed {seed}'

    result = dict(tuple(r.split('=')) for r in __shell(cmd).split())

    str_params = {key: str(value) for key, value in params.items()}
    if 'convergence' in result and result['convergence'] == '[]':
        result['convergence'] = '?'

    return {
        **str_params,
        'seed': str(seed),
        'instance': instance,
        'ans': result['ans'],
        'elapsed': result['elapsed'],
        'convergence': result.get('convergence', '?'),
    }


def main():
    problem = 'tsp'

    # compile only
    # run_experiment(problem, {}, [], test_count=0)

    # results = run_tool(ToolName.YELMEWAD2021_CUSTOMER, instances)
    # output = OUTPUT_PATH.joinpath(problem)
    # output.mkdir(parents=True, exist_ok=True)
    # results.to_csv(output.joinpath(f'yielmewad2021.tsv'), index=False, sep='\t')

    for tool in ['brkga-cuda', 'gpu-brkga']:
        mode = 'release'
        params = {
            'threads': 256,
            'generations': 20000,
            'exchange-interval': 50,
            'exchange-count': 1,
            'pop-count': 3,
            'pop-size': 128,
            'elite': .1,
            'mutant': .1,
            'rho': .75,
            'decode': 'device-sorted',
            'tool': tool,
            'problem': problem,
            'log-step': 50,
        }
        if mode == 'debug':
            params['generations'] = 10
            params['exchange-interval'] = 2

        results = run_experiment(problem, params, INSTANCES[problem],
                                 test_count=5)
        if results is not None and not results.empty:
            output = OUTPUT_PATH.joinpath(problem)
            output.mkdir(parents=True, exist_ok=True)
            test_date = results.iloc[0]['test_date']
            results.to_csv(output.joinpath(f'{test_date}.tsv'),
                           index=False, sep='\t')


if __name__ == '__main__':
    main()
