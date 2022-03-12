import datetime
import logging
from pathlib import Path
import subprocess
from typing import Dict, List, Optional, Union
import pandas as pd


SOURCE_PATH = Path('applications')
INSTANCES_PATH = Path('instances')
OUTPUT_PATH = Path('experiments', 'results')

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)8s]"
           " %(filename)s:%(lineno)s: %(message)s",
    datefmt='%Y-%m-%dT%H:%M:%S',
)


def run_experiment(
        application: str,
        params: Dict[str, Union[str, float, int]],
        instances: List[str],
        test_count: int,
        mode: str = 'release',
        ) -> Optional[pd.DataFrame]:
    default_columns = ['test_date', 'tool', 'problem', 'instance',
                       'ans', 'elapsed', 'seed']
    for p in default_columns:
        if p in params and p != 'tool':
            raise ValueError(f'Parameter cannot be named `{p}`')

    executable = __compile(application, mode)
    if test_count == 0:
        logging.warning('Test count is zero; ignoring the execution')
        return

    # Take info here to avoid changes by the user
    test_date = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    info = __get_system_info()

    results = []
    for instance in instances:
        logging.info(f'[{application}] Testing {instance}')
        for test in range(test_count):
            seed = test + 1
            ans = __run_test(application, executable, params, instance, seed)
            results.append(ans)

    results = [{**r, **info} for r in results]
    results = pd.DataFrame(results)
    results['test_date'] = test_date
    results['tool'] = params.get('tool', 'default')
    results['problem'] = application

    columns = [c for c in results.columns if c not in default_columns]
    results = results[default_columns + columns]
    return results


def __compile(application: str, mode: str) -> Path:
    mode = mode.lower()
    folder = f'build-{mode}'
    target = f'brkga-{application}'

    load = f'cmake -DCMAKE_BUILD_TYPE={mode} -B{folder} {str(SOURCE_PATH)}'
    build = f'cmake --build {folder} --target {target}'
    __shell(load, get=False)
    __shell(build, get=False)

    return Path(folder, application, target)


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
        'gpu-memory': __shell("nvidia-smi -q | grep -m1 Total | awk '{print $3}'")
                + 'MiB',
        'nvcc': __shell('nvcc --version | grep "release"'),
        'g++': __shell('g++ --version | grep "g++"'),
    }


def __shell(cmd: str, get: bool = True) -> str:
    logging.debug(f'Execute command `{cmd}`')
    try:
        stdout = subprocess.PIPE if get else None
        process = subprocess.run(
                cmd, stdout=stdout, text=True, shell=True, check=True)
    except subprocess.CalledProcessError as error:
        if get:
            logging.info(f'Script failed: stdout:\n{error.stdout}')
        raise

    if not get:
        return ''

    output = process.stdout.strip()
    logging.debug(f'Script output:\n--- begin ---\n{output}\n==== end ====')
    return output


def __run_test(
        application: str,
        executable: Path,
        params: Dict[str, Union[str, float, int]],
        instance: str,
        seed: int,
        ) -> Dict[str, str]:
    instance_path = __get_instance_path(application, instance)

    cmd = str(executable.absolute())
    cmd += ''.join(f' --{arg} {value}' for arg, value in params.items())
    cmd += f' --instance {str(instance_path.absolute())} --seed {seed}'

    result = dict(tuple(r.split('=')) for r in __shell(cmd).split())

    str_params = {key: str(value) for key, value in params.items()}
    return {
        **str_params,
        'seed': str(seed),
        'instance': instance,
        'ans': result['ans'],
        'elapsed': result['elapsed'],
        'convergence': result.get('convergence', '?'),
    }


def __get_instance_path(application: str, instance: str):
    group = None
    ext = None
    if application == 'cvrp':
        ext = 'vrp'
        if instance[:2] == 'X-':
            group = 'set-x'

    if ext is None:
        raise ValueError(f'Unknown application `{instance}`')
    if group is None:
        raise ValueError(f'Unknown instance set `{instance}`')
    return INSTANCES_PATH.joinpath(application, group, f'{instance}.{ext}')


def main():
    mode = 'release'
    params = {
        'generations': 1000,
        'exchange-interval': 50,
        'exchange-count': 2,
        'pop_count': 3,
        'pop_size': 256,
        'elite': .1,
        'mutant': .1,
        'rho': .7,
        'decode': 'host-sorted',
        'tool': 'brkga-cuda',
    }
    if mode == 'debug':
        params['generations'] = 10
        params['exchange-interval'] = 2

    instances = ['X-n1001-k43']
    results = run_experiment('cvrp', params, instances, test_count=10, mode=mode)
    if results is not None and not results.empty:
        output = OUTPUT_PATH.joinpath('cvrp')
        output.mkdir(parents=True, exist_ok=True)
        test_date = results.iloc[0]['test_date']
        results.to_csv(output.joinpath(f'{test_date}.tsv'),
                    index=False, sep='\t')


if __name__ == '__main__':
    main()
