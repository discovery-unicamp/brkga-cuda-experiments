import datetime
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, Iterable, List, Union
import pandas as pd

from instance import get_instance_path


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

DEVICE = int(os.environ['DEVICE'])
TEST_COUNT = 10
BUILD_TYPE = 'release'
BUILD_TARGET = 'brkga-optimizer'
TWEAKS_FILE_PATH = Path('applications', 'src', 'Tweaks.hpp')
SOURCE_PATH = Path('applications')
OUTPUT_PATH = Path('experiments', 'results')

PROBLEMS = {
    'cvrp': 'cvrp',
    'cvrp_greedy': 'cvrp',
    'scp': 'scp',
    'tsp': 'tsp',
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


def compile_optimizer(target: str, problem: str):
    return __cmake(str(SOURCE_PATH.absolute()), BUILD_TYPE, target,
                   tweaks=[problem.upper()])


def __cmake(
        src: str,
        build: str,
        target: str,
        threads: int = 6,
        tweaks: List[str] = [],
):
    with open(TWEAKS_FILE_PATH, 'w') as tweak_file:
        tweak_file.write('#pragma once\n')
        for tweak in tweaks:
            tweak_file.write(f'#define {tweak}\n')

    folder = f'build-{build}'
    __shell(f'cmake -DCMAKE_BUILD_TYPE={build} -B{folder} {src}', get=False)
    __shell(f'cmake --build {folder} --target {target} -j{threads}', get=False)
    return Path(folder, target)


def __get_system_info() -> Dict[str, str]:
    # Tell to git on docker that this is safe.
    __shell('git config --global --add safe.directory /brkga')
    return {
        'commit': __shell('git log --format="%H" -n 1'),
        'system': __shell('uname -v'),
        'cpu': __shell('cat /proc/cpuinfo | grep "model name"'
                       ' | uniq | cut -d" " -f 3-'),
        'host-memory':
            __shell('grep MemTotal /proc/meminfo | awk \'{print $2 / 1024}\'')
            + 'MiB',
        'gpu': (__shell('lspci | grep " VGA " | cut -d" " -f 5-')
                .split('\n')[DEVICE]
                .strip()),
        'gpu-memory':
            (__shell('lshw -C display | grep product | cut -d":" -f2-')
             .split('\n')[DEVICE]
             .strip()),
        'nvcc': __shell('nvcc --version | grep "release" | grep -o "V.*"'),
        'g++': __shell('g++ --version | grep "g++"'),
    }


def __shell(cmd: str, get: bool = True) -> str:
    logging.debug(f'Execute command `{cmd}`')
    output = ''
    try:
        stdout = subprocess.PIPE if get else None
        process = subprocess.run(
            cmd, stdout=stdout, text=True, shell=True, check=True)
        output = process.stdout.strip() if get else ''
    except subprocess.CalledProcessError as error:
        output = error.stdout.strip() if get else ''
        raise
    finally:
        if output:
            logging.info(f'Script output:\n{output}')

    return output


def __run_test(
        executable: Path,
        params: Dict[str, Union[str, float, int]],
) -> Dict[str, str]:
    logging.info(f'Test instance {params["instance"]}')
    logging.debug(f'Executable: {str(executable)}')
    logging.debug(f'Test with params {params}')

    parsed_params = {key: __parse_param(value)
                     for key, value in params.items()}

    cmd = str(executable.absolute())
    cmd += ''.join(f' --{arg} {value}' for arg, value in parsed_params.items())

    result = dict(tuple(r.split('=')) for r in __shell(cmd).split())
    if 'convergence' not in result or result['convergence'] == '[]':
        result['convergence'] = '?'

    return {
        **parsed_params,
        'ans': result['ans'],
        'elapsed': result['elapsed'],
        'convergence': result.get('convergence', '?'),
    }


def __parse_param(value: Union[int, float, str]) -> str:
    if isinstance(value, float):
        return str(round(value, 6))
    return str(value)


def experiment(
    problems: List[str],
    tools: List[str],
    decoders: List[str],
    test_count: int = TEST_COUNT,
) -> Iterable[Dict[str, str]]:
    if 'tsp' in problems and ('gpu-brkga' in tools
                              or 'gpu-brkga-fixed' in tools):
        logging.warning('Ignoring GPU-BRKGA for TSP')

    for problem in problems:
        pname = PROBLEMS[problem]
        for tool in tools:
            if pname == 'tsp' and (tool == 'gpu-brkga'
                                    or tool == 'gpu-brkga-fixed'):
                continue

            executable = compile_optimizer(tool, problem)
            for decoder in decoders:
                for instance in INSTANCES[pname]:
                    instance_path = str(get_instance_path(pname, instance))
                    for seed in range(1, test_count + 1):
                        params = {
                            'threads': 256,
                            'omp-threads': int(__shell('nproc')),
                            'generations': 1000,
                            'exchange-interval': 50,
                            'exchange-count': 2,
                            'pop-count': 3,
                            'pop-size': 256,
                            'elite': .1,
                            'mutant': .1,
                            'rhoe': .75,
                            'decode': decoder,
                            'instance': instance_path,
                            'seed': seed,
                            'log-step': 25,
                        }

                        result = __run_test(executable, params)
                        result['tool'] = tool
                        result['problem'] = problem
                        result['instance'] = instance
                        yield result


def save_results(info: Dict[str, str], iter_results: Iterable[Dict[str, str]]):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    backup_file = OUTPUT_PATH.joinpath('.backup.tsv')
    results = pd.DataFrame()
    for res in iter_results:
        results = pd.concat((results, pd.DataFrame([{**res, **info}])))
        results.to_csv(backup_file, index=False, sep='\t')

    if results.empty:
        logging.warning('All tests failed')
        return

    test_time = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    results['test_time'] = test_time

    # Define the first columns of the .tsv
    # The others are still written to the file after these ones
    first_columns = ['test_time', 'commit', 'tool',
                     'problem', 'instance', 'ans', 'elapsed', 'seed']
    other_columns = [c for c in results.columns if c not in first_columns]
    results = results[first_columns + other_columns]

    output = OUTPUT_PATH.joinpath(f'{test_time}.tsv')
    results.to_csv(output, index=False, sep='\t')


def main():
    # Execute here to avoid changes by the user.
    info = __get_system_info()

    # save_results(info, experiment(
    #     problems=['tsp', 'cvrp'],
    #     tools=['brkga-cuda-2.0'],
    #     decoders=['gpu-permutation'],
    #     test_count=3,
    # ))
    # save_results(info, experiment(
    #     problems=['scp'],
    #     tools=['brkga-cuda-2.0'],
    #     decoders=['gpu'],
    #     test_count=3,
    # ))
    # exit()

    save_results(info, experiment(
        problems=['cvrp', 'scp', 'tsp'],
        tools=['brkga-api'],
        decoders=['cpu'],
        test_count=10,
    ))
    save_results(info, experiment(
        problems=['scp'],
        tools=['brkga-cuda-2.0'],
        decoders=['cpu', 'all-cpu', 'gpu', 'all-gpu'],
        test_count=10,
    ))
    save_results(info, experiment(
        problems=['cvrp', 'tsp'],
        tools=['brkga-cuda-2.0'],
        decoders=['cpu', 'all-cpu', 'cpu-permutation', 'all-cpu-permutation',
                  'gpu', 'all-gpu', 'gpu-permutation', 'all-gpu-permutation'],
        test_count=10,
    ))
    save_results(info, experiment(
        problems=['cvrp', 'scp'],
        tools=['gpu-brkga'],
        decoders=['cpu', 'gpu'],
        test_count=10,
    ))


if __name__ == '__main__':
    main()
