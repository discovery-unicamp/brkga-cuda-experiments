import datetime
import itertools
import logging
import os
from pathlib import Path
import traceback
from typing import Any, Dict, Iterable, List, Optional, Union
import pandas as pd

from result import save_results
from instance import get_instance_path
from shell import CATCH_FAILURES, shell


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
RESUME_FROM_BACKUP = False
TEST_COUNT = 1
BUILD_TYPE = 'release'
TWEAKS_FILE_PATH = Path('applications', 'src', 'Tweaks.hpp')
SOURCE_PATH = Path('applications')
OUTPUT_PATH = Path('experiments', 'results')

GENE_TYPE = {
    'brkga-api': 'double',
    'brkga-cuda-1.0': 'float',
    'brkga-cuda-2.0': 'float',
    'brkga-mp-ipr': 'double',
    'gpu-brkga': 'float',
    'gpu-brkga-fix': 'float',
}

PROBLEM_NAME = {
    'cvrp': 'cvrp',
    'cvrp_greedy': 'cvrp',
    'scp': 'scp',
    'tsp': 'tsp',
}
INSTANCES = {
    'cvrp': [
        'X-n219-k73',
        # 'X-n266-k58',
        # 'X-n317-k53',
        # 'X-n336-k84',
        # 'X-n376-k94',
        # 'X-n384-k52',
        # 'X-n420-k130',
        # 'X-n429-k61',
        # 'X-n469-k138',
        # 'X-n480-k70',
        # 'X-n548-k50',
        # 'X-n586-k159',
        'X-n599-k92',
        # 'X-n655-k131',
        # # The following doesn't work with the original GPU-BRKGA code
        # 'X-n733-k159',
        # 'X-n749-k98',
        # 'X-n819-k171',
        # 'X-n837-k142',
        # 'X-n856-k95',
        # 'X-n916-k207',
        # 'X-n957-k87',
        # 'X-n979-k58',
        'X-n1001-k43',
    ],
    'scp': [
        'scp41',
        # 'scp42',
        # 'scp43',
        # 'scp44',
        'scp45',
        # 'scp46',
        # 'scp47',
        # 'scp48',
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
        # 'zi929',
        'lu980',
        # 'rw1621',
        # 'mu1979',
        # 'nu3496',
        # 'ca4663',
        # 'tz6117',
        # 'eg7146',
        # 'ym7663',
        # 'pm8079',
        # 'ei8246',
        # 'ar9152',
        # 'ja9847',
        # 'gr9882',
        # 'kz9976',
        'fi10639',
        # 'mo14185',
        # 'ho14473',
        # 'it16862',
        # 'vm22775',
        # 'sw24978',
        'bm33708',
    ]
}


def main():
    # Execute here to avoid changes by the user.
    info = __get_system_info()

    results = __experiment(itertools.chain(
        # __build_params(
        #     tool='brkga-api',
        #     problems=['scp', 'tsp', 'cvrp_greedy', 'cvrp'],
        #     decoders=['cpu'],
        #     test_count=TEST_COUNT,
        # ),
        # __build_params(
        #     tool='gpu-brkga',
        #     problems=['scp', 'cvrp_greedy', 'cvrp'],
        #     decoders=['cpu', 'gpu'],
        #     test_count=TEST_COUNT,
        # ),
        # __build_params(
        #     tool='gpu-brkga-fix',
        #     problems=['scp', 'cvrp_greedy', 'cvrp'],
        #     decoders=['cpu', 'gpu'],
        #     test_count=TEST_COUNT,
        # ),
        # __build_params(
        #     tool='brkga-cuda-1.0',
        #     problems=['tsp', 'cvrp_greedy', 'cvrp'],
        #     decoders=['cpu', 'gpu', 'gpu-permutation'],
        #     test_count=TEST_COUNT,
        # ),
        # __build_params(
        #     tool='brkga-cuda-1.0',
        #     problems=['scp'],
        #     decoders=['cpu', 'gpu'],
        #     test_count=TEST_COUNT,
        # ),
        # __build_params(
        #     tool='brkga-cuda-2.0',
        #     problems=['scp'],
        #     decoders=['cpu', 'all-cpu', 'gpu', 'all-gpu'],
        #     test_count=TEST_COUNT,
        # ),
        # __build_params(
        #     tool='brkga-cuda-2.0',
        #     problems=['tsp', 'cvrp', 'cvrp_greedy'],
        #     decoders=[
        #         'cpu', 'all-cpu', 'cpu-permutation', 'all-cpu-permutation',
        #         'gpu', 'all-gpu', 'gpu-permutation', 'all-gpu-permutation',
        #     ],
        #     test_count=TEST_COUNT,
        # ),
        __build_params(
            tool='brkga-mp-ipr',
            problems=['tsp', 'cvrp', 'scp'],
            decoders=['cpu'],
            test_count=TEST_COUNT,
        ),
        __build_params(
            tool='brkga-cuda-2.0',
            problems=['tsp', 'cvrp', 'scp'],
            decoders=['cpu', 'gpu'],
            test_count=TEST_COUNT,
        ),
        __build_params(
            tool='brkga-cuda-2.0',
            problems=['tsp', 'cvrp'],
            decoders=['cpu-permutation', 'gpu-permutation'],
            test_count=TEST_COUNT,
        ),
    ))

    __save_results(info, results)


def __get_system_info() -> Dict[str, str]:
    # Tell to git on docker that this is safe.
    shell('git config --global --add safe.directory /brkga')
    return {
        'commit': shell('git log --format="%H" -n 1'),
        'system': shell('uname -v'),
        'cpu': shell('cat /proc/cpuinfo | grep "model name"'
                     ' | uniq | cut -d" " -f 3-'),
        'cpu-memory':
            shell("grep MemTotal /proc/meminfo | awk '{print $2 / 1024}'"),
        'gpu':
            shell("nvidia-smi -L | grep -oP 'NVIDIA.*(?= \(UUID)'")
            .split('\n')[DEVICE],
        'gpu-memory': shell(f"nvidia-smi -i {DEVICE}"
                            " | grep -m1 -oP '[0-9]*(?=MiB)'"
                            " | tail -n1"),
        'nvcc': shell('nvcc --version | grep "release" | grep -o "V.*"'),
        'g++': shell('g++ --version | grep "g++"'),
    }


def __build_params(
        tool: str,
        problems: List[str],
        decoders: List[str],
        test_count: int,
) -> Iterable[Dict[str, Union[str, int, float]]]:
    param_combinations = {
        'tool': tool,
        'problem': problems,
        'decoder': decoders,
        'seed': range(1, test_count + 1),
        'omp-threads': int(shell('nproc')),
        'threads': 256,
        'generations': 200,
        'pop-count': 3,
        'pop-size': 256,
        'rhoe': .75,
        'elite': .10,
        'mutant': .10,
        'exchange-interval': 50,
        'exchange-count': 2,
        'similarity-threshold': .90,
        'log-step': 1,
    }
    for params in __combinations(param_combinations):
        params['problem-name'] = PROBLEM_NAME[params['problem']]
        params['instance-name'] = INSTANCES[params['problem-name']]
        yield from __combinations(params)


def __combinations(of: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    def get(value):
        if isinstance(value, str):
            return [value]
        try:  # Try to return an iterator
            return (x for x in value)
        except TypeError:  # Not iterable
            return [value]

    dict_of_lists: Dict[str, Any] = {k: get(v) for k, v in of.items()}
    keys, values = zip(*dict_of_lists.items())
    return (dict(zip(keys, v)) for v in itertools.product(*values))


def __experiment(
        parameters: Iterable[Dict[str, Any]],
) -> Iterable[Dict[str, str]]:
    UNUSED_PARAMS = {'tool', 'problem', 'problem-name', 'instance-name'}
    for params in parameters:
        executable = __compile_optimizer(params['tool'], params['problem'])
        params['instance'] = get_instance_path(params['problem-name'],
                                               params['instance-name'])

        test_params = {
            name: param
            for name, param in params.items() if name not in UNUSED_PARAMS
        }

        logging.info("Test %s on problem %s (%s) with instance %s",
                     params['tool'], params['problem-name'].upper(),
                     params['problem'], params['instance-name'])
        logging.debug("Parameters:\n%s",
                      '\n'.join(f"\t- {name} = {value}"
                                for name, value in test_params.items()))

        result = __run_test(executable, test_params)
        if result is not None:
            result['tool'] = params['tool']
            result['problem'] = params['problem']
            result['instance'] = params['instance-name']
            yield result


def __compile_optimizer(target: str, problem: str) -> Path:
    return __cmake(str(SOURCE_PATH.absolute()), BUILD_TYPE, target,
                   tweaks=[problem.upper(), f"Gene {GENE_TYPE[target]}"])


def __cmake(
        src: str,
        build: str,
        target: str,
        threads: int = 6,
        tweaks: List[str] = [],
) -> Path:
    tweaks_content = (
        "#pragma once\n"
        + ''.join(f"#define {tweak}\n" for tweak in tweaks)
    )
    try:
        with open(TWEAKS_FILE_PATH, 'r') as tweak_file:
            existing_tweaks_content = tweak_file.read()
    except FileNotFoundError:
        existing_tweaks_content = ''

    if tweaks_content == existing_tweaks_content:
        logging.info("Tweaks file hasn't changed")
    else:
        with open(TWEAKS_FILE_PATH, 'w') as tweak_file:
            tweak_file.write(tweaks_content)

    folder = f'build-{build}'
    shell(f'cmake -DCMAKE_BUILD_TYPE={build} -B{folder} {src}', get=False)
    shell(f'cmake --build {folder} --target {target} -j{threads}', get=False)
    return Path(folder, target)


def __run_test(
        executable: Path,
        params: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    parsed_params = {
        key: str(round(value, 6) if isinstance(value, float) else value)
        for key, value in params.items()
    }

    cmd = str(executable.absolute())
    cmd += ''.join(f' --{arg} {value}' for arg, value in parsed_params.items())

    try:
        result = dict(tuple(r.split('=')) for r in shell(cmd).split())
        if 'convergence' not in result or result['convergence'] == '[]':
            result['convergence'] = '?'

        return {
            **parsed_params,
            'ans': result['ans'],
            'elapsed': result['elapsed'],
            'convergence': result.get('convergence', '?'),
        }
    except Exception:
        if not CATCH_FAILURES:
            raise

        logging.warning("Test failed")
        with open('errors.txt', 'a') as errors:
            errors.write(traceback.format_exc() + '\n')
            errors.write(f'=======\n\n')

        return None


def __save_results(info: Dict[str, str], iter_results: Iterable[Dict[str, str]]):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    backup_file = OUTPUT_PATH.joinpath('.backup.tsv')
    if RESUME_FROM_BACKUP and backup_file.is_file():
        logging.warning("Using results from past backup file")
        results = pd.read_csv(backup_file, sep='\t')
    else:
        results = pd.DataFrame()
    for res in iter_results:
        results = pd.concat((results, pd.DataFrame([{**res, **info}])))
        results.to_csv(backup_file, index=False, sep='\t')

    if results.empty:
        logging.warning("All tests failed")
        return

    test_time = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
    results['test_time'] = test_time

    # Define the first columns of the .tsv
    # The others are still written to the file after these ones
    first_columns = ['test_time', 'commit', 'tool',
                     'problem', 'instance', 'ans', 'elapsed', 'seed']
    other_columns = [c for c in results.columns if c not in first_columns]
    results = results[first_columns + other_columns]

    save_results(results, OUTPUT_PATH.joinpath(f'{test_time}.tsv'))


if __name__ == '__main__':
    main()
