import datetime
import itertools
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union
import pandas as pd

from result import compress_convergence, save_results
from instance import get_instance_path
from shell import shell


#             threads exchange-interval exchange-count pop-count pop-size elite   mutant  rhoe
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
TEST_COUNT = 20
MAX_GENERATIONS = 100
TIMEOUT_SECONDS = 2 * 60
OMP_THREADS = int(shell('nproc'))
BUILD_TYPE = 'release'
TWEAKS_FILE = Path('applications', 'src', 'Tweaks.hpp')
SOURCE_PATH = Path('applications')
OUTPUT_PATH = Path('experiments', 'results')
PARAMS_PATH = Path('experiments', 'parameters')
BACKUP_FILE = OUTPUT_PATH.joinpath('.backup.tsv')

MAX_TIME_SECONDS = {
    'cvrp': 60 * 60,
    'scp': 5 * 60,
    'tsp': 60 * 60,
}
PROBLEM_NAME = {
    'cvrp': 'cvrp',
    'cvrp_greedy': 'cvrp',
    'scp': 'scp',
    'tsp': 'tsp',
}
INSTANCES = {
    'cvrp': list(([
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
    ])),
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


def main():
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
            tool='brkga-cuda-2.0',
            problems=['scp'],
            decoders=['cpu'],
            test_count=TEST_COUNT,
        ),
        # __build_params(
        #     tool='brkga-mp-ipr',
        #     problems=['cvrp'],
        #     decoders=['cpu'],
        #     test_count=TEST_COUNT,
        # ),
    ))

    __save_results(results)


def __build_params(
        tool: str,
        problems: List[str],
        decoders: List[str],
        test_count: int,
) -> Iterable[Dict[str, Union[str, int, float]]]:
    if test_count == 0:
        logging.warning("Test count is 0; ignoring build params")
        return []
    for problem in problems:
        problem_name = PROBLEM_NAME[problem]
        for decoder in decoders:
            tuned_params = (
                PARAMS_PATH
                .joinpath(f'{tool}_{problem}_{decoder}.txt')
                .read_text()
                .split('\n')
            )
            tuned_params = [line.split() for line in tuned_params if line]
            param_names = [name.replace('_', '-') for name in tuned_params[0]]
            param_values = tuned_params[1]
            tuned_params = {
                name: value
                for name, value in zip(param_names, param_values)
                if value != 'NA'
            }
            tuned_params = {
                **tuned_params,
                'tool': tool,
                'problem': problem,
                'problem-name': problem_name,
                'instance-name': INSTANCES[problem_name],
                'decoder': decoder,
                'seed': range(1, test_count + 1),
                'omp-threads': OMP_THREADS,
                'generations': MAX_GENERATIONS,
                'max-time': MAX_TIME_SECONDS[problem],
                'log-step': 25,
            }

            yield from __combinations(tuned_params)


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
        executable = compile_optimizer(params['tool'], params['problem'])
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


def compile_optimizer(target: str, problem: str) -> Path:
    return __cmake(str(SOURCE_PATH.absolute()), BUILD_TYPE, target,
                   tweaks=[problem.upper()])


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
        with open(TWEAKS_FILE, 'r') as tweak_file:
            existing_tweaks_content = tweak_file.read()
    except FileNotFoundError:
        logging.warning("Tweaks file not found; generating one")
        existing_tweaks_content = ''

    if tweaks_content == existing_tweaks_content:
        # Doesn't rewrite to avoid make thinking it should recompile the code
        logging.info("Tweaks file hasn't changed")
    else:
        with open(TWEAKS_FILE, 'w') as tweak_file:
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

    cmd = f'timeout {TIMEOUT_SECONDS}s'
    cmd += ' ' + str(executable.absolute())
    cmd += ''.join(f' --{arg} {value}' for arg, value in parsed_params.items())

    start_time = __now()
    output = shell(cmd)
    output = output.split('\n')[-1]  # Read only the last line
    assert output
    result = dict(tuple(r.split('=')) for r in output.split())
    assert 'convergence' in result
    assert result['convergence'] != '[]'

    return {
        **parsed_params,
        'ans': result['ans'],
        'elapsed': result['elapsed'],
        'convergence': compress_convergence(result['convergence']),
        'start_time': start_time,
    }


def __save_results(iter_results: Iterable[Dict[str, str]]):
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    if RESUME_FROM_BACKUP and BACKUP_FILE.is_file():
        logging.warning("Using results from past backup file")
        results = pd.read_csv(BACKUP_FILE, sep='\t')
    else:
        results = pd.DataFrame()

    start_time = __now()
    for res in iter_results:
        results = pd.concat((results, pd.DataFrame([res])))
        results.to_csv(BACKUP_FILE, index=False, sep='\t')

    assert not results.empty

    save_results(
        results,
        OUTPUT_PATH.joinpath(f'{start_time}.tsv'),
        system=['system', 'cpu', 'cpu-memory', 'gpu', 'gpu-memory',
                'nvcc', 'g++', 'commit'],
        device=DEVICE,
    )


def __now() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()


if __name__ == '__main__':
    main()
