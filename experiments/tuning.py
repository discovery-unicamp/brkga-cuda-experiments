from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

from experiment import (MAX_GENERATIONS, MAX_TIME_SECONDS, PROBLEM_NAME,
                        TIMEOUT_SECONDS, compile_optimizer)
from instance import get_instance_path
from shell import shell


MAX_EXPERIMENTS = 300
VALIDATION_ONLY = False
PRECISION = 2
TUNING_PATH = Path('experiments', 'tuning')

TUNING_INSTANCES = {
    'cvrp': [
        'X-n219-k73',
        'X-n599-k92',
        'X-n1001-k43',
    ],
    'scp': [
        'scp41',
        'scp45',
        'scp49',
    ],
    'tsp': [
        'zi929',
        'mu1979',
        'ca4663',
    ],
}


class IraceParam:
    TYPES = {
        'int': 'i',
        'float': 'r',
        'category': 'c',
        'ordinal': 'o',
    }

    def __init__(
            self,
            name: str,
            param_type: Literal['int', 'float', 'category', 'ordinal'],
            values: Union[Tuple[Union[int, float, str], ...],
                          List[int],
                          List[float]],
            condition: Optional[str] = None,
    ):
        if param_type not in IraceParam.TYPES:
            raise ValueError(f'Unknown type: {param_type}')

        if param_type in ('int', 'float'):
            if not isinstance(values, list):
                raise TypeError(f"For {param_type} you should provide a values "
                                f"as a list, not {type(values)}")
            if len(values) != 2:
                raise ValueError("You should specify the begin and the end"
                                 f"of the range of {param_type}")
            if values[0] > values[1]:
                raise ValueError(f"Empty range for {param_type}: {values}")
        else:
            if not isinstance(values, tuple):
                raise TypeError(f"For {param_type} you should provide a values "
                                f"as a tuple, not {type(values)}")
            if not values:
                raise ValueError(f"There no values to select in {param_type}")

        self.name = name
        self.param_type = param_type
        self.values = values
        self.condition = condition

    def __str__(self) -> str:
        name = self.name.replace('-', '_')
        arg = f"--{self.name} "
        param_type = IraceParam.TYPES[self.param_type]
        values = ', '.join(f'"{v}"' if isinstance(v, str) else str(v)
                           for v in self.values)
        cond = '' if not self.condition else f' | {self.condition}'
        return f'{name}\t"{arg}"\t{param_type}\t({values}){cond}'


def irace(
        results_path: Path,
        executable: Path,
        instances: List[Path],
        fixed_params: Dict[str, Union[str, int, float]],
        tune_params: List[IraceParam],
        forbidden_combinations: List[str],
        timeout_seconds: int = 62 * 60,
):
    def text(lines: Iterable):
        return ''.join(str(k) + '\n' for k in lines)

    results_path.joinpath('logs').mkdir(parents=True, exist_ok=True)

    instances_path = results_path.joinpath('instances-list.txt')
    instances_path.write_text(text(i.absolute() for i in instances))

    params_path = results_path.joinpath('parameters.txt')
    params_path.write_text(text(map(str, tune_params)))

    forbidden_path = results_path.joinpath('forbidden.txt')
    forbidden_path.write_text(text(forbidden_combinations))

    scenario = [
        'trainInstancesDir = ""',
        f'trainInstancesFile = "{str(instances_path.absolute())}"',
        f'parameterFile = "{str(params_path.absolute())}"',
        f'forbiddenFile = "{str(forbidden_path.absolute())}"',
        f'digits = {PRECISION}',
    ]
    scenario_path = results_path.joinpath('scenario.txt')
    scenario_path.write_text(text(scenario))

    experiments_log_path = results_path.joinpath('experiments.txt')
    experiments_log_path.unlink(missing_ok=True)

    runner = f"""#!/bin/bash
error() {{
    echo "`TZ=UTC date`: $0: error: $@"
    exit 1
}}

EXE={str(executable.absolute())}
FIXED_PARAMS="{" ".join(f"--{p} {v}" for p, v in fixed_params.items())}"

CONFIG_ID=$1
INSTANCE_ID=$2
SEED=$3
INSTANCE=$4
shift 4 || error "Not enough parameters"
TUNE_PARAMS=$*

STDOUT=logs/c$CONFIG_ID-i$INSTANCE_ID-s$SEED.stdout
STDERR=logs/c$CONFIG_ID-i$INSTANCE_ID-s$SEED.stderr

if [ ! -x "$EXE" ]; then
    error "$EXE: not found or not executable (pwd: $(pwd))"
fi

echo "$EXE --instance \"$INSTANCE\" --seed $SEED $FIXED_PARAMS $TUNE_PARAMS" \\
     >>{str(experiments_log_path.absolute())}
max_time={timeout_seconds}
timeout $max_time \\
    $EXE --instance "$INSTANCE" --seed $SEED $FIXED_PARAMS $TUNE_PARAMS \\
     1> $STDOUT 2> $STDERR

if [ -s "$STDOUT" ]; then
    cost=$(tail -n 1 "$STDOUT" | grep -o 'ans=[0-9.]*' | grep -o '[0-9.]*')
    time=$(tail -n 1 "$STDOUT" | grep -o 'elapsed=[0-9.]*' | grep -o '[0-9.]*')
    echo $cost $time
    exit 0
else
    error "$STDOUT: No such file or directory"
fi
"""
    runner_path = results_path.joinpath('target-runner')
    runner_path.write_text(runner)
    runner_path.chmod(777)

    irace_params = {
        'max-experiments': MAX_EXPERIMENTS,
        'log-file': str(results_path.joinpath('results.Rdata').absolute()),
        'deterministic': 0,  # the BRKGA is stochastic
        'parallel': 1,  # number of threads that irace should use
        'seed': 0,
    }
    if VALIDATION_ONLY:
        irace_params['check'] = ''  # no value required

    output_path = results_path.joinpath('output.txt')
    irace_cmd = (
        "irace "
        + " ".join(f"--{name} {value}" for name, value in irace_params.items())
        + " >" + str(output_path.absolute())
    )
    shell(f'cd {str(results_path.absolute())} && {irace_cmd}', get=False)
    results_path.chmod(777)


def tune_brkga_cuda(problem: str):
    tool = 'brkga-cuda-1.0'
    decoder = 'cpu'
    problem_name = PROBLEM_NAME[problem]
    irace(
        results_path=TUNING_PATH.joinpath(f'{tool}_{problem}'),
        executable=compile_optimizer(tool, problem),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS[problem_name],
            'decoder': decoder,
            'log-step': 0,
            'threads': 256,
        },
        tune_params=[
            IraceParam('pop-count', 'int', [1, 8]),
            IraceParam('pop-size', 'category', (256, 512, 768, 1024)),
            IraceParam('elite', 'float', [.02, .20]),
            IraceParam('mutant', 'float', [.02, .20]),
            IraceParam('rhoe', 'float', [.6, .9]),
            IraceParam('exchange-interval', 'int', [0, 200]),
            IraceParam('exchange-count', 'int', [1, 10]),
        ],
        forbidden_combinations=[
            'as.numeric(elite) * as.numeric(pop_size)'
                ' < as.numeric(exchange_count)',
        ],
        timeout_seconds=MAX_TIME_SECONDS[problem_name] + TIMEOUT_SECONDS,
    )


def tune_brkga_cuda_ii(problem: str):
    tool = 'brkga-cuda-2.0'
    decoder = 'cpu'
    problem_name = PROBLEM_NAME[problem]
    irace(
        results_path=TUNING_PATH.joinpath(f'{tool}_{problem}_{decoder}'),
        executable=compile_optimizer(tool, problem),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'threads': 256,
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS[problem_name],
            'decoder': decoder,
            'log-step': 0,
            'pr-interval': 0,
            'prune-interval': 0,
        },
        tune_params=[
            IraceParam('pop-count', 'int', [1, 8]),
            IraceParam('pop-size', 'int', [64, 1024]),
            IraceParam('parents', 'int', [2, 10]),
            IraceParam('elite-parents', 'int', [1, 9]),
            IraceParam('rhoe-function', 'category',
                       ('LINEAR', 'QUADRATIC', 'CUBIC', 'EXPONENTIAL',
                        'LOGARITHM', 'CONSTANT')),
            IraceParam('elite', 'float', [.02, .20]),
            IraceParam('mutant', 'float', [.02, .20]),
            IraceParam('exchange-interval', 'int', [0, 200]),
            IraceParam('exchange-count', 'int', [1, 10]),
            # IraceParam('pr-interval', 'int', [0, 200]),
            # IraceParam('pr-pairs', 'int', [1, 5]),
            # IraceParam('pr-block-factor', 'float', [.05, .15]),
            # IraceParam('pr-min-diff', 'float', [.20, .90]),
            # IraceParam('prune-interval', 'int', [0, 200]),
            # IraceParam('prune-threshold', 'float', [.90, .99]),
        ],
        forbidden_combinations=[
            'as.numeric(elite) * as.numeric(pop_size)'
                ' < as.numeric(elite_parents)',
            'as.numeric(elite) * as.numeric(pop_size)'
                ' < as.numeric(exchange_count)',
            'as.numeric(parents) <= as.numeric(elite_parents)',
            # 'as.numeric(elite) * as.numeric(pop_size) < as.numeric(pr_pairs)',
        ],
        timeout_seconds=MAX_TIME_SECONDS[problem_name] + TIMEOUT_SECONDS,
    )


def tune_gpu_brkga(problem: str, fix: bool):
    tool = 'gpu-brkga' + ('-fix' if fix else '')
    decoder = 'cpu'
    problem_name = PROBLEM_NAME[problem]
    assert problem_name != 'tsp'
    irace(
        results_path=TUNING_PATH.joinpath(f'{tool}_{problem}'),
        executable=compile_optimizer(tool, problem),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS[problem_name],
            'decoder': decoder,
            'log-step': 0,
        },
        tune_params=[
            IraceParam('pop-count', 'int', [1, 8]),
            IraceParam('pop-size', 'int', [64, 1024]),
            IraceParam('elite', 'float', [.02, .20]),
            IraceParam('mutant', 'float', [.02, .20]),
            IraceParam('rhoe', 'float', [.6, .9]),
            IraceParam('exchange-interval', 'int', [0, 200]),
            IraceParam('exchange-count', 'int', [1, 10]),
        ],
        forbidden_combinations=[
            'as.numeric(elite) * as.numeric(pop_size)'
                ' < as.numeric(exchange_count)',
        ],
        timeout_seconds=MAX_TIME_SECONDS[problem_name] + TIMEOUT_SECONDS,
    )


def tune_brkga_api(problem: str):
    tool = 'brkga-api'
    decoder = 'cpu'
    problem_name = PROBLEM_NAME[problem]
    irace(
        results_path=TUNING_PATH.joinpath(f'{tool}_{problem}'),
        executable=compile_optimizer(tool, problem),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS[problem_name],
            'decoder': decoder,
            'log-step': 0,
        },
        tune_params=[
            IraceParam('pop-count', 'int', [1, 8]),
            IraceParam('pop-size', 'int', [64, 1024]),
            IraceParam('elite', 'float', [.02, .20]),
            IraceParam('mutant', 'float', [.02, .20]),
            IraceParam('rhoe', 'float', [.6, .9]),
            IraceParam('exchange-interval', 'int', [0, 200]),
            IraceParam('exchange-count', 'int', [1, 10]),
        ],
        forbidden_combinations=[
            'elite * pop_size < exchange_count',
        ],
        timeout_seconds=MAX_TIME_SECONDS[problem_name] + TIMEOUT_SECONDS,
    )


def tune_brkga_mp_ipr(problem: str):
    tool = 'brkga-mp-ipr'
    decoder = 'cpu'
    problem_name = PROBLEM_NAME[problem]
    irace(
        results_path=TUNING_PATH.joinpath(f'{tool}_{problem}_{decoder}'),
        executable=compile_optimizer(tool, problem),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS[problem_name],
            'decoder': decoder,
            'log-step': 0,
            'pr-pairs': 1,
        },
        tune_params=[
            IraceParam('pop-count', 'int', [1, 8]),
            IraceParam('pop-size', 'int', [64, 1024]),
            IraceParam('parents', 'int', [2, 10]),
            IraceParam('elite-parents', 'int', [1, 9]),
            IraceParam('rhoe-function', 'category',
                       ('LINEAR', 'QUADRATIC', 'CUBIC', 'EXPONENTIAL',
                        'LOGARITHM', 'CONSTANT')),
            IraceParam('elite', 'float', [.02, .20]),
            IraceParam('mutant', 'float', [.02, .20]),
            IraceParam('exchange-interval', 'int', [0, 200]),
            IraceParam('exchange-count', 'int', [1, 10]),
            IraceParam('pr-interval', 'int', [0, 200]),
            IraceParam('pr-block-factor', 'float', [.05, 1.0]),
            IraceParam('pr-max-time', 'int', [1, 30]),
            IraceParam('pr-select', 'category', ('best', 'random')),
            IraceParam('pr-min-diff', 'float', [.20, .90]),
        ],
        forbidden_combinations=[
            'elite * pop_size < elite_parents',
            'elite * pop_size < exchange_count',
            'parents <= elite_parents',
        ],
        timeout_seconds=MAX_TIME_SECONDS[problem_name] + TIMEOUT_SECONDS,
    )


if __name__ == '__main__':
    # tune_gpu_brkga('scp', fix=False)
    # tune_gpu_brkga('cvrp_greedy', fix=False)
    # tune_gpu_brkga('cvrp', fix=False)
    # tune_gpu_brkga('scp', fix=True)
    # tune_gpu_brkga('cvrp_greedy', fix=True)
    # tune_gpu_brkga('cvrp', fix=True)
    # tune_brkga_api('scp')
    # tune_brkga_api('cvrp_greedy')
    # tune_brkga_api('cvrp')
    # tune_brkga_api('tsp')
    # tune_brkga_cuda('scp')
    # tune_brkga_cuda('cvrp_greedy')
    # tune_brkga_cuda('cvrp')
    # tune_brkga_cuda('tsp')
    tune_brkga_cuda_ii('scp')
    tune_brkga_cuda_ii('cvrp_greedy')
    tune_brkga_cuda_ii('cvrp')
    tune_brkga_cuda_ii('tsp')
