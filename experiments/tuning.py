from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple, Union

from experiment import MAX_GENERATIONS, MAX_TIME_SECONDS, PROBLEM_NAME, compile_optimizer
from instance import get_instance_path
from shell import shell


PRECISION = 2

TUNING_INSTANCES = {
    'cvrp': [
        'X-n219-k73',
        'X-n599-k92',
        'X-n1001-k43',
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
            type: Literal['int', 'float', 'category', 'ordinal'],
            range: Union[Tuple[int, int],
                         Tuple[float, float],
                         List[Union[str, int, float]]],
    ):
        self.name = name
        self.type = type
        self.range = range

    def __str__(self) -> str:
        name = self.name.replace('-', '')
        arg = f"--{self.name} "
        tp = IraceParam.TYPES[self.type]
        rng = ', '.join(f'"{v}"' if isinstance(v, str) else str(v)
                        for v in self.range)
        return f'{name}\t"{arg}"\t{tp}\t({rng})'


def irace(
        results_path: Path,
        executable: Path,
        instances: List[Path],
        fixed_params: Dict[str, Union[str, int, float]],
        params: List[IraceParam],
        check: bool = False,
):
    def text(lines: Iterable):
        return ''.join(str(k) + '\n' for k in lines)

    results_path.joinpath('logs').mkdir(parents=True, exist_ok=True)

    instances_path = results_path.joinpath('instances-list.txt')
    instances_path.write_text(text(i.absolute() for i in instances))

    params_path = results_path.joinpath('parameters.txt')
    params_path.write_text(text(map(str, params)))

    scenario = [
        'trainInstancesDir = ""',
        f'trainInstancesFile = "{str(instances_path.absolute())}"',
        f'parameterFile = "{str(params_path.absolute())}"',
        f'digits = {PRECISION}',
    ]
    scenario_path = results_path.joinpath('scenario.txt')
    scenario_path.write_text(text(scenario))

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
CONFIG_PARAMS=$*

STDOUT=logs/$CONFIG_ID-$INSTANCE_ID-$SEED.stdout.txt
STDERR=logs/$CONFIG_ID-$INSTANCE_ID-$SEED.stderr.txt

if [ ! -x "$EXE" ]; then
    error "$EXE: not found or not executable (pwd: $(pwd))"
fi

$EXE --instance "$INSTANCE" --seed $SEED $FIXED_PARAMS $CONFIG_PARAMS \\
     1> $STDOUT 2> $STDERR

# # This may be used to introduce a delay if there are filesystem
# # issues.
# SLEEPTIME=1
# while [ ! -s "$STDOUT" ]; do
#     sleep $SLEEPTIME
#     let "SLEEPTIME += 1"
# done

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
        'max-experiments': 300,
        'log-file': str(results_path.joinpath('results.txt').absolute()),
        'deterministic': 0,
        'parallel': 1,
        'seed': 0,
    }
    if check:
        irace_params['check'] = ''  # no value required
    irace = 'irace ' + " ".join(f"--{name} {value}"
                                for name, value in irace_params.items())
    shell(f'cd {str(results_path.absolute())} && {irace}', get=False)


def tuning(tool: str, problem: str):
    problem_name = PROBLEM_NAME[problem]
    executable = compile_optimizer(tool, problem_name)
    instances_path = [get_instance_path(problem_name, i)
                      for i in TUNING_INSTANCES[problem_name]]

    fixed_params = {
        'omp-threads': shell('nproc'),
        'generations': MAX_GENERATIONS,
        'max-time': MAX_TIME_SECONDS,
        'log-step': 0,
    }
    tune_params = [
        IraceParam('threads', 'category', [64, 128, 256]),
        IraceParam('pop-count', 'int', (3, 8)),
        IraceParam('pop-size', 'category', [64, 128, 256]),
        IraceParam('decoder', 'category', ['cpu']),
        IraceParam('rhoe', 'float', (.70, .90)),
        IraceParam('elite', 'float', (.02, .20)),
        IraceParam('mutant', 'float', (.02, .20)),
        IraceParam('exchange-interval', 'category', [25, 50, 75, 100]),
        IraceParam('exchange-count', 'int', (1, 3)),
        IraceParam('pr-interval', 'category', [50, 100, 150, 200]),
        IraceParam('pr-pairs', 'int', (2, 5)),
        IraceParam('pr-block-factor', 'float', (.05, .15)),
        IraceParam('similarity-threshold', 'float', (.90, .98)),
    ]

    irace(Path('experiments', 'tuning', problem, tool), executable,
          instances_path, fixed_params, tune_params)


if __name__ == '__main__':
    tuning('brkga-cuda-2.0', 'cvrp')
