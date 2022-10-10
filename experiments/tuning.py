from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union

from experiment import MAX_GENERATIONS, MAX_TIME_SECONDS, PROBLEM_NAME, compile_optimizer
from instance import get_instance_path
from shell import shell


PRECISION = 2
TUNING_PATH = Path('experiments', 'tuning')

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
            ptype: Literal['int', 'float', 'category', 'ordinal'],
            values: Union[Tuple[int, int],
                          Tuple[float, float],
                          List[Union[str, int, float]]],
            condition: Optional[str] = None,
    ):
        if ptype not in IraceParam.TYPES:
            raise ValueError(f'Unknown type: {ptype}')

        if ptype in ('int', 'float'):
            if not isinstance(values, tuple):
                raise ValueError(f"For {ptype} you should provide a values as a"
                                 f" tuple, not {type(values)}")
        else:
            if not isinstance(values, list):
                raise ValueError(f"For {ptype} you should provide a values as a"
                                 f" list, not {type(values)}")

        self.name = name
        self.ptype = ptype
        self.values = values
        self.condition = condition

    def __str__(self) -> str:
        name = self.name.replace('-', '_')
        arg = f"--{self.name} "
        ptype = IraceParam.TYPES[self.ptype]
        values = ', '.join(f'"{v}"' if isinstance(v, str) else str(v)
                        for v in self.values)
        cond = '' if not self.condition else f' | {self.condition}'
        return f'{name}\t"{arg}"\t{ptype}\t({values}){cond}'


def irace(
        results_path: Path,
        executable: Path,
        instances: List[Path],
        fixed_params: Dict[str, Union[str, int, float]],
        tune_params: List[IraceParam],
        forbidden_combinations: List[str],
        check: bool = False,
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
     >>experiments.txt
$EXE --instance "$INSTANCE" --seed $SEED $FIXED_PARAMS $TUNE_PARAMS \\
     1> $STDOUT 2> $STDERR

if [ -s "$STDOUT" ]; then
    cost=$(tail -n 1 "$STDOUT" | grep -o 'ans=[0-9.]*' | grep -o '[0-9.]*')
    time=$(tail -n 1 "$STDOUT" | grep -o 'elapsed=[0-9.]*' | grep -o '[0-9.]*')
    echo $cost $time
    rm -f "$STDOUT" "$STDERR"
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
        'log-file': str(results_path.joinpath('results.Rdata').absolute()),
        'deterministic': 0,
        'parallel': 1,
        'seed': 0,
    }
    if check:
        irace_params['check'] = ''  # no value required

    output_path = results_path.joinpath('output.txt')
    irace_cmd = (
        "irace "
        + " ".join(f"--{name} {value}" for name, value in irace_params.items())
        + " >" + str(output_path.absolute())
    )
    shell(f'cd {str(results_path.absolute())} && {irace_cmd}', get=False)


def tune_box_2(problem: str):
    tool = 'brkga-cuda-2.0'
    problem_name = PROBLEM_NAME[problem]
    irace(
        results_path=TUNING_PATH.joinpath(problem, tool),
        executable=compile_optimizer(tool, problem_name),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS,
            'decoder': 'cpu',
            'log-step': 0,
        },
        tune_params=[
            IraceParam('threads', 'category', [64, 128, 256]),
            IraceParam('pop-count', 'int', (3, 8)),
            IraceParam('pop-size', 'category', [64, 128, 256]),
            IraceParam('rhoe', 'float', (.70, .90)),
            IraceParam('elite', 'float', (.02, .20)),
            IraceParam('mutant', 'float', (.02, .20)),
            IraceParam('exchange-interval', 'category', [25, 50, 75, 100]),
            IraceParam('exchange-count', 'int', (1, 3)),
            IraceParam('pr-interval', 'category', [50, 100, 150, 200]),
            IraceParam('pr-pairs', 'int', (2, 5)),
            IraceParam('pr-block-factor', 'float', (.05, .15)),
            IraceParam('similarity-threshold', 'float', (.90, .98)),
        ],
        forbidden_combinations=[
            'elite * as.numeric(pop_size) < pr_pairs',
            'elite * as.numeric(pop_size) < exchange_count',
        ],
        check=False,
    )


def tune_brkga_mp_ipr(problem: str):
    tool = 'brkga-mp-ipr'
    problem_name = PROBLEM_NAME[problem]
    irace(
        results_path=TUNING_PATH.joinpath(problem, tool),
        executable=compile_optimizer(tool, problem_name),
        instances=[get_instance_path(problem_name, i)
                   for i in TUNING_INSTANCES[problem_name]],
        fixed_params={
            'omp-threads': shell('nproc'),
            'generations': MAX_GENERATIONS,
            'max-time': MAX_TIME_SECONDS,
            'decoder': 'cpu',
            'log-step': 0,
            'pr-pairs': 1,
        },
        tune_params=[
            IraceParam('pop-count', 'int', (1, 8)),
            IraceParam('pop-size', 'int', (64, 256)),
            IraceParam('rhoe-function', 'category',
                       ['lin', 'quad', 'cub', 'exp', 'log', 'const', 'rhoe']),
            IraceParam('rhoe', 'float', (.70, .90),
                       condition='rhoe_function == "rhoe"'),
            IraceParam('elite', 'float', (.02, .20)),
            IraceParam('mutant', 'float', (.02, .20)),
            IraceParam('parents', 'int', (2, 10)),
            IraceParam('elite-parents', 'int', (1, 9)),
            IraceParam('exchange-interval', 'category', [25, 50, 75, 100]),
            IraceParam('exchange-count', 'int', (1, 3)),
            IraceParam('pr-interval', 'category', [50, 100, 150, 200]),
            IraceParam('pr-block-factor', 'float', (.05, 1.0)),
            IraceParam('pr-max-time', 'int', (1, 30)),
            IraceParam('pr-select', 'category', ['best', 'random']),
            IraceParam('similarity-threshold', 'float', (.90, .98)),
        ],
        forbidden_combinations=[
            '(elite * pop_size < elite_parents)',
            '(elite * pop_size < exchange_count)',
            '(parents <= elite_parents)',
            '(rhoe_function == "rhoe") & (parents > 2)',
        ],
        check=False,
    )


if __name__ == '__main__':
    # tune_box_2('cvrp')
    tune_brkga_mp_ipr('cvrp')
