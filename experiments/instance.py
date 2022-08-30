from pathlib import Path


INSTANCES_PATH = Path('experiments', 'instances')


def get_instance_path(problem: str, instance: str) -> Path:
    group = None
    ext = None
    if problem == 'cvrp':
        ext = 'vrp'
        if instance[:2] == 'X-':
            group = 'set-x'
    elif problem == 'scp':
        ext = 'txt'
        group = 'scplib'
    elif problem == 'tsp':
        ext = 'tsp'
        group = 'tsplib'

    if ext is None:
        raise ValueError(f'Unknown problem `{problem}`')
    if group is None:
        raise ValueError(f'Unknown instance set `{instance}`')

    path = INSTANCES_PATH.joinpath(problem, group, f'{instance}.{ext}')
    if not path.is_file():
        raise FileNotFoundError(f'Could not find instance {instance}')

    return path


def get_bks_path(problem: str, instance: str) -> Path:
    path = get_instance_path(problem, instance).with_suffix('.sol')
    if not path.is_file():
        raise FileNotFoundError(f'Could not find solution file for {instance}')

    return path


def get_bks_value(problem: str, instance: str) -> float:
    data = get_bks_path(problem, instance).read_text().split('\n')
    bks = next(line for line in data if line.startswith('Cost '))
    return float(bks.split()[-1])
