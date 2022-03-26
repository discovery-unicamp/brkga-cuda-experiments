from pathlib import Path


INSTANCES_PATH = Path('instances')


def get_instance_path(application: str, instance: str) -> Path:
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


def get_bks_path(application: str, instance: str) -> Path:
    path = get_instance_path(application, instance).with_suffix('.sol')
    if not path.is_file():
        raise FileNotFoundError('Cannot find the solution file')
    return path


def get_bks_value(application: str, instance: str) -> float:
    data = get_bks_path(application, instance).read_text().split('\n')
    bks = next(line for line in data if line.startswith('Cost '))
    return float(bks.split()[-1])
