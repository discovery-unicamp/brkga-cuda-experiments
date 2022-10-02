import logging
import subprocess


def shell(cmd: str, get: bool = True) -> str:
    logging.debug(f'Execute command `{cmd}`')
    output = ''
    try:
        stdout = subprocess.PIPE if get else None
        process = subprocess.run(cmd, stdout=stdout, text=True,
                                 shell=True, check=True)
        output = process.stdout.strip() if get else ''
    except subprocess.CalledProcessError as error:
        output = error.stdout.strip() if get else ''
        raise
    finally:
        if output:
            logging.debug(f'Script output:\n{output}')

    return output
