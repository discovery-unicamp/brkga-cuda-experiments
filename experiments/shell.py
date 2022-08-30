import logging
import subprocess


CATCH_FAILURES = False


class ShellError(RuntimeError):
    def __init__(self, stderr: str):
        stderr = stderr.strip()
        stderr = f':\n{stderr}' if stderr else ''
        super().__init__('Shell exited with error' + stderr)


def shell(cmd: str, get: bool = True) -> str:
    logging.debug(f'Execute command `{cmd}`')
    output = ''
    errors = ''
    try:
        stdout = subprocess.PIPE if get else None
        stderr = subprocess.PIPE if CATCH_FAILURES else None
        process = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True,
                                 shell=True, check=True)
        output = process.stdout.strip() if get else ''
        errors = process.stderr.strip() if CATCH_FAILURES else ''
        if errors:
            logging.warning(f'Script stderr:\n{errors}\n=======')
    except subprocess.CalledProcessError as error:
        output = error.stdout.strip() if get else ''
        errors = error.stderr.strip() if CATCH_FAILURES else ''
        raise ShellError(errors)
    finally:
        if output:
            logging.debug(f'Script output:\n{output}\n=======')

    return output
