import json
import time
import webbrowser
from multiprocessing import Process
from typing import Optional
from urllib.request import urlopen
from optuna_dashboard import run_server


def get_study_filepath(league_id: str, entity: str) -> str:
    """ Returns storage url of optuna's study filepath for the specified league.

        Args
        ----
        :param league_id: (str) league id for optuna study.
        :param entity: (str) entity of the study. It supports "models" and "agents".
    """

    # Validating entity param.
    if not (entity == 'models' or entity == 'agents'):
        raise ValueError(f'Expected entity to equal "models" or "agents", got "{entity}"')

    # Load storage settings for the study.
    with open('storage/configs/studies.json', mode='r', encoding='utf-8') as jsonfile:
        config = json.load(jsonfile)

    base_dir = config['storage_base_directory']
    return f'{base_dir}/{league_id}/{entity}_study.db'


def get_storage_url(study_filepath: str) -> str:
    """ Helper function that returns storage url of optuna's study url.

        Args
        ----
        :param study_filepath: (str) valid filepath of the study.
    """

    return f'sqlite:///{study_filepath}'


def start_optuna_dashboard(storage_url: str, host: str = '127.0.0.1', port: int = 8080) -> Process:
    """ Initializes Optuna Dashboard in a separate process.

        :param storage_url: (str) storage url for optuna study.
        :param host: (str) host for optuna study. Defaults to 127.0.0.1 (localhost).
        :param port: (int) port for optuna study. Defaults to 8080.
     """

    process = Process(target=run_server, args=(storage_url,), kwargs={'host': host, 'port': port})
    process.start()
    return process


def stop_optuna_dashboard(process: Optional[Process]):
    """ Stop the Optuna Dashboard process.

        :param process: (Process) the Optuna Dashboard process, initialized by "start_optuna_dashboard" method.
    """

    if process is None:
        return

    if process.is_alive():
        process.terminate()

    # Reap the child even if it exited by itself during server startup.
    if process.pid is not None:
        process.join()


def wait_optuna_server(
        host: str = '127.0.0.1',
        port: int = 8080,
        timeout: float = 60.0,
        retry_fallback: float = 5.0,
        open_webbrowser: bool = True
):
    """ Polls an HTTP URL until it returns a response or times out.

        :param host: (str) host for optuna study. Defaults to 127.0.0.1 (localhost).
        :param port: (int) port for optuna study. Defaults to 8080.
        :param timeout: (float) time to keep trying (in seconds).
        :param retry_fallback: (float) fallback time to retry (in seconds).
        :param open_webbrowser: (bool) whether to open webbrowser after the server has started.
    """

    dashboard_url = f'http://{host}:{port}'

    is_listening = False
    start_time = time.time()
    while time.time() - start_time < timeout and not is_listening:
        try:
            with urlopen(url=dashboard_url, timeout=1.0) as response:
                if response.status == 200:
                    is_listening = True
        except Exception as _:
            pass
        time.sleep(retry_fallback)

    if open_webbrowser and is_listening:
        try:
            webbrowser.open('http://127.0.0.1:8080')
        except Exception as _:
            pass
