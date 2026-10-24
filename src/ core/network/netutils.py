import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

_public_servers = [
    'https://www.google.com/',
    'https://www.cloudflare.com/',
    'https://www.opendns.com/',
    'https://quad9.net/',
    'https://www.baidu.com/',
    'https://www.alibabacloud.com/'
]


def check_internet_connection(timeout: float = 5.0) -> bool:
    """ Checks internet connection by sending requests to public servers. Returns True If at least one responds.

        Args
        ----
        :param timeout: (float) connection timeout in seconds.
    """

    def probe(url: str) -> bool:
        """ Sends a request and returns Whether sever responded or not.

            Args
            ----
            :param url: (str) public server URL to probe.
        """

        try:
            requests.head(url=url, timeout=timeout)
            return True
        except (requests.ConnectionError, requests.Timeout):
            return False

    # Computing max concurrent workers.
    max_workers = min(len(_public_servers), max(1, os.cpu_count() - 1))

    # Sending requests to public domains to check internet connection.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(probe, url) for url in _public_servers]

        for fut in as_completed(futures):
            if fut.result():
                return True
    return False
