import requests
import webbrowser


def check_internet_connection() -> bool:
    url = 'https://www.google.com/'
    timeout = 5

    try:
        requests.get(url, timeout=timeout)
        return True
    except (requests.ConnectionError, requests.Timeout) as exception:
        return False


def load_page(url: str):
    webbrowser.get().open(url)
