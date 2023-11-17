import requests


def check_internet_connection() -> bool:
    try:
        r = requests.head("http://www.google.com", timeout=2)
        return r.status_code == requests.codes.ok
    except (requests.ConnectionError, requests.Timeout):
        return False
