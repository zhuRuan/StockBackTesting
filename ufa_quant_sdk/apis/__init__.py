import requests
from ufa_quant_sdk.config import API_KEY
from ufa_quant_sdk.utils import abspath
from ufa_quant_sdk.utils.logger_tools import get_general_logger

logger = get_general_logger("API_DEBUG", path=abspath("logs"))
DEBUG = False


def _post_request(site, endpoint, payload):
    payload["API_KEY"] = API_KEY
    r = requests.post(site + endpoint, json=payload, timeout=10)
    if DEBUG:
        logger.info(f"{r}")
    return r


def _post_request_data(site, endpoint, payload):
    return _post_request(site, endpoint, payload).json()["data"]


def _get_request(site, endpoint, payload):
    payload["API_KEY"] = API_KEY
    r = requests.get(site + endpoint, params=payload, timeout=10)
    if DEBUG:
        logger.info(f"{r}")
    return r


def _get_request_data(site, endpoint, payload):
    return _get_request(site, endpoint, payload).json()["data"]
