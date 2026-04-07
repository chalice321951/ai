# -*- coding: utf-8 -*-
import logging
import requests

logger = logging.getLogger(__name__)


def get_requests_response(url, data, headers, params, timeout=10):
    try:
        _timeout = params.pop('_timeout_s', timeout) if isinstance(params, dict) else timeout
        response = requests.get(url, headers=headers, params=params, timeout=_timeout)
        return response.json()
    except Exception as e:
        logger.error(f"GET请求失败 {url}: {e}")
        return {}
