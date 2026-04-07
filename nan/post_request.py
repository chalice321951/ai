# -*- coding: utf-8 -*-
import logging
import traceback
import requests

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'User-Agent': 'PostmanRuntime/7.43.0',
    'Connection': 'keep-alive',
}


def _preview_text(text, limit=300):
    value = str(text or '').replace('\r', ' ').replace('\n', ' ')
    if len(value) > limit:
        return value[:limit] + '...'
    return value


def _merge_headers(headers):
    merged = dict(DEFAULT_HEADERS)
    if isinstance(headers, dict):
        merged.update(headers)
    return merged


def post_requests_response(url, data, headers, params, timeout=10):
    try:
        request_headers = _merge_headers(headers)
        response = requests.post(url, json=data, headers=request_headers, params=params, timeout=timeout)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as e:
            logger.error(
                f"POST请求返回非JSON {url}: status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type', '')}, body={_preview_text(response.text)}, err={e}"
            )
            return {}
    except requests.HTTPError as e:
        response = getattr(e, 'response', None)
        if response is not None:
            logger.error(
                f"POST请求失败 {url}: status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type', '')}, body={_preview_text(response.text)}"
            )
        else:
            logger.error(f"POST请求失败 {url}: {e}")
        return {}
    except Exception as e:
        logger.error(f"POST请求失败 {url}: {e}")
        return {}


def get_requests_response(url, data, headers, params, timeout=10):
    try:
        _timeout = params.pop('_timeout_s', timeout) if isinstance(params, dict) else timeout
        request_headers = _merge_headers(headers)
        response = requests.get(url, headers=request_headers, params=params, timeout=_timeout)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as e:
            logger.error(
                f"GET请求返回非JSON {url}: status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type', '')}, body={_preview_text(response.text)}, err={e}"
            )
            return {}
    except requests.HTTPError as e:
        response = getattr(e, 'response', None)
        if response is not None:
            logger.error(
                f"GET请求失败 {url}: status={response.status_code}, "
                f"content_type={response.headers.get('Content-Type', '')}, body={_preview_text(response.text)}"
            )
        else:
            logger.error(f"GET请求失败 {url}: {e}")
        return {}
    except Exception as e:
        logger.error(f"GET请求失败 {url}: {e}")
        return {}
