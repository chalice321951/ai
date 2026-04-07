# -*- coding: utf-8 -*-
"""
告警上报接口模块：登录平台并上报告警信息
"""
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urljoin

from nan.post_request import post_requests_response

logger = logging.getLogger(__name__)


class PlatformApiClient:
    """巡检平台 API 客户端：负责登录、token 复用与告警上报。"""

    LOGIN_PATH = "/jeecg-boot/sys/login/permanent"
    ALARM_REPORT_PATH = "/jeecg-boot/sys/api/ai-alarm"
    TOKEN_REFRESH_SECONDS = 30 * 24 * 3600

    def __init__(self, config=None):
        self.config = config
        self._token: str = ""
        self._token_time = 0.0
        self._lock = threading.RLock()

    def is_enabled(self) -> bool:
        return bool(
            self.config
            and getattr(self.config, 'platform_report_enabled', True)
            and getattr(self.config, 'platform_base_url', '')
            and getattr(self.config, 'platform_username', '')
            and getattr(self.config, 'platform_password', '')
        )

    def _build_url(self, path: str) -> str:
        base_url = str(getattr(self.config, 'platform_base_url', '') or '').rstrip('/') + '/'
        return urljoin(base_url, path.lstrip('/'))

    def _token_expired(self) -> bool:
        if not self._token:
            return True
        return (time.time() - self._token_time) >= self.TOKEN_REFRESH_SECONDS

    def login(self, force: bool = False) -> str:
        if not self.is_enabled():
            return ""
        with self._lock:
            if not force and not self._token_expired():
                return self._token

            url = self._build_url(self.LOGIN_PATH)
            payload = {
                'username': getattr(self.config, 'platform_username', ''),
                'password': getattr(self.config, 'platform_password', ''),
                'captcha': getattr(self.config, 'platform_captcha', ''),
                'checkKey': getattr(self.config, 'platform_check_key', ''),
            }
            headers = {'Content-Type': 'application/json'}
            timeout = float(getattr(self.config, 'platform_login_timeout', 10.0) or 10.0)
            response = post_requests_response(url, payload, headers, params={}, timeout=timeout)
            token = str(((response or {}).get('result') or {}).get('token', '') or '')
            if token:
                self._token = token
                self._token_time = time.time()
                logger.info("平台永久token获取成功")
                return token

            logger.error(f"平台登录失败: {response}")
            self._token = ""
            self._token_time = 0.0
            return ""

    def _build_alarm_payload(self, stream_cfg: dict, alert_event, image_url: str, video_url: str) -> Dict[str, Any]:
        metadata = dict(getattr(alert_event, 'metadata', {}) or {})
        target_info = dict(metadata.get('target_info') or {})
        classes_value = metadata.get('classes') or target_info.get('classes') or metadata.get('class_name') or target_info.get('class_name') or alert_event.rule_id
        if isinstance(classes_value, (list, tuple, set)):
            classes_value = ','.join(str(item) for item in classes_value if item not in (None, ''))
        classes_value = str(classes_value or alert_event.rule_id)

        position = metadata.get('alarmAccuratePosition') or target_info.get('alarmAccuratePosition') or stream_cfg.get('alarmAccuratePosition', '')
        if isinstance(position, dict):
            longitude = position.get('longitude')
            latitude = position.get('latitude')
            if longitude is not None and latitude is not None:
                position = f"{longitude},{latitude}"
            else:
                position = json.dumps(position, ensure_ascii=False)
        position = str(position or '')

        level_map = {
            'low': 1,
            'medium': 2,
            'high': 3,
            'critical': 4,
        }
        alert_level_value = level_map.get(getattr(getattr(alert_event, 'alert_level', None), 'value', ''), 1)
        alarm_timestamp = int(float(getattr(alert_event, 'timestamp', time.time())) * 1000)
        alarm_id = str(getattr(alert_event, 'event_id', '') or uuid.uuid4())

        payload = {
            'deviceType': int(stream_cfg.get('deviceType', getattr(self.config, 'platform_device_type', 1)) or 1),
            'aiAlgorithmVendorId': int(stream_cfg.get('aiAlgorithmVendorId', getattr(self.config, 'platform_vendor_id', 0)) or 0),
            'alarmId': alarm_id,
            'taskId': str(stream_cfg.get('taskId', stream_cfg.get('task_id', stream_cfg.get('name', '')))),
            'gatewayDeviceSn': str(stream_cfg.get('gatewayDeviceSn', stream_cfg.get('gateway_device_sn', ''))),
            'droneDeviceSn': str(stream_cfg.get('droneDeviceSn', stream_cfg.get('drone_device_sn', ''))),
            'classes': classes_value,
            'taskType': int(stream_cfg.get('taskType', getattr(self.config, 'platform_task_type', 1)) or 1),
            'alarmLevel': alert_level_value,
            'monitorEq': str(stream_cfg.get('monitorEq', stream_cfg.get('monitor_eq', stream_cfg.get('name', '')))),
            'area': str(stream_cfg.get('area', '')),
            'alarmLine': str(stream_cfg.get('rootName', stream_cfg.get('alarmLine', stream_cfg.get('name', '')))),
            'alarmTimestamp': alarm_timestamp,
            'alarmImageUrl': str(image_url or ''),
            'videoUrl': str(video_url or ''),
            'alarmAccuratePosition': position,
        }
        return payload

    def report_alarm(self, stream_cfg: dict, alert_event, image_url: str, video_url: str) -> bool:
        if not self.is_enabled():
            logger.info("平台上报告警未启用，跳过报警接口调用")
            return False

        payload = self._build_alarm_payload(stream_cfg, alert_event, image_url, video_url)
        required_fields = [
            'taskId', 'gatewayDeviceSn', 'droneDeviceSn', 'classes', 'monitorEq',
            'area', 'alarmLine', 'alarmImageUrl', 'videoUrl', 'alarmAccuratePosition',
        ]
        missing = [field for field in required_fields if not str(payload.get(field, '')).strip()]
        if missing:
            logger.warning(f"报警上报缺少必要字段，跳过上报: {missing}")
            return False

        token = self.login(force=False)
        if not token:
            logger.error("报警上报前登录失败，跳过")
            return False

        url = self._build_url(self.ALARM_REPORT_PATH)
        timeout = float(getattr(self.config, 'platform_report_timeout', 10.0) or 10.0)
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
        }
        response = post_requests_response(url, payload, headers, params={}, timeout=timeout)
        if bool((response or {}).get('success')):
            logger.info(f"报警上报成功: {payload.get('alarmId')}")
            return True

        logger.warning(f"报警上报返回异常，尝试重新登录后重试一次: {response}")
        token = self.login(force=True)
        if not token:
            return False
        headers['Authorization'] = f'Bearer {token}'
        response = post_requests_response(url, payload, headers, params={}, timeout=timeout)
        if bool((response or {}).get('success')):
            logger.info(f"报警上报成功(重试): {payload.get('alarmId')}")
            return True

        logger.error(f"报警上报失败: {response}")
        return False


def alarm_info_post(stream_cfg: dict, alert_event, image_url: str = '', video_url: str = '', config=None) -> bool:
    client = PlatformApiClient(config=config)
    return client.report_alarm(stream_cfg, alert_event, image_url, video_url)
