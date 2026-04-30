# -*- coding: utf-8 -*-
"""
告警上报接口模块：登录平台并上报告警信息
"""
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict
from urllib.parse import urljoin

from nan.post_request import post_requests_response_with_meta

logger = logging.getLogger(__name__)


class PlatformApiClient:
    """巡检平台 API 客户端：负责登录、全局 token 复用与告警上报。"""

    LOGIN_PATH = "/jeecg-boot/sys/login/simple"
    ALARM_REPORT_PATH = "/jeecg-boot/sys/api/ai-alarm"
    TOKEN_CACHE_FILE = os.path.join(os.path.dirname(__file__), "platform_token_cache.json")

    _shared_lock = threading.RLock()
    _shared_tokens: Dict[str, str] = {}
    _cache_hit_logged_keys = set()
    _cache_loaded = False

    def __init__(self, config=None):
        self.config = config

    def is_enabled(self) -> bool:
        return bool(
            self.config
            and getattr(self.config, "platform_report_enabled", True)
            and getattr(self.config, "platform_base_url", "")
            and getattr(self.config, "platform_username", "")
            and getattr(self.config, "platform_password", "")
        )

    def _build_url(self, path: str) -> str:
        base_url = str(getattr(self.config, "platform_base_url", "") or "").rstrip("/") + "/"
        return urljoin(base_url, path.lstrip("/"))

    def _cache_key(self) -> str:
        base_url = str(getattr(self.config, "platform_base_url", "") or "").rstrip("/")
        username = str(getattr(self.config, "platform_username", "") or "")
        return f"{base_url}|{username}"

    @classmethod
    def _load_token_cache(cls) -> None:
        if cls._cache_loaded:
            return
        cls._cache_loaded = True
        try:
            if not os.path.exists(cls.TOKEN_CACHE_FILE):
                return
            with open(cls.TOKEN_CACHE_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cls._shared_tokens = {
                    str(key): str(value)
                    for key, value in data.items()
                    if str(value or "").strip()
                }
        except Exception as e:
            logger.warning(f"加载平台token缓存失败: {e}")

    @classmethod
    def _save_token_cache(cls) -> None:
        try:
            with open(cls.TOKEN_CACHE_FILE, "w", encoding="utf-8") as fh:
                json.dump(cls._shared_tokens, fh, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存平台token缓存失败: {e}")

    @classmethod
    def _extract_json(cls, response_meta: Dict[str, Any]) -> Dict[str, Any]:
        value = (response_meta or {}).get("json") or {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _is_auth_error(cls, response_meta: Dict[str, Any]) -> bool:
        response_json = cls._extract_json(response_meta)
        status_code = (response_meta or {}).get("status_code")
        message = str(response_json.get("message", "") or "")
        text = str((response_meta or {}).get("text", "") or "")
        code = response_json.get("code")
        return (
            status_code == 401
            or str(code) == "401"
            or "Token失效" in message
            or "Token失效" in text
            or "重新登录" in message
            or "重新登录" in text
        )

    def _get_cached_token(self) -> str:
        self.__class__._load_token_cache()
        cache_key = self._cache_key()
        token = str(self.__class__._shared_tokens.get(cache_key, "") or "")
        if token and cache_key not in self.__class__._cache_hit_logged_keys:
            logger.info("使用全局缓存token")
            self.__class__._cache_hit_logged_keys.add(cache_key)
        return token

    def _set_cached_token(self, token: str) -> None:
        self.__class__._load_token_cache()
        cache_key = self._cache_key()
        if token:
            self.__class__._shared_tokens[cache_key] = str(token)
            self.__class__._cache_hit_logged_keys.discard(cache_key)
        else:
            self.__class__._shared_tokens.pop(cache_key, None)
            self.__class__._cache_hit_logged_keys.discard(cache_key)
        self.__class__._save_token_cache()

    def invalidate_token(self) -> None:
        with self.__class__._shared_lock:
            self.__class__._load_token_cache()
            had_token = bool(self.__class__._shared_tokens.get(self._cache_key(), ""))
            self._set_cached_token("")
            if had_token:
                logger.warning("全局缓存token已清空")

    def login(self, force: bool = False) -> str:
        if not self.is_enabled():
            return ""

        with self.__class__._shared_lock:
            if not force:
                cached_token = self._get_cached_token()
                if cached_token:
                    return cached_token

            if force:
                logger.info("开始强制重新登录获取token")
            else:
                logger.info("开始首次登录获取token")
            url = self._build_url(self.LOGIN_PATH)
            payload = {
                "username": getattr(self.config, "platform_username", ""),
                "password": getattr(self.config, "platform_password", ""),
                "captcha": getattr(self.config, "platform_captcha", ""),
                "checkKey": getattr(self.config, "platform_check_key", ""),
            }
            headers = {"Content-Type": "application/json"}
            timeout = float(getattr(self.config, "platform_login_timeout", 10.0) or 10.0)
            response_meta = post_requests_response_with_meta(url, payload, headers, params={}, timeout=timeout)
            response = self._extract_json(response_meta)
            token = str(((response or {}).get("result") or {}).get("token", "") or "")
            if token:
                self._set_cached_token(token)
                if force:
                    logger.info("重新登录成功，token已写入全局缓存")
                else:
                    logger.info("平台token获取成功，已写入全局缓存")
                return token

            logger.error(f"平台登录失败: {response}")
            self._set_cached_token("")
            return ""

    def _build_alarm_payload(self, stream_cfg: dict, alert_event, image_url: str, video_url: str) -> Dict[str, Any]:
        metadata = dict(getattr(alert_event, "metadata", {}) or {})
        target_info = dict(metadata.get("target_info") or {})
        classes_value = (
            metadata.get("classes")
            or target_info.get("classes")
            or metadata.get("class_name")
            or target_info.get("class_name")
            or alert_event.rule_id
        )
        if isinstance(classes_value, (list, tuple, set)):
            classes_value = ",".join(str(item) for item in classes_value if item not in (None, ""))
        classes_value = str(classes_value or alert_event.rule_id)

        position = (
            metadata.get("alarmAccuratePosition")
            or target_info.get("alarmAccuratePosition")
            or stream_cfg.get("alarmAccuratePosition", "")
        )
        if isinstance(position, dict):
            longitude = position.get("longitude")
            latitude = position.get("latitude")
            if longitude is not None and latitude is not None:
                position = f"{longitude},{latitude}"
            else:
                position = json.dumps(position, ensure_ascii=False)
        position = str(position or "")

        level_map = {
            "low": 1,
            "medium": 2,
            "high": 3,
            "critical": 4,
        }
        alert_level_value = level_map.get(getattr(getattr(alert_event, "alert_level", None), "value", ""), 1)
        alarm_timestamp = int(float(getattr(alert_event, "timestamp", 0.0) or 0.0) * 1000)
        alarm_id = str(getattr(alert_event, "event_id", "") or uuid.uuid4())

        return {
            "deviceType": int(stream_cfg.get("deviceType", getattr(self.config, "platform_device_type", 1)) or 1),
            "aiAlgorithmVendorId": int(
                stream_cfg.get("aiAlgorithmVendorId", getattr(self.config, "platform_vendor_id", 0)) or 0
            ),
            "alarmId": alarm_id,
            "taskId": str(stream_cfg.get("taskId", stream_cfg.get("task_id", stream_cfg.get("name", "")))),
            "gatewayDeviceSn": str(stream_cfg.get("gatewayDeviceSn", stream_cfg.get("gateway_device_sn", ""))),
            "droneDeviceSn": str(stream_cfg.get("droneDeviceSn", stream_cfg.get("drone_device_sn", ""))),
            "classes": classes_value,
            "taskType": int(stream_cfg.get("taskType", getattr(self.config, "platform_task_type", 1)) or 1),
            "alarmLevel": alert_level_value,
            "monitorEq": str(stream_cfg.get("monitorEq", stream_cfg.get("monitor_eq", stream_cfg.get("name", "")))),
            "area": str(stream_cfg.get("area", "")),
            "alarmLine": str(stream_cfg.get("rootName", stream_cfg.get("alarmLine", stream_cfg.get("name", "")))),
            "alarmTimestamp": alarm_timestamp,
            "alarmImageUrl": str(image_url or ""),
            "videoUrl": str(video_url or ""),
            "alarmAccuratePosition": position,
        }

    def report_alarm(self, stream_cfg: dict, alert_event, image_url: str, video_url: str) -> bool:
        if not self.is_enabled():
            logger.info("平台上报告警未启用，跳过报警接口调用")
            return False

        payload = self._build_alarm_payload(stream_cfg, alert_event, image_url, video_url)
        required_fields = [
            "taskId",
            "gatewayDeviceSn",
            "droneDeviceSn",
            "classes",
            "monitorEq",
            "area",
            "alarmLine",
            "alarmImageUrl",
            "videoUrl",
            "alarmAccuratePosition",
        ]
        missing = [field for field in required_fields if not str(payload.get(field, "")).strip()]
        if missing:
            logger.warning(f"报警上报缺少必要字段，跳过上报: {missing}")
            return False

        token = self.login(force=False)
        if not token:
            logger.error("报警上报前登录失败，跳过")
            return False

        url = self._build_url(self.ALARM_REPORT_PATH)
        timeout = float(getattr(self.config, "platform_report_timeout", 10.0) or 10.0)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        response_meta = post_requests_response_with_meta(url, payload, headers, params={}, timeout=timeout)
        response = self._extract_json(response_meta)
        if bool((response or {}).get("success")):
            logger.info(f"报警上报成功: {payload.get('alarmId')}")
            return True

        if self._is_auth_error(response_meta):
            logger.warning("报警上报发现token失效，清空全局token并重新登录")
            self.invalidate_token()
        else:
            logger.warning(f"报警上报返回异常，尝试重新登录后重试一次: {response}")

        token = self.login(force=True)
        if not token:
            return False

        headers["Authorization"] = f"Bearer {token}"
        response_meta = post_requests_response_with_meta(url, payload, headers, params={}, timeout=timeout)
        response = self._extract_json(response_meta)
        if bool((response or {}).get("success")):
            logger.info(f"重新登录后上报成功: {payload.get('alarmId')}")
            return True

        logger.error(f"报警上报失败: {response}")
        return False


def alarm_info_post(stream_cfg: dict, alert_event, image_url: str = "", video_url: str = "", config=None) -> bool:
    client = PlatformApiClient(config=config)
    return client.report_alarm(stream_cfg, alert_event, image_url, video_url)
