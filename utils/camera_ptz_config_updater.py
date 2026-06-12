# -*- coding: utf-8 -*-
"""实时同步当前被控摄像头 PTZ 到 camera_projector_config.json。

本版本把“接口中文名 -> camera_name”的映射、yaw 校准偏移量、PTZ 接口地址
从独立运行时配置文件中读取，便于在线调试时只改配置、不改代码。
默认运行时配置文件建议放在：config/camera_projector_runtime_config.json。
"""
from __future__ import annotations

import json
import logging
import math
import os
import ssl
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.request import Request, urlopen

PTZ_URL = "https://10.1.129.99:8443/camcontrol/camera/ptz?handle=1&chn=1"
CURRENT_DEVICE_NAME_URL = "https://10.1.129.99:8443/camcontrol/camera/currentDeviceName"

# 兜底默认值。实际运行时优先使用 config/camera_projector_runtime_config.json 中的 realtime_ptz 配置。
INTERFACE_NAME_TO_CAMERA_NAME = {
    "罗家集": "罗家集",
    "商储大厦": "商储大厦",
    "世茂广场": "世贸广场",
    "岗下江南郡九栋楼面": "岗下江南郡",
    "中央香榭2栋楼面": "中央香榭",
    "新建长堎立交西单杆站": "国动塔",
    "龙王庙小区104栋楼顶": "龙王庙",
    "锦天府汉庭酒店楼顶": "锦天府汉庭酒店",
    "绿地国际博览城-博浩205栋楼顶": "绿地国际博览城",
    "交投大厦旁国动塔": "交投大厦旁国动塔",
    "东湖急救中心": "东湖急救中心",
}

YAW_OFFSET_BY_CAMERA_NAME = {
    "罗家集": 253.5,
    "商储大厦": 10.27,
    "世贸广场": 110.9,
    "岗下江南郡": -318.16,
    "中央香榭": 34.22,
    "国动塔": -90.0,
    "龙王庙": 133.85,
    "锦天府汉庭酒店": -54.0,
    "绿地国际博览城": 0.0,
    "交投大厦旁国动塔": -72.73,
    "东湖急救中心": 145.57,
}

_DYNAMIC_KEYS = ("gimbal_yaw", "gimbal_pitch", "gimbal_roll", "zoom_factor")
_EPS = {"gimbal_yaw": 0.01, "gimbal_pitch": 0.01, "gimbal_roll": 0.001, "zoom_factor": 0.001}


def _normalize360(angle: float) -> float:
    value = float(angle) % 360.0
    if value >= 360.0:
        value -= 360.0
    if value < 0.0:
        value += 360.0
    return value


def _angle_diff_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _config_file_sig(path_value: Optional[Path]) -> Optional[Tuple[str, int, int]]:
    if path_value is None:
        return None
    try:
        st = path_value.stat()
        return str(path_value.resolve()), int(st.st_mtime_ns), int(st.st_size)
    except Exception:
        return None


class RealtimePtzConfigUpdater:
    def __init__(
        self,
        config_path: str,
        poll_interval: float = 1.0,
        request_timeout: float = 1.5,
        verify_ssl: bool = False,
        logger: Optional[logging.Logger] = None,
        runtime_config_path: Optional[str] = None,
    ):
        # camera_projector_config.json：由本线程实时写入 PTZ 结果。
        self.config_path = Path(str(config_path)).resolve()

        # camera_projector_runtime_config.json：由用户手动维护；本线程只读，不写。
        self.runtime_config_path = Path(str(runtime_config_path)).resolve() if runtime_config_path else None
        self._runtime_config_sig: Optional[Tuple[str, int, int]] = None

        self.poll_interval = max(0.2, float(poll_interval))
        self.request_timeout = max(0.2, float(request_timeout))
        self.logger = logger or logging.getLogger(__name__)
        self._ssl_context = None if verify_ssl else ssl._create_unverified_context()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_warning_ts: Dict[str, float] = {}
        self._last_written_signature: Optional[Tuple[str, Tuple[Tuple[str, float], ...]]] = None

        self.ptz_url = PTZ_URL
        self.current_device_name_url = CURRENT_DEVICE_NAME_URL
        self.interface_name_to_camera_name = dict(INTERFACE_NAME_TO_CAMERA_NAME)
        self.yaw_offset_by_camera_name = dict(YAW_OFFSET_BY_CAMERA_NAME)

        self._reload_runtime_config_if_changed(force=True)

    def start(self) -> "RealtimePtzConfigUpdater":
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RealtimePtzConfigUpdater", daemon=True)
        self._thread.start()
        self.logger.info(
            f"实时PTZ配置同步线程已启动: target={self.config_path}, runtime_config={self.runtime_config_path}"
        )
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.logger.info("实时PTZ配置同步线程已停止")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as exc:
                self._warn_throttled("poll_exception", f"实时PTZ配置同步异常: {exc}", 5.0)
            self._stop_event.wait(self.poll_interval)

    def _reload_runtime_config_if_changed(self, force: bool = False) -> bool:
        """热加载只读运行时配置：PTZ接口地址、中文名映射、yaw偏移量。"""
        if self.runtime_config_path is None:
            return False
        sig = _config_file_sig(self.runtime_config_path)
        if sig is None:
            self._warn_throttled(
                "runtime_config_missing",
                f"实时PTZ运行时配置文件不存在或不可读: {self.runtime_config_path}",
                10.0,
            )
            return False
        if not force and sig == self._runtime_config_sig:
            return False
        try:
            with open(self.runtime_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("运行时配置 JSON 顶层必须是 object")
            rt = data.get("realtime_ptz") or data.get("ptz") or {}
            if not isinstance(rt, dict):
                rt = {}

            self.ptz_url = str(rt.get("ptz_url") or PTZ_URL).strip()
            self.current_device_name_url = str(rt.get("current_device_name_url") or CURRENT_DEVICE_NAME_URL).strip()

            mapping = rt.get("interface_name_to_camera_name") or rt.get("INTERFACE_NAME_TO_CAMERA_NAME")
            if isinstance(mapping, dict) and mapping:
                self.interface_name_to_camera_name = {str(k).strip(): str(v).strip() for k, v in mapping.items() if str(k).strip() and str(v).strip()}
            else:
                self.interface_name_to_camera_name = dict(INTERFACE_NAME_TO_CAMERA_NAME)

            offsets = rt.get("yaw_offset_by_camera_name") or rt.get("YAW_OFFSET_BY_CAMERA_NAME")
            if isinstance(offsets, dict) and offsets:
                parsed = {}
                for k, v in offsets.items():
                    try:
                        parsed[str(k).strip()] = float(v)
                    except Exception:
                        pass
                self.yaw_offset_by_camera_name = parsed or dict(YAW_OFFSET_BY_CAMERA_NAME)
            else:
                self.yaw_offset_by_camera_name = dict(YAW_OFFSET_BY_CAMERA_NAME)

            self._runtime_config_sig = sig
            self.logger.info(f"实时PTZ运行时配置已加载: {self.runtime_config_path}")
            return True
        except Exception as exc:
            self._warn_throttled("runtime_config_bad", f"加载实时PTZ运行时配置失败: {exc}", 5.0)
            return False

    def poll_once(self) -> bool:
        self._reload_runtime_config_if_changed(force=False)
        sample = self._read_synced_sample()
        if sample is None:
            return False
        interface_name, camera_name, ptz = sample
        values = self._convert_ptz(camera_name, ptz)
        if values is None:
            return False
        return self._update_config_if_needed(interface_name, camera_name, ptz, values)

    def _http_get_text(self, url: str) -> str:
        req = Request(url, headers={"User-Agent": "camera-projector-ptz-updater/1.0"})
        with urlopen(req, timeout=self.request_timeout, context=self._ssl_context) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
        for enc in (charset, "utf-8", "gbk"):
            try:
                return raw.decode(enc).strip()
            except Exception:
                pass
        return raw.decode("utf-8", errors="ignore").strip()

    def _fetch_device_name(self) -> Optional[str]:
        text = self._http_get_text(self.current_device_name_url)
        name = text.strip().strip('"').strip()
        return name or None

    def _fetch_ptz(self) -> Optional[Dict[str, float]]:
        text = self._http_get_text(self.ptz_url)
        try:
            data = json.loads(text)
        except Exception:
            self._warn_throttled("bad_ptz_json", f"PTZ接口返回内容不是有效JSON: {text[:200]}", 5.0)
            return None
        if not isinstance(data, dict) or "error" in data:
            return None
        try:
            return {"x": float(data["x"]), "y": float(data["y"]), "z": float(data["z"])}
        except Exception:
            self._warn_throttled("bad_ptz_fields", f"PTZ接口缺少 x/y/z 或数值异常: {data}", 5.0)
            return None

    def _read_synced_sample(self) -> Optional[Tuple[str, str, Dict[str, float]]]:
        # 同步确认：设备名 -> PTZ -> 设备名。两次设备名一致才使用中间这次 PTZ。
        name_before = self._fetch_device_name()
        if not name_before:
            return None
        ptz = self._fetch_ptz()
        if ptz is None:
            return None
        name_after = self._fetch_device_name()
        if name_before != name_after:
            self._warn_throttled(
                "device_name_changed",
                f"跳过本次PTZ更新：读取过程中被控摄像头发生变化，before={name_before}, after={name_after}",
                3.0,
            )
            return None
        camera_name = self.interface_name_to_camera_name.get(name_before)
        if not camera_name:
            self._warn_throttled(f"unknown:{name_before}", f"接口摄像头名称未配置映射: {name_before}", 10.0)
            return None
        return name_before, camera_name, ptz

    def _convert_ptz(self, camera_name: str, ptz: Dict[str, float]) -> Optional[Dict[str, float]]:
        if camera_name not in self.yaw_offset_by_camera_name:
            self._warn_throttled(f"offset:{camera_name}", f"缺少yaw校准偏移量: {camera_name}", 10.0)
            return None
        yaw = _normalize360(ptz["x"] + self.yaw_offset_by_camera_name[camera_name])
        pitch = -ptz["y"]
        zoom = ptz["z"]
        if not all(math.isfinite(v) for v in (yaw, pitch, zoom)):
            return None
        return {
            "gimbal_yaw": round(yaw, 6),
            "gimbal_pitch": round(pitch, 6),
            "gimbal_roll": 0.0,
            "zoom_factor": round(zoom, 6),
        }

    def _load_config(self) -> Optional[Dict[str, Any]]:
        if not self.config_path.exists():
            self._warn_throttled("config_missing", f"投影配置文件不存在: {self.config_path}", 10.0)
            return None
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _find_camera_entry(data: Dict[str, Any], camera_name: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        cameras = data.get("cameras")
        if not isinstance(cameras, dict):
            return None, None
        for key, cfg in cameras.items():
            if not isinstance(cfg, dict):
                continue
            if str(cfg.get("camera_name", "")).strip() == camera_name:
                return str(key), cfg
            aliases = cfg.get("aliases") or []
            if isinstance(aliases, (list, tuple)) and camera_name in [str(x).strip() for x in aliases]:
                return str(key), cfg
        return None, None

    @staticmethod
    def _changed(old_info: Dict[str, Any], new_values: Dict[str, float]) -> bool:
        for key in _DYNAMIC_KEYS:
            if key not in old_info:
                return True
            try:
                diff = _angle_diff_deg(old_info[key], new_values[key]) if key == "gimbal_yaw" else abs(float(old_info[key]) - float(new_values[key]))
            except Exception:
                return True
            if diff > _EPS[key]:
                return True
        return False

    def _update_config_if_needed(self, interface_name: str, camera_name: str, raw_ptz: Dict[str, float], values: Dict[str, float]) -> bool:
        data = self._load_config()
        if data is None:
            return False
        camera_key, camera_cfg = self._find_camera_entry(data, camera_name)
        if camera_cfg is None:
            self._warn_throttled(f"notfound:{camera_name}", f"配置文件中找不到 camera_name={camera_name}", 10.0)
            return False
        info = camera_cfg.get("info")
        if not isinstance(info, dict):
            info = {}
            camera_cfg["info"] = info
        if not self._changed(info, values):
            return False

        info.update(values)
        camera_cfg["realtime_ptz"] = {
            "enabled": True,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "interface_device_name": interface_name,
            "camera_name": camera_name,
            "raw_ptz": {"x": raw_ptz["x"], "y": raw_ptz["y"], "z": raw_ptz["z"]},
        }
        self._atomic_write_json(data)

        signature = (str(camera_key), tuple((k, float(values[k])) for k in _DYNAMIC_KEYS))
        if signature != self._last_written_signature:
            self.logger.info(
                f"实时PTZ已写入投影配置: camera_key={camera_key}, camera_name={camera_name}, "
                f"yaw={values['gimbal_yaw']:.3f}, pitch={values['gimbal_pitch']:.3f}, zoom={values['zoom_factor']:.3f}"
            )
            self._last_written_signature = signature
        return True

    def _atomic_write_json(self, data: Dict[str, Any]) -> None:
        parent = self.config_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.config_path.name}.", suffix=".tmp", dir=str(parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.config_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _warn_throttled(self, key: str, message: str, min_interval: float) -> None:
        now = time.time()
        if now - self._last_warning_ts.get(key, 0.0) >= min_interval:
            self.logger.warning(message)
            self._last_warning_ts[key] = now


_UPDATER_INSTANCE: Optional[RealtimePtzConfigUpdater] = None
_UPDATER_LOCK = threading.Lock()


def start_realtime_ptz_config_updater(
    config_path: str,
    enabled: bool = True,
    poll_interval: float = 1.0,
    request_timeout: float = 1.5,
    verify_ssl: bool = False,
    logger: Optional[logging.Logger] = None,
    runtime_config_path: Optional[str] = None,
) -> Optional[RealtimePtzConfigUpdater]:
    global _UPDATER_INSTANCE
    if not enabled:
        return None
    with _UPDATER_LOCK:
        if _UPDATER_INSTANCE is None:
            _UPDATER_INSTANCE = RealtimePtzConfigUpdater(
                config_path=config_path,
                poll_interval=poll_interval,
                request_timeout=request_timeout,
                verify_ssl=verify_ssl,
                logger=logger,
                runtime_config_path=runtime_config_path,
            )
        return _UPDATER_INSTANCE.start()
