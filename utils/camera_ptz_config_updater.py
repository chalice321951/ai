# -*- coding: utf-8 -*-
"""实时同步全部摄像头 PTZ 到 camera_projector_config.json。

本版本使用“按 sn 查询单个摄像头 PTZ”的新接口，按轮询周期逐个查询所有摄像头：
    https://10.1.129.99:8443/camscontrol/camera/ptz?sn={sn}&chn=1

接口返回示例：
    {"name":"罗家集","x":263.22,"y":20.62,"z":1.0,"sn":"02352A22LA"}

写入策略：
1. 直接覆盖 camera_projector_config.json 中对应摄像头 info 下的：
   - gimbal_yaw = x
   - gimbal_pitch = -y
   - gimbal_roll = 0.0
   - zoom_factor = z
2. 不再写入 cameras.*.realtime_ptz；历史遗留的 realtime_ptz 字段会被自动删除。
3. 只有某个摄像头 info 发生变化，或需要清理 realtime_ptz / 补充 sn 字段时，才会原子写回 JSON。
4. camera_projector_runtime_config.json 仍然只读，用于维护接口地址、sn 列表、中文名映射等在线可调配置。
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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote
from urllib.request import Request, urlopen

# 新接口：按 sn 查询指定摄像头实时 PTZ。
ALL_CAMERA_PTZ_URL_TEMPLATE = "https://10.1.129.99:8443/camscontrol/camera/ptz?sn={sn}&chn=1"

# 旧接口保留为配置兼容字段，本版本默认不再使用。
LEGACY_PTZ_URL = "https://10.1.129.99:8443/camcontrol/camera/ptz?handle=1&chn=1"
LEGACY_CURRENT_DEVICE_NAME_URL = "https://10.1.129.99:8443/camcontrol/camera/currentDeviceName"

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

# 旧接口时代的 yaw 校准偏移量。新接口默认认为 x 已经是可直接写入的 gimbal_yaw，
# 因此 apply_yaw_offset 默认 False；若确需恢复旧行为，可在运行时配置中设为 True。
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

DEFAULT_CAMERA_SN_LIST = [
    {"port": 8282, "ip": "223.84.147.133", "name": "东湖急救中心", "online": True, "handle": 7, "sn": "02323A2MLJ"},
    {"port": 8282, "ip": "223.83.140.48", "name": "锦天府汉庭酒店楼顶", "online": True, "handle": 3, "sn": "02323A3MKL"},
    {"port": 8282, "ip": "223.83.128.219", "name": "岗下江南郡九栋楼面", "online": True, "handle": 2, "sn": "02323A2MKM"},
    {"port": 8282, "ip": "223.84.57.8", "name": "世茂广场", "online": True, "handle": 1, "sn": "02323A2ML3"},
    {"port": 8282, "ip": "223.83.133.38", "name": "中央香榭2栋楼面", "online": True, "handle": 4, "sn": "02323A3MLD"},
    {"port": 8282, "ip": "183.216.49.235", "name": "商储大厦", "online": True, "handle": 5, "sn": "02323A2MKT"},
    {"port": 8282, "ip": "223.83.147.78", "name": "新建长堎立交西单杆站", "online": True, "handle": 6, "sn": "02323A3MLF"},
    {"port": 8282, "ip": "223.83.140.41", "name": "绿地国际博览城-博浩205栋楼顶", "online": True, "handle": 8, "sn": "02323A3MLE"},
    {"port": 8282, "ip": "223.83.128.221", "name": "罗家集", "online": True, "handle": 9, "sn": "02352A22LA"},
    {"port": 8282, "ip": "223.83.133.39", "name": "龙王庙小区104栋楼顶", "online": True, "handle": 10, "sn": "Y2611A228T"},
    {"port": 8282, "ip": "223.83.140.38", "name": "交投大厦旁国动塔", "online": True, "handle": 11, "sn": "02311A26W9"},
]

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


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "enable", "enabled"):
        return True
    if s in ("0", "false", "no", "n", "off", "disable", "disabled"):
        return False
    return bool(default)


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
        self._last_written_signature: Optional[Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...]] = None

        self.all_camera_ptz_url_template = ALL_CAMERA_PTZ_URL_TEMPLATE
        self.legacy_ptz_url = LEGACY_PTZ_URL
        self.legacy_current_device_name_url = LEGACY_CURRENT_DEVICE_NAME_URL
        self.interface_name_to_camera_name = dict(INTERFACE_NAME_TO_CAMERA_NAME)
        self.yaw_offset_by_camera_name = dict(YAW_OFFSET_BY_CAMERA_NAME)
        self.apply_yaw_offset = True
        self.camera_sn_list: List[Dict[str, Any]] = list(DEFAULT_CAMERA_SN_LIST)

        self._reload_runtime_config_if_changed(force=True)

    def start(self) -> "RealtimePtzConfigUpdater":
        if self._thread and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="RealtimePtzConfigUpdater", daemon=True)
        self._thread.start()
        self.logger.info(
            f"实时PTZ配置同步线程已启动: target={self.config_path}, runtime_config={self.runtime_config_path}, "
            f"camera_count={len(self.camera_sn_list)}"
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
        """热加载只读运行时配置：全部摄像头 PTZ 接口地址、sn列表、中文名映射、yaw偏移量。"""
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

            self.all_camera_ptz_url_template = str(
                rt.get("all_camera_ptz_url_template")
                or rt.get("ptz_all_url_template")
                or rt.get("ptz_url_template")
                or ALL_CAMERA_PTZ_URL_TEMPLATE
            ).strip()
            self.legacy_ptz_url = str(rt.get("legacy_ptz_url") or rt.get("ptz_url") or LEGACY_PTZ_URL).strip()
            self.legacy_current_device_name_url = str(
                rt.get("legacy_current_device_name_url")
                or rt.get("current_device_name_url")
                or LEGACY_CURRENT_DEVICE_NAME_URL
            ).strip()
            self.apply_yaw_offset = _safe_bool(rt.get("apply_yaw_offset"), True)

            mapping = rt.get("interface_name_to_camera_name") or rt.get("INTERFACE_NAME_TO_CAMERA_NAME")
            if isinstance(mapping, dict) and mapping:
                self.interface_name_to_camera_name = {
                    str(k).strip(): str(v).strip()
                    for k, v in mapping.items()
                    if str(k).strip() and str(v).strip()
                }
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

            sn_entries = self._parse_runtime_sn_entries(data, rt)
            self.camera_sn_list = sn_entries or list(DEFAULT_CAMERA_SN_LIST)

            self._runtime_config_sig = sig
            self.logger.info(
                f"实时PTZ运行时配置已加载: {self.runtime_config_path}, camera_count={len(self.camera_sn_list)}, "
                f"apply_yaw_offset={self.apply_yaw_offset}"
            )
            return True
        except Exception as exc:
            self._warn_throttled("runtime_config_bad", f"加载实时PTZ运行时配置失败: {exc}", 5.0)
            return False

    @staticmethod
    def _parse_runtime_sn_entries(data: Dict[str, Any], rt: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从运行时配置读取 sn 列表。支持 realtime_ptz.camera_sn_list，也支持 cameras.*.sn。"""
        out: List[Dict[str, Any]] = []

        raw_list = (
            rt.get("camera_sn_list")
            or rt.get("camera_sn_info")
            or rt.get("sn_list")
            or rt.get("cameras")
        )
        if isinstance(raw_list, list):
            for item in raw_list:
                if isinstance(item, dict) and str(item.get("sn", "")).strip():
                    out.append(dict(item))

        cameras = data.get("cameras")
        if isinstance(cameras, dict):
            for key, cfg in cameras.items():
                if not isinstance(cfg, dict):
                    continue
                sn = str(cfg.get("sn", "")).strip()
                if not sn:
                    continue
                item = {
                    "key": str(key),
                    "sn": sn,
                    "name": cfg.get("interface_name") or cfg.get("name") or cfg.get("camera_name"),
                    "camera_name": cfg.get("camera_name"),
                }
                for k in ("port", "ip", "online", "handle"):
                    if k in cfg:
                        item[k] = cfg[k]
                out.append(item)

        # 按 sn 去重，保留第一次出现的配置。
        dedup: List[Dict[str, Any]] = []
        seen = set()
        for item in out:
            sn = str(item.get("sn", "")).strip()
            if not sn or sn in seen:
                continue
            seen.add(sn)
            dedup.append(item)
        return dedup

    def poll_once(self) -> bool:
        self._reload_runtime_config_if_changed(force=False)
        data = self._load_config()
        if data is None:
            return False

        samples = self._read_all_camera_samples(data)
        return self._update_config_if_needed(data, samples)

    def _http_get_text(self, url: str) -> str:
        req = Request(url, headers={"User-Agent": "camera-projector-ptz-updater/2.0"})
        with urlopen(req, timeout=self.request_timeout, context=self._ssl_context) as resp:
            raw = resp.read()
            charset = resp.headers.get_content_charset() or "utf-8"
        for enc in (charset, "utf-8", "gbk"):
            try:
                return raw.decode(enc).strip()
            except Exception:
                pass
        return raw.decode("utf-8", errors="ignore").strip()

    def _build_all_camera_ptz_url(self, sn: str) -> str:
        tpl = self.all_camera_ptz_url_template or ALL_CAMERA_PTZ_URL_TEMPLATE
        safe_sn = quote(str(sn).strip(), safe="")
        if "{sn}" in tpl:
            return tpl.format(sn=safe_sn)
        join_char = "&" if "?" in tpl else "?"
        return f"{tpl}{join_char}sn={safe_sn}"

    def _fetch_ptz_by_sn(self, sn: str) -> Optional[Dict[str, Any]]:
        url = self._build_all_camera_ptz_url(sn)
        text = self._http_get_text(url)
        try:
            data = json.loads(text)
        except Exception:
            self._warn_throttled(f"bad_ptz_json:{sn}", f"PTZ接口返回内容不是有效JSON: sn={sn}, text={text[:200]}", 5.0)
            return None
        if not isinstance(data, dict) or "error" in data:
            self._warn_throttled(f"ptz_error:{sn}", f"PTZ接口返回错误: sn={sn}, data={data}", 5.0)
            return None
        try:
            return {
                "name": str(data.get("name", "")).strip(),
                "sn": str(data.get("sn", sn)).strip() or str(sn).strip(),
                "x": float(data["x"]),
                "y": float(data["y"]),
                "z": float(data["z"]),
            }
        except Exception:
            self._warn_throttled(f"bad_ptz_fields:{sn}", f"PTZ接口缺少 name/sn/x/y/z 或数值异常: sn={sn}, data={data}", 5.0)
            return None

    def _effective_camera_sn_entries(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """合并运行时配置和 camera_projector_config.json 中的 sn 信息。"""
        out: List[Dict[str, Any]] = []

        for item in self.camera_sn_list:
            if isinstance(item, dict) and str(item.get("sn", "")).strip():
                out.append(dict(item))

        cameras = config_data.get("cameras")
        if isinstance(cameras, dict):
            for key, cfg in cameras.items():
                if not isinstance(cfg, dict):
                    continue
                sn = str(cfg.get("sn", "")).strip()
                if not sn:
                    continue
                out.append({
                    "key": str(key),
                    "sn": sn,
                    "name": cfg.get("interface_name") or cfg.get("camera_name"),
                    "camera_name": cfg.get("camera_name"),
                })

        if not out:
            out = list(DEFAULT_CAMERA_SN_LIST)

        dedup: List[Dict[str, Any]] = []
        seen = set()
        for item in out:
            sn = str(item.get("sn", "")).strip()
            if not sn or sn in seen:
                continue
            seen.add(sn)
            dedup.append(item)
        return dedup

    def _read_all_camera_samples(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        entries = self._effective_camera_sn_entries(config_data)
        samples: List[Dict[str, Any]] = []

        for entry in entries:
            # 运行时配置中 online=false 时跳过该摄像头，便于现场临时屏蔽离线设备。
            if entry.get("online") is False:
                continue
            sn = str(entry.get("sn", "")).strip()
            if not sn:
                continue
            ptz = self._fetch_ptz_by_sn(sn)
            if ptz is None:
                continue

            # 新接口返回的 name 是 interface_name_to_camera_name 左侧的名称。
            interface_name = str(ptz.get("name") or entry.get("name") or entry.get("interface_name") or "").strip()
            camera_name = self.interface_name_to_camera_name.get(interface_name)

            # 如果接口名称没有配置映射，则优先使用 sn 列表中已经给出的 camera_name。
            if not camera_name:
                camera_name = str(entry.get("camera_name") or "").strip()

            if not camera_name:
                self._warn_throttled(
                    f"unknown_name:{interface_name or sn}",
                    f"PTZ接口摄像头名称未配置映射且无兜底 camera_name: interface_name={interface_name}, sn={sn}",
                    10.0,
                )
                continue

            values = self._convert_ptz(camera_name, ptz)
            if values is None:
                continue

            samples.append({
                "sn": sn,
                "interface_name": interface_name,
                "camera_name": camera_name,
                "raw_ptz": {"x": ptz["x"], "y": ptz["y"], "z": ptz["z"]},
                "values": values,
                "source": "all_sn",
            })

        return samples

    def _convert_ptz(self, camera_name: str, ptz: Dict[str, Any]) -> Optional[Dict[str, float]]:
        try:
            raw_yaw = float(ptz["x"])
            if self.apply_yaw_offset:
                if camera_name not in self.yaw_offset_by_camera_name:
                    self._warn_throttled(f"offset:{camera_name}", f"缺少yaw校准偏移量: {camera_name}", 10.0)
                    return None
                yaw = _normalize360(raw_yaw + self.yaw_offset_by_camera_name[camera_name])
            else:
                yaw = _normalize360(raw_yaw)

            pitch = -float(ptz["y"])
            zoom = float(ptz["z"])
        except Exception:
            return None

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

    @staticmethod
    def _sample_signature(samples: List[Dict[str, Any]]) -> Tuple[Tuple[str, Tuple[Tuple[str, float], ...]], ...]:
        rows = []
        for item in samples:
            values = item.get("values") or {}
            camera_name = str(item.get("camera_name") or "")
            try:
                rows.append((camera_name, tuple((k, float(values[k])) for k in _DYNAMIC_KEYS)))
            except Exception:
                continue
        rows.sort(key=lambda x: x[0])
        return tuple(rows)

    def _update_config_if_needed(self, data: Dict[str, Any], samples: List[Dict[str, Any]]) -> bool:
        cameras = data.get("cameras")
        if not isinstance(cameras, dict):
            self._warn_throttled("bad_config_cameras", "camera_projector_config.json 缺少 cameras 对象", 10.0)
            return False

        changed = False
        removed_realtime_count = 0
        metadata_count = 0
        updated_count = 0

        # 1. 清理历史遗留 realtime_ptz 字段。
        for cfg in cameras.values():
            if isinstance(cfg, dict) and "realtime_ptz" in cfg:
                cfg.pop("realtime_ptz", None)
                removed_realtime_count += 1
                changed = True

        # 2. 根据 sn 列表给 camera_projector_config.json 补充 sn / interface_name，方便后续排查和兜底。
        by_camera_name: Dict[str, Dict[str, Any]] = {}
        for entry in self.camera_sn_list:
            if not isinstance(entry, dict):
                continue
            interface_name = str(entry.get("name") or entry.get("interface_name") or "").strip()
            camera_name = self.interface_name_to_camera_name.get(interface_name) or str(entry.get("camera_name") or "").strip()
            sn = str(entry.get("sn") or "").strip()
            if camera_name and sn:
                by_camera_name[camera_name] = {"sn": sn, "interface_name": interface_name}

        for camera_name, meta in by_camera_name.items():
            _key, cfg = self._find_camera_entry(data, camera_name)
            if cfg is None:
                continue
            if str(cfg.get("sn", "")).strip() != meta["sn"]:
                cfg["sn"] = meta["sn"]
                metadata_count += 1
                changed = True
            if meta.get("interface_name") and str(cfg.get("interface_name", "")).strip() != meta["interface_name"]:
                cfg["interface_name"] = meta["interface_name"]
                metadata_count += 1
                changed = True

        # 3. 批量覆盖各摄像头 info 下的实时 PTZ 字段。
        for item in samples:
            camera_name = str(item.get("camera_name") or "").strip()
            values = item.get("values") or {}
            if not camera_name or not values:
                continue
            camera_key, camera_cfg = self._find_camera_entry(data, camera_name)
            if camera_cfg is None:
                self._warn_throttled(f"notfound:{camera_name}", f"配置文件中找不到 camera_name={camera_name}", 10.0)
                continue

            info = camera_cfg.get("info")
            if not isinstance(info, dict):
                info = {}
                camera_cfg["info"] = info

            if self._changed(info, values):
                for k in _DYNAMIC_KEYS:
                    info[k] = values[k]
                updated_count += 1
                changed = True

            # 如果接口返回的 sn/interface_name 和配置不一致，顺手校正。
            sn = str(item.get("sn") or "").strip()
            interface_name = str(item.get("interface_name") or "").strip()
            if sn and str(camera_cfg.get("sn", "")).strip() != sn:
                camera_cfg["sn"] = sn
                metadata_count += 1
                changed = True
            if interface_name and str(camera_cfg.get("interface_name", "")).strip() != interface_name:
                camera_cfg["interface_name"] = interface_name
                metadata_count += 1
                changed = True

        if not changed:
            return False

        self._atomic_write_json(data)

        signature = self._sample_signature(samples)
        if signature != self._last_written_signature or removed_realtime_count or metadata_count:
            self.logger.info(
                f"实时PTZ已批量写入投影配置: updated_info={updated_count}, "
                f"cleaned_realtime_ptz={removed_realtime_count}, metadata_updates={metadata_count}, "
                f"sample_success={len(samples)}"
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
