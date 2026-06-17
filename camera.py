#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI摄像头流检测主程序
支持多路RTSP/RTMP流并发检测、AI推理、告警、FFmpeg推送AI结果流
"""

import atexit
import faulthandler
import json
import logging
import multiprocessing
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

# ── 全局 NVENC 会话计数器 ──
_nvenc_lock = threading.Lock()
_nvenc_count = 0
_NVENC_MAX_SESSIONS = 11  # 当前按 10 路压测，超出后自动回退 libx264

def _terminate_subprocess(proc: Optional[subprocess.Popen], timeout: float = 3.0):
    if proc is None:
        return

    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except Exception:
        pass

    try:
        if os.name != 'nt':
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except Exception:
            pass

def _nvenc_acquire() -> bool:
    """尝试获取一个 NVENC 会话槽位，成功返回 True"""
    global _nvenc_count
    with _nvenc_lock:
        if _nvenc_count < _NVENC_MAX_SESSIONS:
            _nvenc_count += 1
            return True
        return False

def _nvenc_release():
    """释放一个 NVENC 会话槽位"""
    global _nvenc_count
    with _nvenc_lock:
        _nvenc_count = max(0, _nvenc_count - 1)

# 项目根目录加入路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.algorithm_config import CameraConfig, VideoCodec
from config.config_manager import ConfigManager
from nan.logger_config import setup_logging
from stream.enhanced_video_processor import (
    EnhancedVideoStreamProcessor,
    VideoStreamConfig,
    VideoStreamStatus,
)
from stream.capture_process import CaptureProxy
from inference.unified_scheduler import UnifiedInferenceScheduler
from alert.alert_system import AlertSystem, create_count_threshold_rule, AlertLevel
from tracking.simple_tracker import SimpleTracker
from utils.alarm_level import classify_point_alarm_level_uv_details

# 保护区界线投影模块：正式部署版放在 utils/zone_projector.py。
try:
    from utils.zone_projector import CameraProjector as BoundaryCameraProjector
except Exception:
    # 兼容临时调试：如果文件被放在项目根目录，也尽量导入。
    try:
        from zone_projector import CameraProjector as BoundaryCameraProjector
    except Exception:
        BoundaryCameraProjector = None

# 实时 PTZ -> camera_projector_config.json 同步线程。
try:
    from utils.camera_ptz_config_updater import start_realtime_ptz_config_updater
except Exception:
    try:
        from camera_ptz_config_updater import start_realtime_ptz_config_updater
    except Exception:
        start_realtime_ptz_config_updater = None


def _enable_fault_logging(log_dir: str = 'log'):
    try:
        os.makedirs(log_dir, exist_ok=True)
        crash_log = os.path.join(log_dir, 'fatal_trace.log')
        fault_file = open(crash_log, 'a', encoding='utf-8')
        faulthandler.enable(file=fault_file, all_threads=True)
        logging.info(f"faulthandler 已启用: {crash_log}")
        return fault_file
    except Exception as e:
        logging.error(f"启用 faulthandler 失败: {e}")
        return None


class StreamProcessor:
    """单路输入流：拉流 → 推理 → 绘框 → 输出推流 → 告警"""

    def __init__(self, stream_cfg: dict, config: CameraConfig, inference_scheduler):
        self.stream_cfg = stream_cfg
        self.config = config
        self.inference_scheduler = inference_scheduler
        self.inference_engine = inference_scheduler
        self._owns_inference_engine = False

        self.name = stream_cfg.get('name', 'unknown')
        self.input_url = stream_cfg.get('input_url') or stream_cfg.get('rtsp_url') or stream_cfg.get('rtmp_url', '')
        self.output_url = (
            stream_cfg.get('output_url')
            or stream_cfg.get('output_rtsp')
            or stream_cfg.get('output_rtmp')
            or ''
        ) if getattr(config, 'push_enabled', True) else ''
        self.stream_tracking_key = stream_cfg.get('stream_id') or self.name or self.input_url

        self.is_running = False
        self._stop_event = threading.Event()

        self.alert_system = AlertSystem(config)
        result_path = os.path.join(getattr(config, 'output_directory', './res'), self.name)
        self.alert_system.initialize_alert_handler(stream_cfg, result_path)
        self._setup_alert_rules()

        self.pipe: Optional[subprocess.Popen] = None
        self._detected_resolution: Optional[tuple] = None
        self._push_ffmpeg_resolution: Optional[tuple] = None
        self._push_reset_needed = False
        self._stream_codec: Optional[str] = None  # 本流实际使用的编码器
        self._using_nvenc = False  # 本流是否占用了 NVENC 槽位
        self._ffmpeg_restart_backoff_initial = max(0.1, float(getattr(self.config, 'push_restart_backoff_initial', 0.3) or 0.3))
        self._ffmpeg_restart_backoff_max = max(
            self._ffmpeg_restart_backoff_initial,
            float(getattr(self.config, 'push_restart_backoff_max', 10.0) or 10.0),
        )
        self._ffmpeg_restart_backoff = self._ffmpeg_restart_backoff_initial  # FFmpeg重启退避时间

        self.video_processor = None
        self.capture_proxy: Optional[CaptureProxy] = None
        self._processor_thread: Optional[threading.Thread] = None
        self._push_thread: Optional[threading.Thread] = None
        self._capture_watchdog_thread: Optional[threading.Thread] = None
        self._capture_restart_lock = threading.Lock()

        self._frame_id = 0
        self._last_infer_frame_id = -1
        self._last_applied_result_frame_id = -1
        self._pending_infer = False
        self._crash_trace_enabled = bool(getattr(self.config, 'crash_trace_enabled', False))
        self._last_detection_overlays = []
        self._last_tracking_summary = {
            'track_count': 0,
            'track_ids': [],
            'classes': [],
        }
        self._logged_active_track_ids = set()
        self._latest_input_frame: Optional[np.ndarray] = None
        self._latest_rendered_frame: Optional[np.ndarray] = None
        self._latest_frame_lock = threading.Lock()
        self._latest_render_lock = threading.Lock()
        self._render_queue = deque(maxlen=max(16, int(getattr(self.config, 'result_max_back_frames', 30) or 30)))
        self._render_queue_lock = threading.Lock()
        self._frame_buffer = deque(maxlen=max(8, int(getattr(self.config, 'result_max_back_frames', 30) or 30)))
        self._frame_ready_event = threading.Event()
        self._last_capture_status = 'init'
        self._last_frame_ts = 0.0
        self._last_processed_ts = 0.0
        self._last_push_ts = 0.0
        self._capture_session_started_ts = 0.0
        self._last_capture_restart_ts = 0.0
        self._last_infer_result_ts = 0.0
        self._last_infer_result_count = 0
        self._last_target_seen_ts = 0.0
        # 2026-06-16 16:49 修改目的：缓存全局类别名称过滤规则，当前用于过滤 guanche。
        self._detection_filtered_class_names = self._normalize_filtered_class_names(
            getattr(self.config, 'detection_filtered_class_names', [])
        )
        self._last_motion_level = 0.0
        self._motion_prev_small: Optional[np.ndarray] = None
        self._capture_watchdog_interval = max(
            1.0,
            float(getattr(self.config, 'capture_watchdog_interval', 5.0) or 5.0),
        )
        self._capture_stall_timeout = max(
            self._capture_watchdog_interval * 2.0,
            float(getattr(self.config, 'capture_stall_timeout', 15.0) or 15.0),
        )
        self._capture_start_timeout = max(
            self._capture_stall_timeout,
            float(getattr(self.config, 'capture_start_timeout', 20.0) or 20.0),
        )
        self._capture_restart_cooldown = max(
            self._capture_watchdog_interval,
            float(getattr(self.config, 'capture_restart_cooldown', 10.0) or 10.0),
        )
        self._tracker = SimpleTracker(
            max_missed=max(5, int(getattr(self.config, 'push_fps', self.config.fps) * 2)),
            min_iou=float(getattr(self.config, 'tracking_match_iou', 0.3) or 0.3),
        ) if bool(getattr(self.config, 'tracking_enabled', False)) else None

        # 保护区界线投影：每路流独立配置、独立 CameraProjector 实例。
        # 推荐在 streams[].camera_id / streams[].stream_id 中显式写稳定ID，
        # 然后在 config/camera_projector_config.json 中按该ID配置 info 与 border_json。
        self.camera_identity = self._resolve_camera_identity()
        self.boundary_projector_enabled = False
        self.boundary_projector = None
        self.boundary_projector_info = None
        self.boundary_projector_border_json = None
        self.boundary_projector_ground_alt = 0.0
        self.boundary_projector_line_thickness = 8
        self._last_boundary_projector_error_ts = 0.0
        self.boundary_projector_cache_overlay = True
        self._boundary_overlay_cache_key = None
        self._boundary_overlay_cache_img = None
        self._boundary_overlay_cache_mask = None
        # 外部 camera_projector_config.json 运行时重载感知。
        # 配合实时 PTZ 同步线程：同步线程负责写 JSON；每路 StreamProcessor 负责每隔约1秒感知 JSON 变化。
        self._boundary_projector_config_path = None
        self._boundary_projector_config_sig = None
        self._last_boundary_projector_config_check_ts = 0.0
        self._last_boundary_projector_reload_error_ts = 0.0

        # 独立运行时配置：用户手动修改，负责画线样式参数、PTZ映射与yaw偏移量。
        # 与 camera_projector_config.json 分开存储，避免被实时PTZ线程覆盖。
        self._boundary_projector_runtime_config_path = None
        self._boundary_projector_runtime_config_sig = None
        self._boundary_projector_draw_options = {}
        self.boundary_projector_matched_key = None
        self.boundary_projector_camera_name = None
        self.boundary_projector_aliases = []
        self._last_boundary_projector_runtime_config_check_ts = 0.0
        self._last_boundary_projector_runtime_config_error_ts = 0.0

        self._boundary_overlay_cache_lock = threading.Lock()
        self._init_boundary_projector()

        logging.info(f"[{self.name}] StreamProcessor 初始化完成, stream_key={self.stream_tracking_key}, camera_identity={self.camera_identity}")

    def _setup_alert_rules(self):
        cooldown = float(getattr(self.config, 'alarm_interval_seconds', 10.0))
        threshold = float(getattr(self.config, 'alarm_target_threshold', 1))
        rule = create_count_threshold_rule(
            rule_id="alarm_any_detection",
            threshold=threshold,
            description="检测到目标",
            level=AlertLevel.MEDIUM,
            cooldown=cooldown,
        )
        self.alert_system.add_rule(rule)
    def _build_capture_options(self) -> str:
        return self.config.get_capture_options(self.input_url)

    @staticmethod
    def _extract_camera_id_from_url(url_value) -> Optional[str]:
        """从 rtmp/rtsp URL 中提取 camera-1002 这类稳定ID。"""
        if url_value is None:
            return None
        s = str(url_value).strip()
        if not s:
            return None
        m = re.search(r'(camera[-_]\d+)', s, flags=re.IGNORECASE)
        if m:
            return m.group(1).replace('_', '-').lower()
        # 没有 camera-xxxx 时，取 URL 最后一段作为候选，但仅作为兜底。
        try:
            tail = s.rstrip('/').split('/')[-1].strip()
            return tail or None
        except Exception:
            return None

    def _resolve_camera_identity(self) -> str:
        """返回用于匹配保护区投影配置的稳定摄像头/流ID。"""
        parsed_input_id = self._extract_camera_id_from_url(self.input_url)
        parsed_output_id = self._extract_camera_id_from_url(self.output_url)
        candidates = [
            self.stream_cfg.get('camera_id'),
            self.stream_cfg.get('cameraId'),
            self.stream_cfg.get('stream_id'),
            self.stream_cfg.get('streamId'),
            parsed_input_id,
            parsed_output_id,
            self.stream_cfg.get('monitorEq'),
            self.stream_cfg.get('device_id'),
            self.stream_cfg.get('deviceId'),
            self.stream_cfg.get('taskId'),
            self.name,
            self.input_url,
        ]
        for value in candidates:
            if value is not None and str(value).strip():
                return str(value).strip()
        return 'unknown'

    def _projector_identity_candidates(self):
        """按优先级返回可用于外部JSON匹配的候选键。"""
        parsed_input_id = self._extract_camera_id_from_url(self.input_url)
        parsed_output_id = self._extract_camera_id_from_url(self.output_url)
        raw = [
            self.stream_cfg.get('camera_id'),
            self.stream_cfg.get('cameraId'),
            self.stream_cfg.get('stream_id'),
            self.stream_cfg.get('streamId'),
            parsed_input_id,
            parsed_output_id,
            self.stream_cfg.get('monitorEq'),
            self.stream_cfg.get('device_id'),
            self.stream_cfg.get('deviceId'),
            self.stream_cfg.get('taskId'),
            self.name,
            self.input_url,
            self.output_url,
            self.stream_tracking_key,
        ]
        out = []
        seen = set()
        for value in raw:
            if value is None:
                continue
            s = str(value).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _resolve_projector_path(self, path_value):
        if path_value is None:
            return None
        p = str(path_value).strip()
        if not p:
            return None
        if os.path.isabs(p):
            return p
        candidates = [
            project_root / p,
            project_root / 'config' / p,
            project_root / 'config' / 'borders' / p,
            project_root / 'projector' / p,
            project_root / 'utils' / p,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        # 找不到也返回 project_root 下的路径，让后续 FileNotFoundError 的日志更直观。
        return str(project_root / p)

    @staticmethod
    def _projector_config_file_sig(path_value):
        try:
            p = Path(str(path_value))
            st = p.stat()
            return str(p.resolve()), int(st.st_mtime_ns), int(st.st_size)
        except Exception:
            return str(path_value), None, None

    def _find_projector_external_config_path(self):
        """查找独立摄像头投影配置文件路径。"""
        path_candidates = [
            self.stream_cfg.get('camera_projector_config_path'),
            self.stream_cfg.get('projector_config_path'),
            getattr(self.config, 'camera_projector_config_path', None),
            os.getenv('CAMERA_PROJECTOR_CONFIG'),
            project_root / 'config' / 'camera_projector_config.json',
            project_root / 'camera_projector_config.json',
        ]
        for p in path_candidates:
            if not p:
                continue
            p = Path(str(p))
            if not p.is_absolute():
                p = project_root / p
            if p.exists():
                return p
        return None

    def _load_projector_external_config(self):
        """读取独立的摄像头投影配置文件。不存在则返回空字典。"""
        p = self._find_projector_external_config_path()
        if not p:
            return {}

        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._boundary_projector_config_path = str(p)
            self._boundary_projector_config_sig = self._projector_config_file_sig(p)

            logging.info(f"[{self.name}] 已读取摄像头投影配置: {p}")
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logging.error(f"[{self.name}] 读取摄像头投影配置失败: {p}, err={e}")
            return {}

    def _find_projector_runtime_config_path(self):
        """查找独立运行时配置文件：画线样式 + PTZ映射。"""
        path_candidates = [
            self.stream_cfg.get('camera_projector_runtime_config_path'),
            self.stream_cfg.get('projector_runtime_config_path'),
            getattr(self.config, 'camera_projector_runtime_config_path', None),
            os.getenv('CAMERA_PROJECTOR_RUNTIME_CONFIG'),
            project_root / 'config' / 'camera_projector_runtime_config.json',
            project_root / 'config' / 'camera_projector_runtime_config_byGPT.json',
            project_root / 'camera_projector_runtime_config.json',
        ]
        for p in path_candidates:
            if not p:
                continue
            p = Path(str(p))
            if not p.is_absolute():
                p = project_root / p
            if p.exists():
                return p
        return None

    @staticmethod
    def _safe_float_runtime(value, default=None, allow_none: bool = True):
        if value is None:
            return None if allow_none else default
        if isinstance(value, str) and value.strip().lower() in ('', 'none', 'null'):
            return None if allow_none else default
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _safe_int_runtime(value, default=None):
        try:
            return int(round(float(value)))
        except Exception:
            return default

    @staticmethod
    def _safe_bool_runtime(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        s = str(value).strip().lower()
        if s in ('1', 'true', 'yes', 'y', 'on', 'enable', 'enabled'):
            return True
        if s in ('0', 'false', 'no', 'n', 'off', 'disable', 'disabled'):
            return False
        return bool(default)

    def _extract_boundary_projector_draw_options(self, runtime_cfg: dict) -> dict:
        """从运行时配置中提取当前摄像头专属的画线参数。

        支持两种配置结构：
        1. 新结构（推荐）：
           {
             "cameras": {
               "camera-1001": {"camera_name": "罗家集", "aliases": [...], "drawing": {...}},
               ...
             }
           }
        2. 旧结构（兼容）：
           {"drawing": {...}}

        新结构下会优先按 camera_projector_config.json 匹配到的 key、stream 中的 camera_id、
        URL 中的 camera-xxxx、camera_name、aliases 等信息匹配对应摄像头。
        """
        if not isinstance(runtime_cfg, dict):
            runtime_cfg = {}

        drawing = self._match_runtime_camera_drawing(runtime_cfg)
        if not isinstance(drawing, dict):
            drawing = {}

        # 支持 lower_snake_case，也兼容本地测试脚本里的 UPPER_CASE 名称。
        def pick(*names, default=None):
            for name in names:
                if name in drawing:
                    return drawing.get(name)
                up = str(name).upper()
                if up in drawing:
                    return drawing.get(up)
            return default

        opts = {
            'fixed_dash_len_px': self._safe_float_runtime(pick('fixed_dash_len_px', default=10.0), 10.0, allow_none=False),
            'fixed_gap_len_px': self._safe_float_runtime(pick('fixed_gap_len_px', default=25.0), 25.0, allow_none=False),
            'json_thickness_enable': self._safe_bool_runtime(pick('json_thickness_enable', default=False), False),
            'line_thickness_inner': self._safe_int_runtime(pick('line_thickness_inner', default=4), 4),
            'line_thickness_middle': self._safe_int_runtime(pick('line_thickness_middle', default=4), 4),
            'line_thickness_outer': self._safe_int_runtime(pick('line_thickness_outer', default=4), 4),
            'line_brightness_inner': self._safe_float_runtime(pick('line_brightness_inner', default=1.0), 1.0, allow_none=False),
            'line_brightness_middle': self._safe_float_runtime(pick('line_brightness_middle', default=1.0), 1.0, allow_none=False),
            'line_brightness_outer': self._safe_float_runtime(pick('line_brightness_outer', default=1.0), 1.0, allow_none=False),
            'drop_max_drop_px': self._safe_float_runtime(pick('drop_max_drop_px', default=0.0), 0.0, allow_none=False),
            'drop_max_step_px': self._safe_float_runtime(pick('drop_max_step_px', default=55.0), 55.0, allow_none=False),
            'drop_near_distance': self._safe_float_runtime(pick('drop_near_distance', default=None), None, allow_none=True),
            'drop_far_distance': self._safe_float_runtime(pick('drop_far_distance', default=None), None, allow_none=True),
            'thickness_near_factor': self._safe_float_runtime(pick('thickness_near_factor', default=1.0), 1.0, allow_none=False),
            'thickness_far_factor': self._safe_float_runtime(pick('thickness_far_factor', default=1.0), 1.0, allow_none=False),
            'thickness_max_step_px': self._safe_float_runtime(pick('thickness_max_step_px', default=55.0), 55.0, allow_none=False),
            'thickness_near_distance': self._safe_float_runtime(pick('thickness_near_distance', default=None), None, allow_none=True),
            'thickness_far_distance': self._safe_float_runtime(pick('thickness_far_distance', default=None), None, allow_none=True),
            'edge_margin_ratio': self._safe_float_runtime(pick('edge_margin_ratio', default=0.0), 0.0, allow_none=False),
        }

        # 基本保护：亮度限定到 0~1；线宽、虚线长度、步长等给出合理下限。
        for k in ('line_brightness_inner', 'line_brightness_middle', 'line_brightness_outer'):
            try:
                opts[k] = max(0.0, min(1.0, float(opts[k])))
            except Exception:
                opts[k] = 1.0
        for k in ('line_thickness_inner', 'line_thickness_middle', 'line_thickness_outer'):
            opts[k] = max(1, int(opts[k] or 1))
        for k in ('fixed_dash_len_px', 'fixed_gap_len_px', 'drop_max_step_px', 'thickness_max_step_px'):
            opts[k] = max(1.0, float(opts[k] or 1.0))
        opts['edge_margin_ratio'] = max(0.0, min(0.30, float(opts['edge_margin_ratio'])))
        return opts

    def _runtime_camera_identity_candidates(self):
        """运行时画线配置按摄像头匹配时使用的候选键。"""
        raw = []
        raw.extend(self._projector_identity_candidates())
        raw.extend([
            self.camera_identity,
            self.boundary_projector_matched_key,
            self.boundary_projector_camera_name,
        ])
        if isinstance(self.boundary_projector_aliases, (list, tuple)):
            raw.extend(self.boundary_projector_aliases)
        out = []
        seen = set()
        for value in raw:
            if value is None:
                continue
            s = str(value).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _match_runtime_camera_drawing(self, runtime_cfg: dict) -> dict:
        """从运行时配置中匹配当前摄像头的 drawing。"""
        if not isinstance(runtime_cfg, dict):
            return {}

        # 兼容旧版全局 drawing；如果用户还没迁移配置，不影响运行。
        global_drawing = (
            runtime_cfg.get('drawing')
            or runtime_cfg.get('boundary_projector_drawing')
            or runtime_cfg.get('line_drawing')
        )

        table = None
        for key in ('cameras', 'camera_drawings', 'drawings_by_camera'):
            if isinstance(runtime_cfg.get(key), dict):
                table = runtime_cfg.get(key)
                break

        if not isinstance(table, dict):
            return global_drawing if isinstance(global_drawing, dict) else {}

        candidates = self._runtime_camera_identity_candidates()

        # 1) 直接用 camera-xxxx / 中文名称等候选键匹配。
        for candidate in candidates:
            item = table.get(candidate)
            if isinstance(item, dict):
                d = item.get('drawing') if isinstance(item.get('drawing'), dict) else item
                return d if isinstance(d, dict) else {}

        # 2) 通过每个摄像头配置中的 camera_name / aliases / input_url / output_url 反向匹配。
        candidates_set = set(candidates)
        for key, item in table.items():
            if not isinstance(item, dict):
                continue
            aliases = [str(key).strip()]
            for alias_key in ('aliases', 'alias', 'camera_name', 'name', 'camera_id', 'cameraId', 'stream_id', 'streamId', 'input_url', 'output_url'):
                alias_val = item.get(alias_key)
                if isinstance(alias_val, (list, tuple)):
                    aliases.extend([str(x).strip() for x in alias_val if str(x).strip()])
                elif alias_val is not None and str(alias_val).strip():
                    aliases.append(str(alias_val).strip())
            for url_key in ('input_url', 'output_url'):
                parsed = self._extract_camera_id_from_url(item.get(url_key))
                if parsed:
                    aliases.append(parsed)

            if candidates_set.intersection(set(aliases)):
                d = item.get('drawing') if isinstance(item.get('drawing'), dict) else item
                return d if isinstance(d, dict) else {}

        return global_drawing if isinstance(global_drawing, dict) else {}

    @staticmethod
    def _boundary_projector_draw_options_signature(opts: dict):
        if not isinstance(opts, dict):
            return ()
        out = []
        for k in sorted(opts.keys()):
            v = opts.get(k)
            if isinstance(v, bool):
                out.append((k, bool(v)))
            elif v is None:
                out.append((k, None))
            else:
                try:
                    out.append((k, round(float(v), 8)))
                except Exception:
                    out.append((k, str(v)))
        return tuple(out)

    def _reload_boundary_projector_runtime_config_if_changed(self, force: bool = False) -> bool:
        """每隔约1秒检查运行时配置是否变化；变化后刷新画线参数并清空 overlay 缓存。"""
        now = time.time()
        if not force and (now - self._last_boundary_projector_runtime_config_check_ts) < 1.0:
            return False
        self._last_boundary_projector_runtime_config_check_ts = now

        runtime_path = self._find_projector_runtime_config_path()
        if not runtime_path:
            # 没有独立运行时配置时，使用 zone_projector 的默认参数，保持兼容。
            if force and not self._boundary_projector_draw_options:
                self._boundary_projector_draw_options = self._extract_boundary_projector_draw_options({})
            return False

        file_sig = self._projector_config_file_sig(runtime_path)
        if not force and file_sig == self._boundary_projector_runtime_config_sig:
            return False

        try:
            with open(runtime_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
            old_sig = self._boundary_projector_draw_options_signature(self._boundary_projector_draw_options)
            self._boundary_projector_runtime_config_path = str(runtime_path)
            self._boundary_projector_runtime_config_sig = file_sig
            self._boundary_projector_draw_options = self._extract_boundary_projector_draw_options(data)
            new_sig = self._boundary_projector_draw_options_signature(self._boundary_projector_draw_options)
            if force or new_sig != old_sig:
                self._clear_boundary_overlay_cache()
                logging.info(
                    f"[{self.name}] 保护区画线运行时配置已刷新: {runtime_path}, "
                    f"draw_options={self._boundary_projector_draw_options}"
                )
                return True
            return False
        except Exception as e:
            now = time.time()
            if now - self._last_boundary_projector_runtime_config_error_ts >= 5.0:
                logging.error(f"[{self.name}] 读取保护区画线运行时配置失败: {runtime_path}, err={e}")
                self._last_boundary_projector_runtime_config_error_ts = now
            return False

    def _match_projector_config(self) -> dict:
        """
        合并投影配置。优先级：
        1. 外部 config/camera_projector_config.json 的 default
        2. 外部 cameras/streams/projectors 中按 camera_id/stream_id/name/input_url 匹配的配置
        3. 当前 streams[] 内联 projector/camera_projector/boundary_projector 配置
        """
        merged = {}
        external = self._load_projector_external_config()
        if isinstance(external, dict):
            default_cfg = external.get('default') or external.get('defaults') or {}
            if isinstance(default_cfg, dict):
                merged.update(default_cfg)

            table = None
            for key in ('cameras', 'streams', 'projectors', 'camera_projectors'):
                if isinstance(external.get(key), dict):
                    table = external.get(key)
                    break
            if table is None and any(k in external for k in self._projector_identity_candidates()):
                table = external
            if isinstance(table, dict):
                matched_directly = False
                for candidate in self._projector_identity_candidates():
                    matched = table.get(candidate)
                    if isinstance(matched, dict):
                        merged.update(matched)
                        merged['_matched_key'] = candidate
                        matched_directly = True
                        break

                # 允许配置项通过 aliases/camera_id/name/input_url 等字段反向匹配，
                # 方便 camera_projector_config.json 使用 camera-1002 作主键，同时兼容中文名称。
                if not matched_directly:
                    candidates_set = set(self._projector_identity_candidates())
                    for key, value in table.items():
                        if not isinstance(value, dict):
                            continue
                        aliases = []
                        for alias_key in ('aliases', 'alias', 'camera_id', 'cameraId', 'name', 'monitorEq', 'taskId', 'input_url', 'output_url'):
                            alias_val = value.get(alias_key)
                            if isinstance(alias_val, (list, tuple)):
                                aliases.extend([str(x).strip() for x in alias_val if str(x).strip()])
                            elif alias_val is not None and str(alias_val).strip():
                                aliases.append(str(alias_val).strip())
                        aliases.append(str(key).strip())
                        # URL 里的 camera-xxxx 也纳入别名匹配。
                        for url_key in ('input_url', 'output_url'):
                            parsed = self._extract_camera_id_from_url(value.get(url_key))
                            if parsed:
                                aliases.append(parsed)
                        if candidates_set.intersection(set(aliases)):
                            merged.update(value)
                            merged['_matched_key'] = str(key)
                            break

        inline = (
            self.stream_cfg.get('projector')
            or self.stream_cfg.get('camera_projector')
            or self.stream_cfg.get('boundary_projector')
            or self.stream_cfg.get('boundary_lines')
        )
        if isinstance(inline, dict):
            merged.update(inline)

        # 兼容把 border_json/info 直接写在 streams[] 顶层的情况。
        top_level_keys = (
            'border_json', 'border_json_path', 'ground_alt', 'line_thickness',
            'info', 'camera_info', 'projector_enabled', 'boundary_projector_enabled'
        )
        for key in top_level_keys:
            if key in self.stream_cfg and key not in merged:
                merged[key] = self.stream_cfg[key]

        return merged

    def _coerce_projector_info(self, cfg: dict):
        info = cfg.get('info') or cfg.get('camera_info') or cfg.get('cameraInfo')
        if not isinstance(info, dict):
            info = {}
            for key in ('latitude', 'longitude', 'height', 'gimbal_yaw', 'gimbal_pitch', 'gimbal_roll', 'zoom_factor'):
                if key in cfg:
                    info[key] = cfg[key]
        required = ('latitude', 'longitude', 'height', 'gimbal_yaw', 'gimbal_pitch', 'gimbal_roll', 'zoom_factor')
        missing = [k for k in required if k not in info or info.get(k) in (None, '')]
        if missing:
            raise ValueError(f"投影 info 缺少字段: {missing}")
        # 提前转成 float，避免每帧在 CameraProjector 内因字符串异常报错。
        return {k: float(info[k]) for k in required}

    def _init_boundary_projector(self):
        cfg = self._match_projector_config()
        has_any_cfg = bool(cfg)
        enabled = bool(cfg.get('enabled', cfg.get('projector_enabled', cfg.get('boundary_projector_enabled', has_any_cfg))))
        if not enabled:
            return
        if BoundaryCameraProjector is None:
            logging.error(f"[{self.name}] 已配置保护区投影，但无法导入 utils.zone_projector.CameraProjector")
            return
        try:
            border_json = cfg.get('border_json') or cfg.get('border_json_path') or cfg.get('borderJson')
            border_json = self._resolve_projector_path(border_json)
            if not border_json:
                raise ValueError('缺少 border_json / border_json_path')
            info = self._coerce_projector_info(cfg)
            ground_alt = float(cfg.get('ground_alt', cfg.get('groundAlt', 0.0)) or 0.0)
            line_thickness = int(cfg.get('line_thickness', cfg.get('lineThickness', 8)) or 8)
            cache_overlay = bool(cfg.get('cache_overlay', cfg.get('cacheOverlay', True)))
            self.boundary_projector_matched_key = cfg.get('_matched_key')
            self.boundary_projector_camera_name = cfg.get('camera_name') or cfg.get('name')
            aliases = cfg.get('aliases') or cfg.get('alias') or []
            if isinstance(aliases, (list, tuple)):
                self.boundary_projector_aliases = [str(x).strip() for x in aliases if str(x).strip()]
            elif aliases is not None and str(aliases).strip():
                self.boundary_projector_aliases = [str(aliases).strip()]
            else:
                self.boundary_projector_aliases = []

            projector_kwargs = cfg.get('projector_kwargs') or cfg.get('projectorKwargs') or {}
            if not isinstance(projector_kwargs, dict):
                projector_kwargs = {}
            self.boundary_projector = BoundaryCameraProjector(**projector_kwargs)
            self.boundary_projector_info = info
            self.boundary_projector_border_json = border_json
            self.boundary_projector_ground_alt = ground_alt
            self.boundary_projector_line_thickness = max(1, line_thickness)
            self.boundary_projector_cache_overlay = cache_overlay
            self.boundary_projector_enabled = True
            self._reload_boundary_projector_runtime_config_if_changed(force=True)

            # 如果 API 支持缓存，启动时预热一次 JSON，避免第一帧卡顿。
            try:
                if hasattr(self.boundary_projector, '_load_curves_cached'):
                    self.boundary_projector._load_curves_cached(border_json)
            except Exception as e:
                logging.warning(f"[{self.name}] 预加载保护区界线JSON失败，后续绘制时会再次尝试: {e}")

            logging.info(
                f"[{self.name}] 保护区界线投影已启用: camera_identity={self.camera_identity}, "
                f"matched_key={cfg.get('_matched_key', '(inline/default)')}, border_json={border_json}, "
                f"ground_alt={ground_alt}, line_thickness={self.boundary_projector_line_thickness}"
            )
        except Exception as e:
            self.boundary_projector_enabled = False
            self.boundary_projector = None
            logging.error(f"[{self.name}] 初始化保护区界线投影失败: {e}")

    def _clear_boundary_overlay_cache(self):
        """清空保护区界线 overlay 缓存，下一帧会重新投影。"""
        try:
            with self._boundary_overlay_cache_lock:
                self._boundary_overlay_cache_key = None
                self._boundary_overlay_cache_img = None
                self._boundary_overlay_cache_mask = None
        except Exception:
            pass

    @staticmethod
    def _projector_info_signature(info: dict):
        if not isinstance(info, dict):
            return ()
        out = []
        for k in sorted(info.keys()):
            try:
                out.append((k, round(float(info[k]), 8)))
            except Exception:
                out.append((k, str(info[k])))
        return tuple(out)

    def _reload_boundary_projector_config_if_changed(self, force: bool = False) -> bool:
        """
        每隔约1秒检查 camera_projector_config.json 是否变化。

        如果变化，则重新匹配当前流对应的投影配置，并更新：
        - boundary_projector_info
        - border_json
        - ground_alt
        - line_thickness
        - cache_overlay

        只要上述内容有变化，就清空 overlay 缓存。
        """
        now = time.time()
        if not force and (now - self._last_boundary_projector_config_check_ts) < 1.0:
            return False
        self._last_boundary_projector_config_check_ts = now

        config_path = self._find_projector_external_config_path()
        if not config_path:
            return False

        file_sig = self._projector_config_file_sig(config_path)
        if not force and file_sig == self._boundary_projector_config_sig:
            return False

        try:
            cfg = self._match_projector_config()
            has_any_cfg = bool(cfg)
            enabled = bool(
                cfg.get('enabled', cfg.get('projector_enabled', cfg.get('boundary_projector_enabled', has_any_cfg))))

            if not enabled:
                if self.boundary_projector_enabled:
                    self.boundary_projector_enabled = False
                    self._clear_boundary_overlay_cache()
                    logging.info(f"[{self.name}] 保护区界线投影配置已变为禁用，已关闭本路投影")
                self._boundary_projector_config_sig = file_sig
                return True

            if BoundaryCameraProjector is None:
                return False

            # 如果原本未启用，则直接走初始化流程。
            if self.boundary_projector is None or not self.boundary_projector_enabled:
                self._init_boundary_projector()
                self._boundary_projector_config_sig = self._projector_config_file_sig(config_path)
                return True

            old_runtime_sig = (
                str(self.boundary_projector_border_json),
                self._projector_info_signature(self.boundary_projector_info),
                round(float(self.boundary_projector_ground_alt), 8),
                int(self.boundary_projector_line_thickness),
                bool(self.boundary_projector_cache_overlay),
            )

            border_json = cfg.get('border_json') or cfg.get('border_json_path') or cfg.get('borderJson')
            border_json = self._resolve_projector_path(border_json)
            if not border_json:
                raise ValueError('缺少 border_json / border_json_path')

            info = self._coerce_projector_info(cfg)
            ground_alt = float(cfg.get('ground_alt', cfg.get('groundAlt', 0.0)) or 0.0)
            line_thickness = int(cfg.get('line_thickness', cfg.get('lineThickness', 8)) or 8)
            cache_overlay = bool(cfg.get('cache_overlay', cfg.get('cacheOverlay', True)))
            self.boundary_projector_matched_key = cfg.get('_matched_key')
            self.boundary_projector_camera_name = cfg.get('camera_name') or cfg.get('name')
            aliases = cfg.get('aliases') or cfg.get('alias') or []
            if isinstance(aliases, (list, tuple)):
                self.boundary_projector_aliases = [str(x).strip() for x in aliases if str(x).strip()]
            elif aliases is not None and str(aliases).strip():
                self.boundary_projector_aliases = [str(aliases).strip()]
            else:
                self.boundary_projector_aliases = []

            self.boundary_projector_info = info
            self.boundary_projector_border_json = border_json
            self.boundary_projector_ground_alt = ground_alt
            self.boundary_projector_line_thickness = max(1, line_thickness)
            self.boundary_projector_cache_overlay = cache_overlay
            self.boundary_projector_enabled = True
            self._boundary_projector_config_sig = self._projector_config_file_sig(config_path)
            # 当前摄像头匹配信息可能变化；即使运行时配置文件本身未变，也需要重新匹配该摄像头的 drawing。
            self._reload_boundary_projector_runtime_config_if_changed(force=True)

            new_runtime_sig = (
                str(self.boundary_projector_border_json),
                self._projector_info_signature(self.boundary_projector_info),
                round(float(self.boundary_projector_ground_alt), 8),
                int(self.boundary_projector_line_thickness),
                bool(self.boundary_projector_cache_overlay),
            )

            if new_runtime_sig != old_runtime_sig:
                self._clear_boundary_overlay_cache()
                logging.info(
                    f"[{self.name}] 检测到保护区投影配置变化，已刷新本路投影参数并清空缓存: "
                    f"camera_identity={self.camera_identity}, info={self.boundary_projector_info}"
                )
                return True

            return False

        except Exception as e:
            now = time.time()
            if now - self._last_boundary_projector_reload_error_ts >= 5.0:
                logging.error(f"[{self.name}] 重载保护区投影配置失败: {e}")
                self._last_boundary_projector_reload_error_ts = now
            return False

    def _boundary_overlay_key(self, frame: np.ndarray):
        """生成保护区界线缓存键：相机参数、图像尺寸、边界文件修改时间任何一项变化都会失效。"""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        try:
            st = os.stat(self.boundary_projector_border_json)
            file_sig = (os.path.abspath(self.boundary_projector_border_json), int(st.st_mtime_ns), int(st.st_size))
        except Exception:
            file_sig = (str(self.boundary_projector_border_json), None, None)
        info_items = tuple(
            (k, round(float(self.boundary_projector_info[k]), 8))
            for k in sorted(self.boundary_projector_info.keys())
        )
        try:
            projector_sig = (
                round(float(getattr(self.boundary_projector, 'sensor_full_w', 0.0)), 8),
                round(float(getattr(self.boundary_projector, 'sensor_full_h', 0.0)), 8),
                round(float(getattr(self.boundary_projector, 'sensor_crop_w', 0.0)), 8),
                round(float(getattr(self.boundary_projector, 'sensor_crop_h', 0.0)), 8),
            )
        except Exception:
            projector_sig = ()
        draw_options_sig = self._boundary_projector_draw_options_signature(self._boundary_projector_draw_options)
        return (
            self.camera_identity,
            int(w), int(h),
            file_sig,
            info_items,
            round(float(self.boundary_projector_ground_alt), 8),
            int(self.boundary_projector_line_thickness),
            projector_sig,
            self._boundary_projector_runtime_config_sig,
            draw_options_sig,
        )

    @staticmethod
    def _apply_boundary_overlay(frame: np.ndarray, overlay: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if overlay is None or mask is None:
            return frame
        if frame.shape[:2] != overlay.shape[:2] or frame.shape[:2] != mask.shape[:2]:
            return frame
        frame[mask] = overlay[mask]
        return frame

    def _build_boundary_overlay(self, frame: np.ndarray) -> tuple:
        overlay = np.zeros_like(frame, dtype=np.uint8)
        try:
            _, overlay = self.boundary_projector.project_points_from_json(
                self.boundary_projector_border_json,
                self.boundary_projector_info,
                overlay,
                ground_alt=self.boundary_projector_ground_alt,
                line_thickness=self.boundary_projector_line_thickness,
                use_cache=True,
                draw=True,
                **self._boundary_projector_draw_options,
            )
        except TypeError:
            # 兼容旧签名，不过正式部署版 utils.zone_projector 支持 line_thickness/use_cache。
            _, overlay = self.boundary_projector.project_points_from_json(
                self.boundary_projector_border_json,
                self.boundary_projector_info,
                overlay,
                ground_alt=self.boundary_projector_ground_alt,
            )
        mask = np.any(overlay != 0, axis=2)
        return np.ascontiguousarray(overlay, dtype=np.uint8), mask

    def _draw_boundary_projector(self, frame: np.ndarray, fid: Optional[int] = None) -> np.ndarray:
        try:
            self._reload_boundary_projector_config_if_changed()
            self._reload_boundary_projector_runtime_config_if_changed()
        except Exception as e:
            now = time.time()
            if now - self._last_boundary_projector_error_ts >= 5.0:
                logging.error(f"[{self.name}] 检查保护区投影配置变化失败 fid={fid}: {e}")
                self._last_boundary_projector_error_ts = now

        if not self.boundary_projector_enabled or self.boundary_projector is None:
            return frame
        try:
            if not self.boundary_projector_cache_overlay:
                # 关闭 overlay 缓存时，仍直接在当前帧上投影绘制。
                try:
                    _, out = self.boundary_projector.project_points_from_json(
                        self.boundary_projector_border_json,
                        self.boundary_projector_info,
                        frame,
                        ground_alt=self.boundary_projector_ground_alt,
                        line_thickness=self.boundary_projector_line_thickness,
                        use_cache=True,
                        **self._boundary_projector_draw_options,
                    )
                except TypeError:
                    _, out = self.boundary_projector.project_points_from_json(
                        self.boundary_projector_border_json,
                        self.boundary_projector_info,
                        frame,
                        ground_alt=self.boundary_projector_ground_alt,
                    )
                return out

            cache_key = self._boundary_overlay_key(frame)
            with self._boundary_overlay_cache_lock:
                cache_valid = (
                    cache_key is not None
                    and cache_key == self._boundary_overlay_cache_key
                    and self._boundary_overlay_cache_img is not None
                    and self._boundary_overlay_cache_mask is not None
                )
                if not cache_valid:
                    overlay, mask = self._build_boundary_overlay(frame)
                    self._boundary_overlay_cache_key = cache_key
                    self._boundary_overlay_cache_img = overlay
                    self._boundary_overlay_cache_mask = mask
                    if np.any(mask):
                        logging.info(f"[{self.name}] 保护区界线投影缓存已更新 fid={fid}, camera_identity={self.camera_identity}")
                    else:
                        logging.info(f"[{self.name}] 保护区界线投影缓存已更新但当前画面无可见线段 fid={fid}, camera_identity={self.camera_identity}")
                overlay = self._boundary_overlay_cache_img
                mask = self._boundary_overlay_cache_mask
            return self._apply_boundary_overlay(frame, overlay, mask)
        except Exception as e:
            now = time.time()
            if now - self._last_boundary_projector_error_ts >= 5.0:
                logging.error(f"[{self.name}] 绘制保护区界线失败 fid={fid}: {e}")
                self._last_boundary_projector_error_ts = now
            return frame


    def _project_boundary_curves_for_frame(self, frame: np.ndarray):
        if (
            frame is None
            or not self.boundary_projector_enabled
            or self.boundary_projector is None
            or not self.boundary_projector_border_json
            or not self.boundary_projector_info
        ):
            return None
        canvas = np.zeros_like(frame, dtype=np.uint8)
        try:
            projected_curves, _ = self.boundary_projector.project_points_from_json(
                self.boundary_projector_border_json,
                self.boundary_projector_info,
                canvas,
                ground_alt=self.boundary_projector_ground_alt,
                line_thickness=self.boundary_projector_line_thickness,
                use_cache=True,
                draw=False,
                **self._boundary_projector_draw_options,
            )
            return projected_curves
        except TypeError:
            try:
                projected_curves, _ = self.boundary_projector.project_points_from_json(
                    self.boundary_projector_border_json,
                    self.boundary_projector_info,
                    canvas,
                    ground_alt=self.boundary_projector_ground_alt,
                )
                return projected_curves
            except Exception:
                return None
        except Exception:
            return None

    def _on_capture_status(self, stream_id: str, status: str):
        """拉流子进程状态回调"""
        status_map = {
            'connecting': VideoStreamStatus.CONNECTING,
            'connected': VideoStreamStatus.CONNECTED,
            'interrupted': VideoStreamStatus.INTERRUPTED,
            'reconnecting': VideoStreamStatus.RECONNECTING,
            'error': VideoStreamStatus.ERROR,
        }
        vs = status_map.get(status)
        if vs:
            self._on_status_change(stream_id, vs)

    def _reset_stream_runtime_state(self, clear_timestamps: bool = False):
        self.inference_scheduler.reset_stream_tracking(self.stream_tracking_key)
        self._last_detection_overlays = []
        self._last_tracking_summary = {
            'track_count': 0,
            'track_ids': [],
            'classes': [],
        }
        self._logged_active_track_ids.clear()
        self._last_infer_frame_id = -1
        self._last_applied_result_frame_id = -1
        self._pending_infer = False
        if self._tracker is not None:
            self._tracker.reset()
        with self._latest_frame_lock:
            self._latest_input_frame = None
        with self._latest_render_lock:
            self._latest_rendered_frame = None
        with self._render_queue_lock:
            self._render_queue.clear()
        self._frame_buffer.clear()
        self._frame_ready_event.clear()
        self._push_reset_needed = True
        if clear_timestamps:
            self._last_frame_ts = 0.0
            self._last_processed_ts = 0.0
            self._last_push_ts = 0.0
            self._last_infer_result_ts = 0.0
            self._last_infer_result_count = 0
            self._last_target_seen_ts = 0.0
            self._last_motion_level = 0.0
            self._motion_prev_small = None

    def _build_capture_proxy(self) -> CaptureProxy:
        capture_options = self._build_capture_options()
        logging.info(f"[{self.name}] 拉流参数: {capture_options}")
        return CaptureProxy(
            stream_id=f"{self.name}_{int(time.time())}",
            stream_url=self.input_url,
            pull_device=self.config.get_pull_device(),
            capture_options=capture_options,
            frame_width=self._detected_resolution[0],
            frame_height=self._detected_resolution[1],
            frame_callback=self._on_frame,
            status_callback=self._on_capture_status,
        )

    def _start_capture_proxy(self):
        self._capture_session_started_ts = time.time()
        proxy = self._build_capture_proxy()
        proxy.start()
        self.capture_proxy = proxy

    def _restart_capture_proxy(self, reason: str) -> bool:
        if not self.is_running or self._stop_event.is_set():
            return False

        old_proxy = None
        with self._capture_restart_lock:
            now = time.time()
            if (now - self._last_capture_restart_ts) < self._capture_restart_cooldown:
                return False
            self._last_capture_restart_ts = now
            logging.warning(
                f"[{self.name}] 检测到拉流停滞，重建CaptureProxy: "
                f"reason={reason}, snapshot={self._build_pipeline_snapshot()}"
            )
            old_proxy = self.capture_proxy
            self.capture_proxy = None
            self._reset_stream_runtime_state(clear_timestamps=True)

        if old_proxy is not None:
            try:
                old_proxy.stop()
            except Exception as e:
                logging.warning(f"[{self.name}] 停止旧的 CaptureProxy 失败: {e}")

        if not self.is_running or self._stop_event.is_set():
            return False

        try:
            self._start_capture_proxy()
            logging.info(f"[{self.name}] CaptureProxy 已重建完成")
            return True
        except Exception as e:
            logging.error(f"[{self.name}] CaptureProxy 重建失败: {e}")
            return False

    def _maybe_restart_stale_capture(self, now: Optional[float] = None) -> bool:
        if not self.is_running or self._stop_event.is_set() or self.capture_proxy is None:
            return False

        now = time.time() if now is None else now
        if self._last_frame_ts > 0:
            frame_age = now - self._last_frame_ts
            if frame_age >= self._capture_stall_timeout:
                return self._restart_capture_proxy(
                    f"frame_timeout frame_age={frame_age:.1f}s status={self._last_capture_status}"
                )
            return False

        if self._capture_session_started_ts > 0:
            startup_age = now - self._capture_session_started_ts
            if startup_age >= self._capture_start_timeout:
                return self._restart_capture_proxy(
                    f"startup_timeout no_frame_for={startup_age:.1f}s status={self._last_capture_status}"
                )
        return False

    def _capture_watchdog_loop(self):
        logging.info(f"[{self.name}] 拉流看门狗线程启动")
        while self.is_running and not self._stop_event.wait(self._capture_watchdog_interval):
            try:
                self._maybe_restart_stale_capture()
            except Exception as e:
                logging.error(f"[{self.name}] 拉流看门狗检查异常: {e}")
        logging.info(f"[{self.name}] 拉流看门狗线程结束")

    def get_activity_snapshot(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        frame_age = (now - self._last_frame_ts) if self._last_frame_ts > 0 else -1.0
        process_age = (now - self._last_processed_ts) if self._last_processed_ts > 0 else -1.0
        push_age = (now - self._last_push_ts) if self._last_push_ts > 0 else -1.0
        startup_age = (now - self._capture_session_started_ts) if self._capture_session_started_ts > 0 else -1.0
        push_timeout = max(5.0, self._capture_watchdog_interval + 2.0)

        active = True
        reasons = []

        if not self.is_running:
            active = False
            reasons.append('stopped')
        elif self._last_frame_ts <= 0:
            if startup_age >= self._capture_start_timeout:
                active = False
                reasons.append('no_frame')
        elif frame_age > self._capture_stall_timeout:
            active = False
            reasons.append(f'frame_age={frame_age:.1f}s')

        if self.output_url and self.is_running:
            if self._last_frame_ts > 0:
                if self._last_push_ts <= 0:
                    active = False
                    reasons.append('no_push')
                elif push_age > push_timeout:
                    active = False
                    reasons.append(f'push_age={push_age:.1f}s')
            elif startup_age >= self._capture_start_timeout and self._last_push_ts <= 0:
                active = False
                reasons.append('no_push')

        if self._last_capture_status in ('error', 'interrupted'):
            if self._last_frame_ts <= 0 or frame_age > push_timeout:
                active = False
                reasons.append(f'status={self._last_capture_status}')

        return {
            'active': active,
            'reason': ','.join(reasons),
            'frame_age': frame_age,
            'process_age': process_age,
            'push_age': push_age,
            'startup_age': startup_age,
            'capture_status': self._last_capture_status,
        }

    def is_stream_active(self, now: Optional[float] = None) -> bool:
        return bool(self.get_activity_snapshot(now=now).get('active'))

    def start(self):
        self.is_running = True
        self._stop_event.clear()
        self.inference_scheduler.ensure_stream(self.stream_tracking_key)

        self._detected_resolution = self.config.get_default_resolution()
        logging.info(
            f"[{self.name}] 启动时跳过阻塞分辨率探测，先使用默认分辨率: "
            f"{self._detected_resolution[0]}x{self._detected_resolution[1]}"
        )

        if self.output_url:
            self._open_ffmpeg(self.output_url)

        self._start_capture_proxy()

        self._processor_thread = threading.Thread(
            target=self._process_frames_loop,
            daemon=True,
            name=f"Process-{self.name}",
        )
        self._processor_thread.start()
        self._push_thread = threading.Thread(target=self._push_loop, daemon=True, name=f"Push-{self.name}")
        self._push_thread.start()
        self._capture_watchdog_thread = threading.Thread(
            target=self._capture_watchdog_loop,
            daemon=True,
            name=f"CaptureWatchdog-{self.name}",
        )
        self._capture_watchdog_thread.start()

        logging.info(f"[{self.name}] 启动完成")

    def stop(self):
        logging.info(f"[{self.name}] 停止中...")
        self.is_running = False
        self._stop_event.set()
        self._frame_ready_event.set()

        if self.capture_proxy:
            self.capture_proxy.stop()
            self.capture_proxy = None

        if self.video_processor:
            self.video_processor.stop()
            self.video_processor = None

        self._reset_stream_runtime_state(clear_timestamps=True)

        self.inference_scheduler.remove_stream(self.stream_tracking_key)
        self._close_ffmpeg(release_nvenc=True)
        if self._owns_inference_engine:
            try:
                self.inference_engine.cleanup()
            except Exception as e:
                logging.error(f"[{self.name}] 清理独立推理引擎失败: {e}")
        logging.info(f"[{self.name}] 已停止")

    def _trace_stage(self, fid: int, stage: str, **kwargs):
        if not getattr(self, '_crash_trace_enabled', False):
            return
        extras = []
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
        suffix = f" {' '.join(extras)}" if extras else ''
        logging.info(f"[{self.name}] fid={fid} stage={stage}{suffix}")
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def _on_frame(self, stream_id: str, frame: np.ndarray):
        self._last_frame_ts = time.time()
        try:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            fh, fw = frame.shape[:2]
            actual_resolution = (fw, fh)
            if self._detected_resolution != actual_resolution:
                previous_resolution = self._detected_resolution
                self._detected_resolution = actual_resolution
                if previous_resolution:
                    logging.info(
                        f"[{self.name}] 首帧确认真实分辨率: {fw}x{fh} "
                        f"(原启动值 {previous_resolution[0]}x{previous_resolution[1]})"
                    )
                else:
                    logging.info(f"[{self.name}] 首帧确认真实分辨率: {fw}x{fh}")
                if self.output_url and self._push_ffmpeg_resolution and self._push_ffmpeg_resolution != actual_resolution:
                    self._push_reset_needed = True
            with self._latest_frame_lock:
                self._latest_input_frame = frame
            self._frame_ready_event.set()
        except Exception as e:
            logging.error(f"[{self.name}] 接收帧异常: {e}")

    def _take_latest_input_frame(self) -> Optional[np.ndarray]:
        with self._latest_frame_lock:
            frame = self._latest_input_frame
            self._latest_input_frame = None
        return frame

    def _store_latest_rendered_frame(self, frame: np.ndarray):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        with self._latest_render_lock:
            self._latest_rendered_frame = frame

    def _enqueue_rendered_frame(self, frame: np.ndarray):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        with self._latest_render_lock:
            self._latest_rendered_frame = frame
        with self._render_queue_lock:
            self._render_queue.append(frame)

    def _take_next_rendered_frame(self) -> Optional[np.ndarray]:
        with self._render_queue_lock:
            if not self._render_queue:
                return None
            return self._render_queue.popleft()

    def _take_latest_rendered_frame(self) -> Optional[np.ndarray]:
        with self._latest_render_lock:
            frame = self._latest_rendered_frame
            self._latest_rendered_frame = None
        return frame

    def _process_frames_loop(self):
        logging.info(f"[{self.name}] 处理线程启动")
        while self.is_running and not self._stop_event.is_set():
            self._frame_ready_event.wait(1.0)
            if self._stop_event.is_set():
                break
            frame = self._take_latest_input_frame()
            if frame is None:
                self._frame_ready_event.clear()
                continue
            if self._latest_input_frame is None:
                self._frame_ready_event.clear()
            self._process_frame(frame)
        logging.info(f"[{self.name}] 处理线程结束")

    def _estimate_motion_level(self, frame: np.ndarray) -> float:
        if not bool(getattr(self.config, 'motion_detection_enabled', True)):
            return 100.0
        try:
            width = int(getattr(self.config, 'motion_resize_width', 160))
            height = int(getattr(self.config, 'motion_resize_height', 90))
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if self._motion_prev_small is None:
                self._motion_prev_small = gray
                return 100.0
            diff = cv2.absdiff(gray, self._motion_prev_small)
            self._motion_prev_small = gray
            return float(diff.mean())
        except Exception:
            return 100.0

    def _current_inference_interval(self, frame: np.ndarray) -> int:
        base_interval = max(1, int(getattr(self.config, 'detection_inference_interval', 5)))
        idle_interval = max(base_interval, int(getattr(self.config, 'inference_idle_interval', base_interval)))
        hold_seconds = float(getattr(self.config, 'inference_active_hold_seconds', 2.0) or 2.0)
        motion_threshold = float(getattr(self.config, 'motion_threshold', 3.5) or 3.5)

        motion_level = self._estimate_motion_level(frame)
        self._last_motion_level = motion_level
        active_recently = self._last_target_seen_ts > 0 and (time.time() - self._last_target_seen_ts) <= hold_seconds
        if active_recently or motion_level >= motion_threshold:
            return base_interval
        return idle_interval

    def _process_frame(self, frame: np.ndarray):
        try:
            self._frame_id += 1
            fid = self._frame_id
            interval = self._current_inference_interval(frame)
            self._frame_buffer.append((fid, frame.copy()))
            if self._tracker is not None:
                self._last_detection_overlays = self._tracker.get_active_tracks()

            alert_detection_dict = {}
            alert_target_info = None
            alert_frame = None
            alert_raw_frame = None
            latest_result = self.inference_scheduler.get_latest_result(self.stream_tracking_key)
            if latest_result:
                result_fid = int(latest_result.get('frame_id', 0) or 0)
                result_ts = float(latest_result.get('result_ts', 0.0) or 0.0)
                result_age = time.time() - result_ts if result_ts > 0 else 0.0
                frame_lag = max(0, fid - result_fid)
                max_result_age = float(getattr(self.config, 'max_infer_result_age', 1.0) or 1.0)
                max_frame_lag = max(1, int(getattr(self.config, 'max_infer_frame_lag', 5) or 5))
                if (
                    result_fid > self._last_applied_result_frame_id
                    and result_age <= max_result_age
                    and frame_lag <= max_frame_lag
                ):
                    alert_payload = self._process_infer_results(latest_result.get('results', {}) or {}, result_fid)
                    if alert_payload:
                        alert_detection_dict = dict(alert_payload.get('detection_dict', {}) or {})
                        alert_target_info = dict(alert_payload.get('target_info', {}) or {})
                        alert_frame = alert_payload.get('frame')
                        alert_raw_frame = alert_payload.get('raw_frame')
                    self._last_applied_result_frame_id = result_fid
                else:
                    # 记录推理结果被丢弃的原因，便于调试
                    if result_fid <= self._last_applied_result_frame_id:
                        logging.debug(f"[{self.name}] 丢弃旧推理结果: result_fid={result_fid} <= last_applied={self._last_applied_result_frame_id}")
                    elif result_age > max_result_age:
                        logging.debug(f"[{self.name}] 丢弃过期推理结果: result_age={result_age:.2f}s > max={max_result_age}s, fid={fid}, result_fid={result_fid}")
                    elif frame_lag > max_frame_lag:
                        logging.debug(f"[{self.name}] 丢弃滞后推理结果: frame_lag={frame_lag} > max={max_frame_lag}, fid={fid}, result_fid={result_fid}")

            # 异步推理：提交新请求
            if self.inference_scheduler.is_loaded() and (fid - self._last_infer_frame_id) >= interval:
                self._last_infer_frame_id = fid
                self._pending_infer = self.inference_scheduler.submit_frame(
                    stream_key=self.stream_tracking_key,
                    frame=frame,
                    algo_id=None,
                    frame_id=fid,
                )

            rendered_frame = frame.copy()
            rendered_frame = self._draw_boundary_projector(rendered_frame, fid=fid)
            if self._last_detection_overlays:
                rendered_frame = self._draw_detection_overlays(rendered_frame, self._last_detection_overlays)
            rendered_frame = self._draw_ai_badge(rendered_frame, fid=fid)
            self._enqueue_rendered_frame(rendered_frame)
            frame_ts = time.time()
            self._last_processed_ts = frame_ts

            if self.alert_system.alert_handler:
                self.alert_system.alert_handler.collect_clip_frame(rendered_frame, frame_ts=frame_ts)
                if alert_detection_dict and alert_target_info and alert_frame is not None:
                    self.alert_system.process_frame_alerts(
                        alert_frame,
                        alert_detection_dict,
                        target_info=alert_target_info,
                        frame_ts=frame_ts,
                        raw_frame=alert_raw_frame,
                    )

        except Exception as e:
            logging.error(f"[{self.name}] 帧处理异常: {e}")

    def _process_infer_results(self, results, fid):
        """处理推理结果，更新检测覆盖层"""
        raw_detections = []
        total = 0
        class_names = set()
        for aid, res in results.items():
            model_detections = self._extract_raw_detections(res, aid, fid=fid)
            # 2026-06-16 16:49 修改目的：过滤后再计数，避免被过滤类别继续触发告警。
            total += len(model_detections)
            raw_detections.extend(model_detections)
            for det in model_detections:
                c = str(det.get('class_name', '')).strip()
                if c:
                    class_names.add(c)
        track_ids = set()
        tracking_enabled = bool(getattr(self.config, 'tracking_enabled', False))
        if self._tracker is not None:
            reference_frame = self._find_frame_in_buffer(fid)
            self._tracker.update(raw_detections, frame=reference_frame)
            overlays = self._tracker.get_active_tracks()
            for overlay in overlays:
                tid = overlay.get('track_id')
                if tid not in (None, ''):
                    try:
                        track_ids.add(int(tid))
                    except Exception:
                        track_ids.add(tid)
        else:
            overlays = self._build_overlays_from_detections(raw_detections)

        alarm_count = len(track_ids) if tracking_enabled and track_ids else total
        if alarm_count > 0:
            self._last_target_seen_ts = time.time()
            if tracking_enabled and track_ids:
                current_track_ids = set(track_ids)
                if current_track_ids != self._logged_active_track_ids:
                    logging.info(
                        f"[{self.name}] 检测到目标: "
                        f"alarm_count={alarm_count}, track_ids={sorted(track_ids, key=lambda x: str(x))}, classes={sorted(class_names)}"
                    )
                    self._logged_active_track_ids = current_track_ids
            elif self._last_infer_result_count <= 0:
                logging.info(
                    f"[{self.name}] 检测到目标: "
                    f"alarm_count={alarm_count}, classes={sorted(class_names)}"
                )
        else:
            self._logged_active_track_ids.clear()
        self._last_infer_result_ts = time.time()
        self._last_infer_result_count = int(alarm_count)
        self._last_tracking_summary = {
            'track_count': int(len(track_ids)) if tracking_enabled else int(total),
            'track_ids': sorted(track_ids, key=lambda x: str(x)),
            'classes': sorted(class_names),
        }
        self._last_detection_overlays = overlays
        if alarm_count <= 0:
            return None

        reference_frame = self._find_frame_in_buffer(fid)
        if reference_frame is None:
            return None

        projected_curves = self._project_boundary_curves_for_frame(reference_frame)
        confirmed_frame = reference_frame.copy()
        confirmed_frame = self._draw_boundary_projector(confirmed_frame, fid=fid)
        if overlays:
            confirmed_frame = self._draw_detection_overlays(confirmed_frame, overlays)
        confirmed_frame = self._draw_ai_badge(confirmed_frame, fid=fid)
        target_info = self._build_alert_target_info(overlays, self._last_tracking_summary, projected_curves)
        if not bool(target_info.get('should_alert')):
            try:
                if self.alert_system and self.alert_system.alert_handler:
                    self.alert_system.alert_handler.save_suppressed_image(
                        confirmed_frame,
                        reason=str(target_info.get('alarm_level_source') or 'outside_orange_or_unresolved'),
                        target_info=target_info,
                        frame_ts=self._last_frame_ts if self._last_frame_ts > 0 else None,
                    )
            except Exception as e:
                logging.debug(f"[{self.name}] 保存未上报告警调试图失败: {e}")
            return None
        return {
            'frame': confirmed_frame,
            'raw_frame': reference_frame,
            'detection_dict': {
                'alarm_any_detection': float(target_info.get('track_count', 0)),
            },
            'target_info': target_info,
        }

    @staticmethod
    def _coerce_alarm_level_value(level_value: Any, default: str = "3") -> str:
        try:
            level_int = int(level_value)
        except Exception:
            return str(default)
        if level_int not in (1, 2, 3, 4):
            return str(default)
        return str(level_int)

    @staticmethod
    def _extract_overlay_target_candidate(overlay: dict) -> Optional[Dict[str, Any]]:
        if not isinstance(overlay, dict):
            return None
        xyxy = tuple(overlay.get('xyxy', ()) or ())
        if len(xyxy) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
        except Exception:
            return None
        return {
            'bbox': [x1, y1, x2, y2],
            'u': (float(x1) + float(x2)) * 0.5,
            'v': float(y2),
            'class_name': str(overlay.get('class_name', '') or '').strip() or 'unknown',
            'track_id': overlay.get('track_id'),
            'confidence': float(overlay.get('confidence', 0.0) or 0.0),
            'algo_id': str(overlay.get('algo_id', '') or ''),
        }

    def _resolve_overlay_alarm_level(
        self,
        target_info: Optional[Dict[str, Any]],
        projected_curves,
    ) -> Tuple[Optional[str], str, Dict[str, Any]]:
        if not isinstance(target_info, dict):
            return None, "no_target", {}
        if not projected_curves or not self.boundary_projector_border_json:
            return None, "no_projected_curves", {}
        try:
            point_uv = (float(target_info.get('u')), float(target_info.get('v')))
            level_details = classify_point_alarm_level_uv_details(
                point_uv,
                projected_curves=projected_curves,
                border_json_path=self.boundary_projector_border_json,
                stream_name=self.name,
            )
            if level_details.get("alarm_level") is None:
                return None, str(level_details.get("reason") or "outside_orange_or_unresolved"), level_details
            return (
                self._coerce_alarm_level_value(level_details.get("alarm_level")),
                str(level_details.get("reason") or "uv_projected_curves"),
                level_details,
            )
        except Exception as e:
            logging.debug(f"[{self.name}] 目标报警等级计算异常: {e}")
            return None, "error", {}

    def _build_alert_target_info(self, overlays, tracking_summary, projected_curves=None):
        overlay_classes = sorted({
            str(o.get('class_name', '')).strip()
            for o in overlays or []
            if str(o.get('class_name', '')).strip()
        })
        class_text = ','.join(overlay_classes) if overlay_classes else 'unknown'
        validation_boxes = []
        for overlay in overlays or []:
            xyxy = tuple(overlay.get('xyxy', ()) or ())
            color = tuple(overlay.get('color', ()) or ())
            if len(xyxy) == 4 and len(color) == 3:
                validation_boxes.append({
                    'xyxy': [int(v) for v in xyxy],
                    'color': [int(v) for v in color],
                })
        target_candidates: List[Dict[str, Any]] = []
        for overlay in overlays or []:
            candidate = self._extract_overlay_target_candidate(overlay)
            if not candidate:
                continue
            alarm_level, alarm_level_source, alarm_level_diag = self._resolve_overlay_alarm_level(candidate, projected_curves)
            boundaries_preview = list((alarm_level_diag or {}).get('boundaries') or [])[:4]
            logging.info(
                f"[{self.name}] [alarm_level_candidate] "
                f"class={candidate.get('class_name')} "
                f"track_id={candidate.get('track_id')} "
                f"bbox={candidate.get('bbox')} "
                f"point=({candidate.get('u')},{candidate.get('v')}) "
                f"level={alarm_level} "
                f"reason={alarm_level_source} "
                f"visible_colors={(alarm_level_diag or {}).get('visible_colors')} "
                f"scan_y={(alarm_level_diag or {}).get('matched_scan_y')} "
                f"boundaries={boundaries_preview}"
            )
            if alarm_level is None:
                continue
            candidate['alarm_level'] = str(alarm_level)
            candidate['alarm_level_source'] = str(alarm_level_source)
            candidate['alarm_level_visible_colors'] = list((alarm_level_diag or {}).get('visible_colors') or [])
            candidate['alarm_level_scan_y'] = (alarm_level_diag or {}).get('matched_scan_y')
            candidate['alarm_level_boundaries'] = list((alarm_level_diag or {}).get('boundaries') or [])
            target_candidates.append(candidate)

        target_candidates.sort(
            key=lambda item: (
                int(item.get('alarm_level', '3')),
                0 if item.get('track_id') not in (None, '') else 1,
                -float(item.get('confidence', 0.0) or 0.0),
            )
        )
        track_count = int(len(target_candidates)) if target_candidates else 0
        out = {
            'classes': class_text,
            'class_name': class_text,
            'count': track_count,
            'track_count': track_count,
            'track_ids': list((tracking_summary or {}).get('track_ids', []) or []),
            'tracking_enabled': bool(getattr(self.config, 'tracking_enabled', False)),
            '_validation_boxes': validation_boxes,
            'target_candidates': target_candidates,
            'alarm_level': None,
            'alarm_level_source': 'outside_orange_or_unresolved',
            'should_alert': bool(target_candidates),
        }
        if target_candidates:
            selected_candidate = target_candidates[0]
            out.update({
                'class_name': str(selected_candidate.get('class_name') or class_text),
                'alarm_level': str(selected_candidate.get('alarm_level')),
                'alarm_level_source': str(selected_candidate.get('alarm_level_source')),
                'bbox': list(selected_candidate.get('bbox') or []),
                'u': selected_candidate.get('u'),
                'v': selected_candidate.get('v'),
                'track_id': selected_candidate.get('track_id'),
                'confidence': selected_candidate.get('confidence'),
                'alarm_level_visible_colors': list(selected_candidate.get('alarm_level_visible_colors') or []),
                'alarm_level_scan_y': selected_candidate.get('alarm_level_scan_y'),
                'alarm_level_boundaries': list(selected_candidate.get('alarm_level_boundaries') or []),
            })
            logging.info(
                f"[{self.name}] [alarm_level_selected] "
                f"class={out.get('class_name')} "
                f"track_id={out.get('track_id')} "
                f"bbox={out.get('bbox')} "
                f"point=({out.get('u')},{out.get('v')}) "
                f"level={out.get('alarm_level')} "
                f"reason={out.get('alarm_level_source')} "
                f"visible_colors={out.get('alarm_level_visible_colors')} "
                f"scan_y={out.get('alarm_level_scan_y')} "
                f"boundaries={list(out.get('alarm_level_boundaries') or [])[:4]} "
                f"candidate_count={len(target_candidates)}"
            )
        else:
            logging.info(
                f"[{self.name}] [alarm_level_suppressed] "
                f"reason=outside_orange_or_unresolved "
                f"raw_candidate_count={len(overlays or [])}"
            )
        return out

    def _on_status_change(self, stream_id: str, status: VideoStreamStatus):
        self._last_capture_status = status.value
        logging.info(f"[{self.name}] 流状态: {status.value}")
        if status == VideoStreamStatus.INTERRUPTED:
            self._reset_stream_runtime_state(clear_timestamps=True)
        elif status in (VideoStreamStatus.READING, VideoStreamStatus.CONNECTED):
            self._push_reset_needed = True

    def _on_error(self, stream_id: str, error_msg: str):
        logging.error(f"[{self.name}] 流错误: {error_msg}")

    def _count_detections(self, results) -> int:
        try:
            r = results[0] if isinstance(results, (list, tuple)) and results else results
            if r is not None and hasattr(r, 'boxes') and r.boxes is not None:
                return int(len(r.boxes))
        except Exception:
            pass
        return 0

    @staticmethod
    def _normalize_filtered_class_names(raw_value) -> set:
        # 2026-06-16 16:49 修改目的：将类别名过滤配置统一为小写集合，避免大小写和空格导致过滤失效。
        if raw_value in (None, ''):
            return set()
        raw_items = raw_value if isinstance(raw_value, (list, tuple, set)) else [raw_value]
        class_names = set()
        for item in raw_items:
            class_name = str(item or '').strip().lower()
            if class_name:
                class_names.add(class_name)
        return class_names

    def _should_keep_detection_class(self, class_name: str) -> bool:
        if not self._detection_filtered_class_names:
            return True
        return str(class_name or '').strip().lower() not in self._detection_filtered_class_names

    def _extract_raw_detections(self, results, algo_id: str, fid: Optional[int] = None):
        detections = []
        try:
            r = results[0] if isinstance(results, (list, tuple)) and results else results
            if r is None or not hasattr(r, 'boxes') or r.boxes is None:
                return detections
            boxes = r.boxes
            if len(boxes) == 0:
                return detections
            if fid is not None:
                self._trace_stage(fid, 'tensor_extract_start', algo_id=algo_id)
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, 'cpu') else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.asarray(boxes.conf)
            clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, 'cpu') else np.asarray(boxes.cls)
            if fid is not None:
                self._trace_stage(fid, 'tensor_extract_end', algo_id=algo_id, box_count=len(xyxy))
            names = getattr(r, 'names', {})
            color = self._color_for_model(algo_id)
            raw_detections = []
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf = float(confs[i])
                cls_id = int(clss[i])
                label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                # 2026-06-16 16:49 修改目的：按类别名过滤检测框，后续画框、跟踪、告警共用同一结果。
                if not self._should_keep_detection_class(label):
                    continue
                detections.append({
                    'xyxy': (x1, y1, x2, y2),
                    'color': color,
                    'class_name': str(label),
                    'algo_id': str(algo_id),
                    'confidence': conf,
                })
        except Exception as e:
            logging.debug(f"[{self.name}] 提取绘框信息异常: {e}")
        return detections

    def _build_overlays_from_detections(self, detections):
        overlays = []
        for det in detections or []:
            overlays.append({
                'xyxy': det['xyxy'],
                'text': f"{det['class_name']} {det['confidence']:.2f}",
                'color': det['color'],
                'class_name': det['class_name'],
                'algo_id': det['algo_id'],
                'confidence': det['confidence'],
                'track_id': None,
            })
        return overlays

    def _find_frame_in_buffer(self, fid: int) -> Optional[np.ndarray]:
        for frame_id, frame in reversed(self._frame_buffer):
            if int(frame_id) == int(fid):
                return frame
        return None
    def _draw_ai_badge(self, frame: np.ndarray, fid: Optional[int] = None) -> np.ndarray:
        try:
            badge_text = "AI"
            ts_text = time.strftime('%H:%M:%S')
            if fid is not None:
                ts_text = f"{ts_text} F{fid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            margin = 12
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)
            (ts_w, ts_h), ts_baseline = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            h, w = frame.shape[:2]
            box_w = max(text_w + 24, ts_w + 24)
            box_h = text_h + ts_h + baseline + ts_baseline + 28
            x1 = w - box_w - margin
            y1 = h - box_h - margin
            x2 = w - margin
            y2 = h - margin
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, badge_text, (x1 + 12, y1 + text_h + 8), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, ts_text, (x1 + 12, y2 - ts_baseline - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        except Exception as e:
            logging.debug(f"[{self.name}] 绘制AI标识异常: {e}")
        return frame

    def _draw_detection_overlays(self, frame: np.ndarray, overlays) -> np.ndarray:
        try:
            for overlay in overlays or []:
                x1, y1, x2, y2 = overlay['xyxy']
                color = overlay['color']
                text = overlay['text']
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except Exception as e:
            logging.debug(f"[{self.name}] 绘制缓存框异常: {e}")
        return frame

    def _draw_detections(self, frame: np.ndarray, results, algo_id: str) -> np.ndarray:
        overlays = self._build_overlays_from_detections(
            self._extract_raw_detections(results, algo_id)
        )
        return self._draw_detection_overlays(frame, overlays)

    def _color_for_model(self, algo_id: str):
        palette = [
            (0, 255, 0),
            (0, 165, 255),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        idx = sum(ord(c) for c in str(algo_id)) % len(palette)
        return palette[idx]

    def _open_ffmpeg(self, output_url: str):
        self.output_url = output_url
        w, h = self._detected_resolution or self.config.get_default_resolution()
        fps = getattr(self.config, 'push_fps', self.config.fps)
        self._push_ffmpeg_resolution = (w, h)
        output_scheme = (urlparse(output_url).scheme or 'rtsp').lower()

        def _build_cmd(codec: str, hw: bool):
            cmd = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning', '-nostats',
                '-fflags', 'nobuffer', '-flags', 'low_delay',
                '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{w}x{h}', '-r', str(fps), '-i', '-',
                '-an',
                '-c:v', codec,
            ]
            if hw:
                cmd += ['-preset', getattr(self.config, 'encoding_preset', 'p4'), '-tune', 'll']
            else:
                cmd += ['-preset', 'ultrafast', '-tune', 'zerolatency']
            cmd += [
                '-g', str(max(getattr(self.config, 'gop_size', 50), fps)),
                '-keyint_min', str(max(1, fps)),
                '-bf', '0',
                '-b:v', getattr(self.config, 'bitrate', '4M'),
                '-maxrate', getattr(self.config, 'max_bitrate', '6M'),
                '-bufsize', getattr(self.config, 'buffer_size', '8M'),
                '-pix_fmt', 'yuv420p',
                '-flush_packets', '1',
            ]
            if output_scheme == 'rtsp':
                cmd += ['-rtsp_transport', 'tcp', '-muxdelay', '0', '-muxpreload', '0', '-f', 'rtsp', output_url]
            elif output_scheme == 'rtmp':
                cmd += ['-flvflags', 'no_duration_filesize', '-f', 'flv', output_url]
            else:
                raise ValueError(f"不支持的推流协议: {output_scheme}")
            return cmd

        # 确定本流使用的编码器（带 NVENC 槽位管理）
        codec = self._resolve_push_codec()
        hw = codec == VideoCodec.H264_NVENC.value

        codecs_to_try = [codec]
        if hw:
            codecs_to_try.append(VideoCodec.LIBX264.value)

        for attempt, try_codec in enumerate(codecs_to_try):
            try_hw = try_codec == VideoCodec.H264_NVENC.value
            cmd = _build_cmd(try_codec, try_hw)
            logging.info(f"[{self.name}] FFmpeg命令(尝试{attempt + 1}, codec={try_codec}): {' '.join(cmd)}")
            try:
                pipe = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    start_new_session=(os.name != 'nt'),
                )
            except Exception as e:
                logging.error(f"[{self.name}] 启动FFmpeg失败: {e}")
                if try_hw and self._using_nvenc:
                    _nvenc_release()
                    self._using_nvenc = False
                continue
            time.sleep(0.3)
            if pipe.poll() is not None:
                err = ''
                try:
                    err = pipe.stderr.read().decode('utf-8', errors='ignore')[:500]
                except Exception:
                    pass
                logging.error(f"[{self.name}] FFmpeg启动即退出(codec={try_codec}): {err}")
                try:
                    pipe.kill()
                except Exception:
                    pass
                # NVENC 失败，释放槽位，回退 libx264
                if try_hw and self._using_nvenc:
                    _nvenc_release()
                    self._using_nvenc = False
                    logging.warning(f"[{self.name}] NVENC编码失败，回退libx264")
                continue
            # 启动成功
            self.pipe = pipe
            self._stream_codec = try_codec
            self._ffmpeg_restart_backoff = self._ffmpeg_restart_backoff_initial  # 成功后重置退避
            logging.info(f"[{self.name}] FFmpeg推流启动成功(codec={try_codec}) -> {output_url}")
            return

        logging.error(f"[{self.name}] FFmpeg推流进程启动失败（所有编码器均失败）")

    def _resolve_push_codec(self) -> str:
        """决定本流使用的编码器，带 NVENC 会话槽位管理"""
        # 如果已经确定了回退编码器，继续使用
        if self._stream_codec == VideoCodec.LIBX264.value:
            return VideoCodec.LIBX264.value

        push_device = self.config.get_push_device()
        want_nvenc = False
        if push_device == 'cpu':
            return VideoCodec.LIBX264.value
        elif push_device == 'gpu':
            want_nvenc = True
        elif self.config.is_auto_codec_enabled():
            want_nvenc = True
        else:
            codec_val = self.config.get_video_codec()
            want_nvenc = (codec_val == VideoCodec.H264_NVENC.value)

        if want_nvenc:
            # 如果本流已经持有 NVENC 槽位，直接用
            if self._using_nvenc:
                return VideoCodec.H264_NVENC.value
            # 尝试获取 NVENC 槽位
            if _nvenc_acquire():
                self._using_nvenc = True
                logging.info(f"[{self.name}] 获取NVENC槽位成功")
                return VideoCodec.H264_NVENC.value
            else:
                logging.warning(f"[{self.name}] NVENC槽位已满，使用libx264")
                self._stream_codec = VideoCodec.LIBX264.value
                return VideoCodec.LIBX264.value

        return VideoCodec.LIBX264.value

    def _close_ffmpeg(self, release_nvenc: bool = False):
        if self.pipe:
            try:
                _terminate_subprocess(self.pipe, timeout=3.0)
            except Exception:
                pass
            self.pipe = None
        if release_nvenc and self._using_nvenc:
            _nvenc_release()
            self._using_nvenc = False
            self._stream_codec = None  # 下次重新决定编码器

    def _read_ffmpeg_stderr(self, pipe: Optional[subprocess.Popen]) -> str:
        if not pipe or not pipe.stderr:
            return ''
        try:
            if pipe.poll() is None:
                return ''
            err = pipe.stderr.read().decode('utf-8', errors='ignore').strip()
            return err[:1000]
        except Exception:
            return ''

    def _describe_ffmpeg_failure(self) -> str:
        pipe = self.pipe
        if not pipe:
            return 'FFmpeg进程不存在'

        reasons = []
        return_code = pipe.poll()
        if return_code is not None:
            reasons.append(f'returncode={return_code}')
            err = self._read_ffmpeg_stderr(pipe)
            if err:
                reasons.append(f'stderr={err}')
        if not pipe.stdin or pipe.stdin.closed:
            reasons.append('stdin已关闭')
        return '; '.join(reasons) if reasons else ''

    def _restart_ffmpeg(self) -> bool:
        self._close_ffmpeg(release_nvenc=True)
        if self.output_url:
            self._open_ffmpeg(self.output_url)
            return self.pipe is not None
        return False

    def _build_pipeline_snapshot(self) -> str:
        now = time.time()
        frame_age = (now - self._last_frame_ts) if self._last_frame_ts else -1.0
        process_age = (now - self._last_processed_ts) if self._last_processed_ts else -1.0
        push_age = (now - self._last_push_ts) if self._last_push_ts else -1.0
        infer_age = (now - self._last_infer_result_ts) if self._last_infer_result_ts else -1.0
        with self._render_queue_lock:
            rendered_ready = bool(self._render_queue)
        return (
            f"capture_status={self._last_capture_status}, "
            f"frame_id={self._frame_id}, "
            f"frame_age={frame_age:.1f}s, "
            f"process_age={process_age:.1f}s, "
            f"push_age={push_age:.1f}s, "
            f"infer_age={infer_age:.1f}s, "
            f"last_alarm_count={self._last_infer_result_count}, "
            f"rendered_ready={rendered_ready}, "
            f"codec={self._stream_codec or 'unknown'}, "
            f"output_url={self.output_url or '(none)'}"
        )

    def _check_ffmpeg_health(self) -> bool:
        if not self.pipe:
            return False
        if self.pipe.poll() is not None:
            return False
        if not self.pipe.stdin or self.pipe.stdin.closed:
            return False
        return True

    def _push_loop(self):
        logging.info(f"[{self.name}] 推流线程启动")
        fps = getattr(self.config, 'push_fps', self.config.fps)
        interval = 1.0 / fps
        next_push_time = time.perf_counter()
        last_frame: Optional[np.ndarray] = None
        repeated_frame_count = 0
        max_repeat_frames = max(1, int(getattr(self.config, 'push_max_repeat_frames', 600) or 600))
        stale_repeat_window = max(
            float(getattr(self.config, 'push_stale_repeat_window', 30.0) or 30.0),
            interval * 10,
        )
        frame_count = 0

        while self.is_running:
            if self._push_reset_needed:
                last_frame = None
                repeated_frame_count = 0
                self._push_reset_needed = False

            now = time.perf_counter()
            sleep_time = next_push_time - now
            if sleep_time > 0:
                time.sleep(min(sleep_time, 0.02))
                continue
            if sleep_time < -interval * 3:
                next_push_time = now

            frame = None
            next_frame = self._take_next_rendered_frame()
            got_new_frame = next_frame is not None

            if got_new_frame:
                frame = next_frame
                last_frame = next_frame
                repeated_frame_count = 0
            elif (
                last_frame is not None
                and repeated_frame_count < max_repeat_frames
                and self._last_processed_ts
                and (time.time() - self._last_processed_ts) <= stale_repeat_window
            ):
                frame = last_frame
                repeated_frame_count += 1
            else:
                next_push_time = now + interval
                time.sleep(0.005)
                continue

            if not self.output_url:
                next_push_time += interval
                continue

            if not self._check_ffmpeg_health():
                failure_reason = self._describe_ffmpeg_failure()
                logging.error(f"[{self.name}] FFmpeg健康检查失败")
                logging.error(f"[{self.name}] 管道状态快照: {self._build_pipeline_snapshot()}")
                if failure_reason:
                    logging.warning(f"[{self.name}] FFmpeg异常，尝试重启(退避{self._ffmpeg_restart_backoff:.1f}s)，原因: {failure_reason}")
                else:
                    logging.warning(f"[{self.name}] FFmpeg异常，尝试重启(退避{self._ffmpeg_restart_backoff:.1f}s)")
                time.sleep(self._ffmpeg_restart_backoff)
                if not self._restart_ffmpeg():
                    # 指数退避，最大30秒
                    self._ffmpeg_restart_backoff = min(self._ffmpeg_restart_backoff_max, self._ffmpeg_restart_backoff * 2)
                    next_push_time = time.perf_counter() + interval
                    continue
                else:
                    self._ffmpeg_restart_backoff = self._ffmpeg_restart_backoff_initial  # 成功后重置
                    next_push_time = time.perf_counter() + interval
                    continue

            try:
                fh, fw = frame.shape[:2]
                exp = self._push_ffmpeg_resolution
                if exp and (fw, fh) != exp:
                    logging.warning(f"[{self.name}] 帧尺寸不一致 {fw}x{fh} vs {exp}，重启FFmpeg")
                    self._detected_resolution = (fw, fh)
                    self._restart_ffmpeg()
                    next_push_time = time.perf_counter() + interval
                    continue
            except Exception:
                pass

            try:
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                self.pipe.stdin.write(memoryview(frame))
                frame_count += 1
                self._last_push_ts = time.time()
                next_push_time += interval

                if self.alert_system.alert_handler:
                    self.alert_system.alert_handler.update_latest_push_frame(frame)

                if frame_count % 500 == 0:
                    logging.info(f"[{self.name}] 已推流 {frame_count} 帧")
            except BrokenPipeError:
                logging.error(f"[{self.name}] 推流管道断开(BrokenPipeError)")
                logging.error(f"[{self.name}] 管道状态快照: {self._build_pipeline_snapshot()}")
                failure_reason = self._describe_ffmpeg_failure()
                if failure_reason:
                    logging.error(f"[{self.name}] 推流管道断开，尝试恢复，原因: {failure_reason}")
                else:
                    logging.error(f"[{self.name}] 推流管道断开，尝试恢复")
                if not self._restart_ffmpeg():
                    self._ffmpeg_restart_backoff = min(self._ffmpeg_restart_backoff_max, self._ffmpeg_restart_backoff * 2)
                    time.sleep(self._ffmpeg_restart_backoff)
                else:
                    self._ffmpeg_restart_backoff = self._ffmpeg_restart_backoff_initial
                next_push_time = time.perf_counter() + interval
            except Exception as e:
                logging.error(f"[{self.name}] 推流写入失败: {e}")
                logging.error(f"[{self.name}] output write-failure snapshot: {self._build_pipeline_snapshot()}")
                time.sleep(0.5)
                next_push_time = time.perf_counter() + interval

        logging.info(f"[{self.name}] 推流线程结束")


class CameraStreamManager:
    """管理所有摄像头流的生命周期"""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.inference_scheduler = UnifiedInferenceScheduler(config)
        self._processors: Dict[str, StreamProcessor] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        self.running = False

    def start_all(self, stream_list: list):
        self.running = True
        for scfg in stream_list:
            if not scfg.get('enabled', True):
                continue
            self._start_stream(scfg)

    def _start_stream(self, scfg: dict):
        name = scfg.get('name', scfg.get('input_url') or scfg.get('rtsp_url') or scfg.get('rtmp_url', 'unknown'))
        with self._lock:
            if name in self._processors:
                logging.warning(f"流 [{name}] 已在运行")
                return
            proc = StreamProcessor(scfg, self.config, self.inference_scheduler)
            self._processors[name] = proc

        t = threading.Thread(target=self._run_stream, args=(name, proc), daemon=True, name=f"Stream-{name}")
        with self._lock:
            self._threads[name] = t
        t.start()
        logging.info(f"流 [{name}] 线程已启动")

    def _run_stream(self, name: str, proc: StreamProcessor):
        try:
            proc.start()
            while self.running and proc.is_running:
                time.sleep(1.0)
        except Exception as e:
            logging.error(f"流 [{name}] 运行异常: {e}")
        finally:
            try:
                proc.stop()
            except Exception:
                pass
            with self._lock:
                self._processors.pop(name, None)
                self._threads.pop(name, None)
            logging.info(f"流 [{name}] 线程退出")

    def stop_all(self):
        self.running = False
        with self._lock:
            procs = list(self._processors.values())
        for proc in procs:
            try:
                proc.stop()
            except Exception as e:
                logging.error(f"停止流异常: {e}")
        with self._lock:
            threads = list(self._threads.values())
        for t in threads:
            t.join(timeout=10.0)
        self.inference_scheduler.cleanup()
        logging.info("所有流已停止")

    def get_active_count(self) -> int:
        active, _ = self.get_activity_summary()
        return active

    def get_activity_summary(self) -> Tuple[int, List[str]]:
        with self._lock:
            items = list(self._processors.items())

        active = 0
        inactive = []
        now = time.time()
        for name, proc in items:
            try:
                if proc.is_stream_active(now=now):
                    active += 1
                else:
                    inactive.append(name)
            except Exception:
                inactive.append(name)
        return active, inactive


def _resolve_global_projector_config_path(config: CameraConfig):
    """为实时PTZ同步线程查找 camera_projector_config.json。"""
    path_candidates = [
        getattr(config, 'camera_projector_config_path', None),
        os.getenv('CAMERA_PROJECTOR_CONFIG'),
        project_root / 'config' / 'camera_projector_config.json',
        project_root / 'camera_projector_config.json',
    ]
    for p in path_candidates:
        if not p:
            continue
        p = Path(str(p))
        if not p.is_absolute():
            p = project_root / p
        if p.exists():
            return str(p)
    return None



def _resolve_global_projector_runtime_config_path(config: CameraConfig):
    """查找只读运行时配置：画线样式参数 + PTZ映射/yaw偏移量。"""
    path_candidates = [
        getattr(config, 'camera_projector_runtime_config_path', None),
        os.getenv('CAMERA_PROJECTOR_RUNTIME_CONFIG'),
        project_root / 'config' / 'camera_projector_runtime_config.json',
        project_root / 'config' / 'camera_projector_runtime_config_byGPT.json',
        project_root / 'camera_projector_runtime_config.json',
    ]
    for p in path_candidates:
        if not p:
            continue
        p = Path(str(p))
        if not p.is_absolute():
            p = project_root / p
        if p.exists():
            return str(p)
    return None

def _env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in ('0', 'false', 'no', 'off', 'disable', 'disabled')


def main():
    setup_logging()
    fault_log_handle = _enable_fault_logging()
    logging.info("=" * 60)
    logging.info("AI摄像头流检测系统启动")
    logging.info("=" * 60)

    config = CameraConfig()
    cfg_mgr = ConfigManager()
    stream_list = cfg_mgr.get_enabled_streams()

    if not stream_list:
        logging.error("未找到任何启用的流配置，请检查 config/config.json")
        return 1

    logging.info(f"共 {len(stream_list)} 路流:")
    for s in stream_list:
        input_url = s.get('input_url') or s.get('rtsp_url') or s.get('rtmp_url', '')
        output_url = s.get('output_url') or s.get('output_rtsp') or s.get('output_rtmp') or '(无推流)'
        camera_id = s.get('camera_id') or s.get('cameraId') or s.get('stream_id') or s.get('streamId') or s.get('monitorEq') or s.get('taskId') or s.get('name')
        logging.info(f"  [{s.get('name')}] camera_id={camera_id} {input_url} -> {output_url}")

    if config.push_enabled:
        push_dev = config.get_push_device()
        nvenc_info = f"，NVENC最大并发={_NVENC_MAX_SESSIONS}，超出自动回退libx264" if push_dev == 'gpu' else ''
        logging.info(f"AI输出流已启用，拉流设备={config.get_pull_device()}，推流设备={push_dev}，编码模式={config.get_video_codec()}{nvenc_info}")
    else:
        logging.info("AI输出流已禁用，仅本地处理")

    manager = CameraStreamManager(config)

    ptz_updater = None
    if start_realtime_ptz_config_updater is not None:
        ptz_config_path = _resolve_global_projector_config_path(config)
        ptz_runtime_config_path = _resolve_global_projector_runtime_config_path(config)
        ptz_enabled = _env_flag_enabled('CAMERA_PROJECTOR_REALTIME_PTZ_ENABLED', True)
        ptz_enabled = bool(getattr(config, 'camera_projector_realtime_ptz_enabled', ptz_enabled)) and ptz_enabled

        if ptz_enabled and ptz_config_path:
            ptz_updater = start_realtime_ptz_config_updater(
                ptz_config_path,
                enabled=True,
                poll_interval=float(getattr(config, 'camera_projector_ptz_poll_interval', 1.0) or 1.0),
                request_timeout=float(getattr(config, 'camera_projector_ptz_request_timeout', 1.5) or 1.5),
                verify_ssl=bool(getattr(config, 'camera_projector_ptz_verify_ssl', False)),
                logger=logging.getLogger(__name__),
                runtime_config_path=ptz_runtime_config_path,
            )
        elif ptz_enabled and not ptz_config_path:
            logging.warning("实时PTZ同步未启动：找不到 camera_projector_config.json")
        else:
            logging.info("实时PTZ同步未启用")
    else:
        logging.warning("实时PTZ同步未启动：无法导入 utils.camera_ptz_config_updater")

    def _signal_handler(signum, frame):
        logging.info(f"收到信号 {signum}，开始优雅退出.")
        try:
            if ptz_updater is not None:
                ptz_updater.stop()
        except Exception:
            pass
        manager.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    manager.start_all(stream_list)
    logging.info("所有流已启动，进入主监控循环（Ctrl+C 退出）")

    try:
        while True:
            active, inactive = manager.get_activity_summary()
            if inactive:
                logging.info(
                    f"[主循环] 活跃流数量: {active}/{len(stream_list)}, "
                    f"inactive={','.join(inactive)}"
                )
            else:
                logging.info(f"[主循环] 活跃流数量: {active}/{len(stream_list)}")
            time.sleep(30)
    except KeyboardInterrupt:
        logging.info("收到键盘中断，退出")
    finally:
        try:
            if ptz_updater is not None:
                ptz_updater.stop()
        except Exception:
            pass

        manager.stop_all()
        if fault_log_handle is not None:
            try:
                fault_log_handle.flush()
                fault_log_handle.close()
            except Exception:
                pass

    return 0


def _kill_child_processes():
    """主进程退出时强制清理所有子进程，防止孤儿进程残留"""
    try:
        import psutil
        current = psutil.Process(os.getpid())
        children = current.children(recursive=True)
        for child in children:
            try:
                child.kill()
            except Exception:
                pass
        try:
            psutil.wait_procs(children, timeout=3.0)
        except Exception:
            pass
    except ImportError:
        # psutil 不可用时用 os.killpg 兜底
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        except Exception:
            pass
    except Exception:
        pass


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver', force=True)
    atexit.register(_kill_child_processes)
    sys.exit(main())

