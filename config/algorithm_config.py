# -*- coding: utf-8 -*-
import json
import os
import contextlib
from enum import Enum
from urllib.parse import urlparse
from typing import Dict, Any, List


class InferenceEngineType(Enum):
    OPTIMIZED = "optimized"
    TENSORRT = "tensorrt"


class VideoCodec(Enum):
    AUTO = "auto"
    H264_NVENC = "h264_nvenc"
    LIBX264 = "libx264"


class DeviceMode(Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


def normalize_device_mode(value: Any, default: DeviceMode) -> DeviceMode:
    raw = str(value or default.value).strip().lower()
    for item in DeviceMode:
        if item.value == raw:
            return item
    return default


def build_capture_options(stream_url: str, pull_device: DeviceMode) -> str:
    scheme = (urlparse(stream_url).scheme or '').lower()
    options = []
    if scheme == 'rtsp':
        options.extend([
            'rtsp_transport;tcp',
            'reorder_queue_size;1024',
            'buffer_size;2097152',
            'max_delay;1000000',
            'stimeout;10000000',
        ])
    if pull_device == DeviceMode.GPU:
        options.extend([
            'hwaccel;cuda',
        ])
    return '|'.join(options)


class AlgorithmMode(Enum):
    REALTIME_MULTI = "realtime_multi"
    TRACKING_ONLY = "tracking_only"
    SEGMENTATION_ONLY = "segmentation_only"
    HYBRID = "hybrid"


class CameraConfig:
    """摄像头流AI检测配置类"""

    @contextlib.contextmanager
    def _suppress_capture_backend_logs(self):
        if os.name != 'nt':
            yield
            return

        stderr_fd = None
        saved_stderr_fd = None
        null_fd = None
        try:
            stderr_fd = os.dup(2)
            saved_stderr_fd = stderr_fd
            null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null_fd, 2)
            yield
        except Exception:
            yield
        finally:
            if saved_stderr_fd is not None:
                try:
                    os.dup2(saved_stderr_fd, 2)
                except Exception:
                    pass
                try:
                    os.close(saved_stderr_fd)
                except Exception:
                    pass
            if null_fd is not None:
                try:
                    os.close(null_fd)
                except Exception:
                    pass

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        self.config = self._load_config_file(config_file)

        self._load_algorithm_config()
        self._load_inference_config()
        self._load_stream_device_config()
        self._load_model_config()
        self._load_class_config()
        self._load_performance_config()
        self._load_alarm_config()
        self._load_ppe_config()
        self._load_video_config()
        self._load_application_config()
        self._load_minio_config()
        self._load_platform_config()

    def _load_config_file(self, config_file: str) -> Dict[str, Any]:
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            print(f"警告: 配置文件不存在: {config_file}")
            return {}
        except Exception as e:
            print(f"错误: 加载配置文件失败: {e}")
            return {}

    def _load_algorithm_config(self):
        algo = self.config.get('algorithm', {})
        mode_str = algo.get('mode', 'realtime_multi')
        try:
            self.algorithm_mode = AlgorithmMode(mode_str)
        except ValueError:
            self.algorithm_mode = AlgorithmMode.REALTIME_MULTI

        engine_str = algo.get('inference_engine_type', 'optimized')
        try:
            self.inference_engine_type = InferenceEngineType(engine_str)
        except ValueError:
            self.inference_engine_type = InferenceEngineType.OPTIMIZED

        tracker_cfg = self.config.get('tracking', {})
        self.tracking_enabled = self.algorithm_mode in {AlgorithmMode.TRACKING_ONLY, AlgorithmMode.HYBRID}
        self.tracking_persist = bool(tracker_cfg.get('persist', True))
        self.tracking_tracker = str(tracker_cfg.get('tracker', 'bytetrack.yaml') or 'bytetrack.yaml')
        self.tracking_conf_threshold = float(tracker_cfg.get('conf_threshold', algo.get('tracking_conf_threshold', 0.3) or 0.3))
        self.tracking_match_iou = float(tracker_cfg.get('match_iou', 0.3) or 0.3)

    def _load_inference_config(self):
        inference = self.config.get('inference', {})
        requested = str(inference.get('device', 'auto')).strip().lower()
        if requested == 'gpu':
            requested = 'cuda'
        self.inference_device = requested or 'auto'
        self.model_device = self.inference_device
        self.inference_single_thread_worker = bool(inference.get('single_thread_worker', False))
        self.inference_submit_timeout = float(inference.get('submit_timeout', 30.0) or 30.0)

    def _load_stream_device_config(self):
        stream = self.config.get('stream', {})
        self.pull_device_mode = normalize_device_mode(stream.get('pull_device', DeviceMode.CPU.value), DeviceMode.CPU)
        self.push_device_mode = normalize_device_mode(stream.get('push_device', DeviceMode.AUTO.value), DeviceMode.AUTO)
        self.pull_device = self.pull_device_mode.value
        self.push_device = self.push_device_mode.value
        self.capture_watchdog_interval = max(1.0, float(stream.get('capture_watchdog_interval', 5.0) or 5.0))
        self.capture_stall_timeout = max(
            self.capture_watchdog_interval * 2.0,
            float(stream.get('capture_stall_timeout', 15.0) or 15.0),
        )
        self.capture_start_timeout = max(
            self.capture_stall_timeout,
            float(stream.get('capture_start_timeout', 20.0) or 20.0),
        )
        self.capture_restart_cooldown = max(
            self.capture_watchdog_interval,
            float(stream.get('capture_restart_cooldown', 10.0) or 10.0),
        )

    def _normalize_model_entries(self, models_cfg: dict) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        model_mappings = models_cfg.get('model_mappings', {})
        if isinstance(model_mappings, dict):
            for task_name, task_models in model_mappings.items():
                if not isinstance(task_models, dict):
                    continue
                for algo_id, model_path in task_models.items():
                    normalized.append({
                        'id': str(algo_id),
                        'name': str(algo_id),
                        'path': str(model_path),
                        'enabled': True,
                        'task': str(task_name),
                        'conf_threshold': float(models_cfg.get('default_conf_threshold', 0.5)),
                        'device': self.inference_device,
                    })
            if normalized:
                return normalized

        model_list = models_cfg.get('models', [])
        if isinstance(model_list, list):
            for index, item in enumerate(model_list):
                if not isinstance(item, dict):
                    continue
                mid = str(item.get('id') or item.get('name') or f'model_{index + 1}')
                normalized.append({
                    'id': mid,
                    'name': item.get('name', mid),
                    'path': item.get('path', ''),
                    'enabled': bool(item.get('enabled', True)),
                    'task': item.get('task', 'detection'),
                    'conf_threshold': float(item.get('conf_threshold', models_cfg.get('default_conf_threshold', 0.5))),
                    'device': str(item.get('device', self.inference_device)),
                })
        return normalized

    def _load_model_config(self):
        models = self.config.get('models', {})
        self.default_conf_threshold = float(models.get('default_conf_threshold', 0.5))
        self.model_definitions = self._normalize_model_entries(models)
        self.model_mappings = {
            'detection': {
                item['id']: item['path']
                for item in self.model_definitions
                if item.get('task') == 'detection' and item.get('path')
            }
        }

        # 多模型独立配置：每个模型的类别过滤列表
        # 格式: {"3001": ["guanche"], "3099": []}
        self.model_class_filters = models.get('model_class_filters', {})

        # 多模型独立配置：每个模型的推理间隔
        # 格式: {"3001": 3, "3099": 5}
        self.model_intervals = models.get('model_intervals', {})

    def _load_class_config(self):
        classes = self.config.get('classes', {})
        filtered = classes.get('filtered_classes', {}) if isinstance(classes, dict) else {}
        # 2026-06-16 16:49 修改目的：类别过滤改为全局类别名配置，不再按算法ID拆分。
        self.detection_filtered_class_names = filtered.get('detection_filtered_class_names', [])

    def _load_performance_config(self):
        perf = self.config.get('performance', {})
        self.detection_inference_interval = max(1, int(perf.get('detection_inference_interval', 1)))
        self.result_max_back_frames = max(30, int(perf.get('result_max_back_frames', 30)))
        self.inference_idle_interval = max(
            self.detection_inference_interval,
            int(perf.get('inference_idle_interval', max(self.detection_inference_interval * 3, 9))),
        )
        self.inference_active_hold_seconds = float(perf.get('inference_active_hold_seconds', 2.0) or 2.0)
        self.motion_detection_enabled = bool(perf.get('motion_detection_enabled', True))
        self.motion_threshold = float(perf.get('motion_threshold', 3.5) or 3.5)
        self.motion_resize_width = max(32, int(perf.get('motion_resize_width', 160) or 160))
        self.motion_resize_height = max(18, int(perf.get('motion_resize_height', 90) or 90))
        self.inference_batch_size = max(1, int(perf.get('inference_batch_size', 4) or 4))
        self.inference_batch_wait_ms = max(0, int(perf.get('inference_batch_wait_ms', 8) or 8))
        self.max_infer_result_age = float(perf.get('max_infer_result_age', 1.0) or 1.0)
        self.max_infer_frame_lag = max(1, int(perf.get('max_infer_frame_lag', 5) or 5))

    def _load_alarm_config(self):
        alarm = self.config.get('alarm', {})
        self.alarm_target_threshold = alarm.get('target_threshold', 1)
        self.alarm_interval_seconds = alarm.get('interval_seconds', 10.0)
        self.video_clip_seconds = alarm.get('video_clip_seconds', 10)
        self.video_buffer_seconds = alarm.get('video_buffer_seconds', 12)
        self.video_pre_alert_seconds = alarm.get('video_pre_alert_seconds', 5)
        self.video_post_alert_seconds = alarm.get('video_post_alert_seconds', 5)
        self.save_raw_image = alarm.get('save_raw_image', True)

        # 报警等级模式开关：true=空间分级（红1/黄2/橙3），false=固定等级
        self.use_spatial_level = bool(alarm.get('use_spatial_level', True))
        # 固定报警等级（use_spatial_level=false 时使用，默认为 1）
        self.fixed_alarm_level = int(alarm.get('fixed_alarm_level', 1))

    def _load_ppe_config(self):
        """加载 PPE（安全帽/反光衣）检测配置"""
        ppe = self.config.get('ppe', {})
        self.ppe_enabled = bool(ppe.get('enabled', False))
        self.ppe_config = ppe if self.ppe_enabled else {}

    def _load_video_config(self):
        video = self.config.get('video', {})
        self.target_width = int(video.get('target_width', 1920))
        self.target_height = int(video.get('target_height', 1080))
        self.default_width = int(video.get('default_width', 1920))
        self.default_height = int(video.get('default_height', 1080))
        self.fps = max(1, int(video.get('fps', 25)))
        self.push_fps = max(1, int(video.get('push_fps', self.fps)))
        self.bitrate = video.get('bitrate', '4M')
        self.max_bitrate = video.get('max_bitrate', '6M')
        self.buffer_size = video.get('buffer_size', '8M')
        self.gop_size = max(1, int(video.get('gop_size', 50)))
        self.encoding_preset = video.get('encoding_preset', 'p4')
        self.auto_detect_resolution = bool(video.get('auto_detect_resolution', True))
        self.push_enabled = bool(video.get('push_enabled', True))
        self.push_restart_backoff_initial = max(0.1, float(video.get('push_restart_backoff_initial', 0.3) or 0.3))
        self.push_restart_backoff_max = max(
            self.push_restart_backoff_initial,
            float(video.get('push_restart_backoff_max', 10.0) or 10.0),
        )
        self.push_max_repeat_frames = max(1, int(video.get('push_max_repeat_frames', 600) or 600))
        self.push_stale_repeat_window = max(1.0, float(video.get('push_stale_repeat_window', 30.0) or 30.0))
        codec_str = str(video.get('push_codec', 'auto')).lower()
        if self.push_device_mode == DeviceMode.CPU and codec_str in {VideoCodec.AUTO.value, VideoCodec.H264_NVENC.value}:
            codec_str = VideoCodec.LIBX264.value
        elif self.push_device_mode == DeviceMode.GPU and codec_str == VideoCodec.AUTO.value:
            codec_str = VideoCodec.H264_NVENC.value
        try:
            self.video_codec = VideoCodec(codec_str)
        except ValueError:
            self.video_codec = VideoCodec.AUTO

    def _load_application_config(self):
        app = self.config.get('application', {})
        self.output_directory = app.get('output_directory', './res')
        self.error_retry_interval = float(app.get('error_retry_interval', 5.0))

    def _load_minio_config(self):
        minio = self.config.get('minio', {})
        self.minio_endpoint = str(minio.get('endpoint', ''))
        self.minio_access_key = str(minio.get('access_key', ''))
        self.minio_secret_key = str(minio.get('secret_key', ''))
        self.minio_secure = bool(minio.get('secure', False))
        self.minio_bucket_name = str(minio.get('bucket_name', ''))

    def _load_platform_config(self):
        platform = self.config.get('platform_api', {})
        self.platform_base_url = str(platform.get('base_url', '')).rstrip('/')
        self.platform_username = str(platform.get('username', ''))
        self.platform_password = str(platform.get('password', ''))
        self.platform_captcha = str(platform.get('captcha', 'e1a709144444b0800585121bb9272318'))
        self.platform_check_key = str(platform.get('checkKey', 'e352eaa3205126f521c027784ce82baf'))
        self.platform_vendor_id = int(platform.get('vendor_id', 0) or 0)
        self.platform_device_type = int(platform.get('device_type', 1) or 1)
        self.platform_task_type = int(platform.get('task_type', 1) or 1)
        self.platform_report_enabled = bool(platform.get('report_enabled', False))
        self.platform_login_timeout = float(platform.get('login_timeout', 10.0) or 10.0)
        self.platform_report_timeout = float(platform.get('report_timeout', 10.0) or 10.0)

    def validate(self) -> List[str]:
        """
        验证配置完整性。

        Returns:
            警告/错误信息列表，空列表表示配置正确
        """
        warnings = []

        # 检查模型文件
        for model in self.model_definitions:
            model_path = model.get('path', '')
            if model_path and not os.path.exists(model_path):
                warnings.append(f"模型文件不存在: [{model['id']}] {model_path}")

        # 检查 PPE 配置
        if self.ppe_enabled:
            ppe_detection = self.ppe_config.get('detection', {})
            ppe_attr = self.ppe_config.get('attribute', {})

            # 检查 PPE 属性分类模型
            attr_model_path = ppe_attr.get('model_path', '')
            if attr_model_path and not os.path.exists(attr_model_path):
                warnings.append(f"PPE 属性分类模型不存在: {attr_model_path}")

            # 检查 PPE 配置完整性
            ppe_model_id = ppe_detection.get('model_id', '')
            if not ppe_model_id:
                warnings.append("PPE 配置缺少 detection.model_id")
            else:
                # 检查 model_id 是否在 model_mappings 中注册
                registered_ids = {m.get('id') for m in self.model_definitions}
                if ppe_model_id not in registered_ids:
                    warnings.append(f"PPE detection.model_id '{ppe_model_id}' 未在 model_mappings 中注册")

        # 检查 tracker 配置
        tracker_path = self.tracking_tracker
        if tracker_path and not os.path.exists(tracker_path):
            # tracker 文件可能在 ultralytics 包内，只做警告
            warnings.append(f"ByteTrack 配置文件不存在（可能在 ultralytics 包内）: {tracker_path}")

        # 检查流配置
        streams = self.config.get('streams', [])
        if not streams:
            warnings.append("未配置任何摄像头流")

        for i, stream in enumerate(streams):
            if not stream.get('input_url') and not stream.get('rtsp_url'):
                warnings.append(f"流 {i} 缺少 input_url 或 rtsp_url")

        return warnings

    def get_video_codec(self) -> str:
        return self.video_codec.value

    def is_hardware_encoding_enabled(self) -> bool:
        return self.video_codec == VideoCodec.H264_NVENC

    def is_auto_codec_enabled(self) -> bool:
        return self.video_codec == VideoCodec.AUTO

    def get_pull_device(self) -> str:
        return self.pull_device_mode.value

    def get_push_device(self) -> str:
        return self.push_device_mode.value

    def prefers_gpu_pull(self) -> bool:
        return self.pull_device_mode == DeviceMode.GPU

    def prefers_cpu_pull(self) -> bool:
        return self.pull_device_mode == DeviceMode.CPU

    def prefers_auto_pull(self) -> bool:
        return self.pull_device_mode == DeviceMode.AUTO

    def prefers_gpu_push(self) -> bool:
        return self.push_device_mode == DeviceMode.GPU

    def prefers_cpu_push(self) -> bool:
        return self.push_device_mode == DeviceMode.CPU

    def get_alarm_config(self) -> dict:
        return {
            'alarm_target_threshold': self.alarm_target_threshold,
            'alarm_interval_seconds': self.alarm_interval_seconds,
            'video_clip_seconds': self.video_clip_seconds,
            'video_buffer_seconds': self.video_buffer_seconds,
            'video_pre_alert_seconds': self.video_pre_alert_seconds,
            'video_post_alert_seconds': self.video_post_alert_seconds,
            'save_raw_image': self.save_raw_image,
        }

    def _build_capture_options(self, stream_url: str) -> str:
        return build_capture_options(stream_url, self.pull_device_mode)

    def get_capture_options(self, stream_url: str) -> str:
        return self._build_capture_options(stream_url)

    def _open_capture(self, stream_url: str):
        import cv2
        options = self._build_capture_options(stream_url)
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = options
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '0')
        os.environ.setdefault('OPENCV_FFMPEG_READ_ATTEMPTS', '20000')
        os.environ.setdefault('OPENCV_FFMPEG_DECODE_ATTEMPTS', '20000')
        with self._suppress_capture_backend_logs():
            if hasattr(cv2, 'CAP_FFMPEG'):
                return cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            return cv2.VideoCapture(stream_url)

    def get_target_resolution(self) -> tuple:
        return (self.target_width, self.target_height)

    def get_default_resolution(self) -> tuple:
        return (self.default_width, self.default_height)

    def get_enabled_models(self) -> List[Dict[str, Any]]:
        return [item for item in self.model_definitions if item.get('enabled', True) and item.get('path')]

    def auto_detect_and_update_resolution(self, stream_url: str) -> tuple:
        import cv2
        import logging
        if not self.auto_detect_resolution:
            return self.get_target_resolution()
        cap = None
        try:
            options = self._build_capture_options(stream_url)
            logging.info(f"分辨率探测拉流参数: {options}")
            cap = self._open_capture(stream_url)
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    self.target_width = w
                    self.target_height = h
                    logging.info(f"自动检测到分辨率: {w}x{h}")
                    return (w, h)
        except Exception as e:
            logging.error(f"分辨率检测失败: {e}")
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        return self.get_default_resolution()
