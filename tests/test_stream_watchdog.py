import threading
from pathlib import Path
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


sys.modules.setdefault('cv2', types.ModuleType('cv2'))

config_pkg = sys.modules.setdefault('config', types.ModuleType('config'))
algorithm_config = _install_module(
    'config.algorithm_config',
    CameraConfig=type('CameraConfig', (), {}),
    VideoCodec=type('VideoCodec', (), {}),
)
config_manager = _install_module(
    'config.config_manager',
    ConfigManager=type('ConfigManager', (), {}),
)
config_pkg.algorithm_config = algorithm_config
config_pkg.config_manager = config_manager

nan_pkg = sys.modules.setdefault('nan', types.ModuleType('nan'))
logger_config = _install_module('nan.logger_config', setup_logging=lambda *args, **kwargs: None)
nan_pkg.logger_config = logger_config

stream_pkg = sys.modules.setdefault('stream', types.ModuleType('stream'))
enhanced_video_processor = _install_module(
    'stream.enhanced_video_processor',
    EnhancedVideoStreamProcessor=type('EnhancedVideoStreamProcessor', (), {}),
    VideoStreamConfig=type('VideoStreamConfig', (), {}),
    VideoStreamStatus=type(
        'VideoStreamStatus',
        (),
        {
            'CONNECTING': SimpleNamespace(value='connecting'),
            'CONNECTED': SimpleNamespace(value='connected'),
            'READING': SimpleNamespace(value='reading'),
            'INTERRUPTED': SimpleNamespace(value='interrupted'),
            'RECONNECTING': SimpleNamespace(value='reconnecting'),
            'ERROR': SimpleNamespace(value='error'),
        },
    ),
)
capture_process = _install_module('stream.capture_process', CaptureProxy=type('CaptureProxy', (), {}))
stream_pkg.enhanced_video_processor = enhanced_video_processor
stream_pkg.capture_process = capture_process

inference_pkg = sys.modules.setdefault('inference', types.ModuleType('inference'))
unified_scheduler = _install_module(
    'inference.unified_scheduler',
    UnifiedInferenceScheduler=type('UnifiedInferenceScheduler', (), {}),
)
inference_pkg.unified_scheduler = unified_scheduler

alert_pkg = sys.modules.setdefault('alert', types.ModuleType('alert'))
alert_system = _install_module(
    'alert.alert_system',
    AlertSystem=type('AlertSystem', (), {}),
    create_count_threshold_rule=lambda *args, **kwargs: None,
    AlertLevel=type('AlertLevel', (), {}),
)
alert_pkg.alert_system = alert_system

tracking_pkg = sys.modules.setdefault('tracking', types.ModuleType('tracking'))
simple_tracker = _install_module('tracking.simple_tracker', SimpleTracker=type('SimpleTracker', (), {}))
tracking_pkg.simple_tracker = simple_tracker

utils_pkg = sys.modules.setdefault('utils', types.ModuleType('utils'))
alarm_level = _install_module(
    'utils.alarm_level',
    classify_point_alarm_level_uv_details=lambda *args, **kwargs: None,
)
utils_pkg.alarm_level = alarm_level

from camera import CameraStreamManager, StreamProcessor


def _dummy_stream(**kwargs):
    base = {
        'is_running': True,
        '_stop_event': threading.Event(),
        'capture_proxy': object(),
        '_last_frame_ts': 0.0,
        '_last_processed_ts': 0.0,
        '_last_push_ts': 0.0,
        '_capture_session_started_ts': 0.0,
        '_capture_start_timeout': 20.0,
        '_capture_stall_timeout': 15.0,
        '_capture_watchdog_interval': 5.0,
        '_last_capture_status': 'connecting',
        'output_url': 'rtmp://example/ai/test',
        '_restart_calls': [],
        '_restart_capture_proxy': None,
    }
    base.update(kwargs)
    stream = SimpleNamespace(**base)
    if stream._restart_capture_proxy is None:
        stream._restart_capture_proxy = lambda reason: stream._restart_calls.append(reason) or True
    return stream


def test_stream_active_during_startup_grace_without_first_frame():
    stream = _dummy_stream(
        _capture_session_started_ts=100.0,
        _last_capture_status='connecting',
        _last_push_ts=0.0,
    )

    snapshot = StreamProcessor.get_activity_snapshot(stream, now=110.0)

    assert snapshot['active'] is True
    assert snapshot['reason'] == ''


def test_stream_inactive_when_frames_stale_even_if_push_recent():
    stream = _dummy_stream(
        _last_frame_ts=100.0,
        _last_processed_ts=101.0,
        _last_push_ts=119.0,
        _last_capture_status='connected',
    )

    snapshot = StreamProcessor.get_activity_snapshot(stream, now=120.0)

    assert snapshot['active'] is False
    assert 'frame_age=20.0s' in snapshot['reason']


def test_watchdog_restarts_when_startup_has_no_first_frame():
    stream = _dummy_stream(
        _capture_session_started_ts=100.0,
        _last_capture_status='connecting',
    )

    restarted = StreamProcessor._maybe_restart_stale_capture(stream, now=121.0)

    assert restarted is True
    assert len(stream._restart_calls) == 1
    assert stream._restart_calls[0].startswith('startup_timeout')


def test_manager_activity_summary_uses_real_stream_activity():
    manager = CameraStreamManager.__new__(CameraStreamManager)
    manager._lock = threading.Lock()
    manager._processors = {
        'ok': SimpleNamespace(is_stream_active=lambda now=None: True),
        'bad': SimpleNamespace(is_stream_active=lambda now=None: False),
    }

    active, inactive = CameraStreamManager.get_activity_summary(manager)

    assert active == 1
    assert inactive == ['bad']
