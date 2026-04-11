# -*- coding: utf-8 -*-
"""
Capture subprocess backed by FFmpeg.

The subprocess decodes the input stream into raw BGR frames and shares the
latest frame with the parent process through shared memory.
"""

import logging
import multiprocessing
import multiprocessing.shared_memory as shm_mod
import os
import psutil
import signal
import subprocess
import threading
import time
from typing import Dict, Optional
from urllib.parse import urlparse

import numpy as np

_CMD_STOP = 'stop'
_CMD_PING = 'ping'

_EVT_FRAME = 'frame'
_EVT_PONG = 'pong'
_EVT_STATUS = 'status'


def _parse_capture_options(capture_options: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    if not capture_options:
        return options
    for item in capture_options.split('|'):
        item = item.strip()
        if not item or ';' not in item:
            continue
        key, value = item.split(';', 1)
        key = key.strip()
        value = value.strip()
        if key:
            options[key] = value
    return options


def _capture_worker(
    stream_id: str,
    stream_url: str,
    pull_device: str,
    capture_options: str,
    frame_width: int,
    frame_height: int,
    ctrl_conn,
    frame_conn,
    main_pid: int,
):
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(levelname)s] [cap-{stream_id}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f"拉流子进程启动: stream={stream_id} pid={os.getpid()}")

    stop_event = threading.Event()
    ffmpeg_state = {'proc': None}
    shm: Optional[shm_mod.SharedMemory] = None
    shm_shape = None
    reconnect_delay = 5.0
    frame_width = max(1, int(frame_width or 1))
    frame_height = max(1, int(frame_height or 1))
    frame_bytes = frame_width * frame_height * 3
    prefer_hwaccel = str(pull_device).lower() == 'gpu'

    def _watch_parent():
        while not stop_event.is_set():
            time.sleep(2.0)
            try:
                os.kill(main_pid, 0)
            except (ProcessLookupError, PermissionError):
                logging.warning(f"主进程 {main_pid} 已退出，拉流子进程自动退出")
                stop_event.set()
                proc = ffmpeg_state.get('proc')
                if proc is not None:
                    _terminate_ffmpeg(proc)
                os._exit(0)
            except Exception:
                pass

    def _ensure_shm(frame: np.ndarray) -> bool:
        nonlocal shm, shm_shape
        needed = frame.nbytes
        if shm is not None:
            if shm.size >= needed and shm_shape == frame.shape:
                return True
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
            shm = None
            shm_shape = None
        try:
            shm = shm_mod.SharedMemory(create=True, size=max(needed, 1))
            shm_shape = frame.shape
            return True
        except Exception as e:
            logging.error(f"创建共享内存失败: {e}")
            return False

    def _send_status(status: str, **kwargs):
        try:
            payload = {'evt': _EVT_STATUS, 'status': status}
            payload.update(kwargs)
            frame_conn.send(payload)
        except Exception:
            pass

    def _build_ffmpeg_cmd(use_hwaccel: bool):
        scheme = (urlparse(stream_url).scheme or '').lower()
        option_map = _parse_capture_options(capture_options)
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostats']

        if scheme == 'rtsp' and 'rtsp_transport' not in option_map:
            cmd += ['-rtsp_transport', 'tcp']

        supported_input_options = {
            'hwaccel',
            'rtsp_transport',
            'reorder_queue_size',
            'buffer_size',
            'max_delay',
            'analyzeduration',
            'probesize',
            'fflags',
            'flags',
            'rw_timeout',
            'timeout',
        }
        for key, value in option_map.items():
            mapped_key = 'rw_timeout' if key == 'stimeout' else key
            if mapped_key == 'hwaccel_output_format':
                continue
            if mapped_key == 'hwaccel' and not use_hwaccel:
                continue
            if mapped_key in supported_input_options:
                cmd += [f'-{mapped_key}', value]

        if use_hwaccel and 'hwaccel' not in option_map:
            cmd += ['-hwaccel', 'cuda']

        cmd += [
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-i', stream_url,
            '-map', '0:v:0',
            '-an',
            '-sn',
            '-dn',
            '-vf', f'scale={frame_width}:{frame_height}',
            '-pix_fmt', 'bgr24',
            '-f', 'rawvideo',
            'pipe:1',
        ]
        return cmd

    def _terminate_ffmpeg(proc: Optional[subprocess.Popen]):
        if proc is None:
            return
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if os.name != 'nt':
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                if os.name != 'nt':
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        if ffmpeg_state.get('proc') is proc:
            ffmpeg_state['proc'] = None

    def _read_ffmpeg_error(proc: Optional[subprocess.Popen]) -> str:
        if proc is None or proc.stderr is None:
            return ''
        try:
            if proc.poll() is None:
                return ''
            return proc.stderr.read().decode('utf-8', errors='ignore').strip()[:1500]
        except Exception:
            return ''

    def _control_loop():
        while not stop_event.is_set():
            try:
                if not ctrl_conn.poll(0.2):
                    continue
                msg = ctrl_conn.recv()
            except EOFError:
                stop_event.set()
                break
            except Exception:
                continue

            cmd = msg.get('cmd')
            if cmd == _CMD_STOP:
                stop_event.set()
                proc = ffmpeg_state.get('proc')
                if proc is not None:
                    _terminate_ffmpeg(proc)
                break
            if cmd == _CMD_PING:
                try:
                    frame_conn.send({'evt': _EVT_PONG})
                except Exception:
                    pass

    def _read_exact(stream, size: int) -> Optional[bytes]:
        buf = bytearray()
        remaining = size
        while remaining > 0 and not stop_event.is_set():
            try:
                chunk = stream.read(min(remaining, 1024 * 1024))
            except Exception as e:
                logging.warning(f"ffmpeg 读帧异常: {e}")
                return None
            if not chunk:
                return None
            buf.extend(chunk)
            remaining -= len(chunk)
        if stop_event.is_set():
            return None
        return bytes(buf)

    threading.Thread(target=_watch_parent, daemon=True).start()
    threading.Thread(target=_control_loop, daemon=True).start()

    while not stop_event.is_set():
        _send_status('connecting')

        proc = None
        launch_error = ''
        launch_modes = [prefer_hwaccel, False] if prefer_hwaccel else [False]
        for use_hwaccel in launch_modes:
            mode_name = 'gpu' if use_hwaccel else 'cpu'
            ffmpeg_cmd = _build_ffmpeg_cmd(use_hwaccel=use_hwaccel)
            logging.info(f"FFmpeg拉流命令(mode={mode_name}): {' '.join(ffmpeg_cmd)}")
            try:
                proc = subprocess.Popen(
                    ffmpeg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    stdin=subprocess.DEVNULL,
                    bufsize=0,
                    start_new_session=(os.name != 'nt'),
                )
                ffmpeg_state['proc'] = proc
            except Exception as e:
                launch_error = str(e)
                logging.error(f"打开流失败(mode={mode_name}): {e}")
                proc = None
                continue

            time.sleep(0.5)
            if proc.poll() is None:
                prefer_hwaccel = use_hwaccel
                break

            launch_error = _read_ffmpeg_error(proc) or 'cannot_open'
            logging.warning(f"无法打开流(mode={mode_name})，准备切换: {launch_error}")
            _terminate_ffmpeg(proc)
            proc = None

        if proc is None:
            _send_status('error', reason=launch_error or 'cannot_open')
            time.sleep(reconnect_delay)
            continue

        first_frame = True
        while not stop_event.is_set():
            if proc.stdout is None:
                _send_status('interrupted', reason='ffmpeg_stdout_missing')
                break

            frame_buf = _read_exact(proc.stdout, frame_bytes)
            if frame_buf is None:
                if stop_event.is_set():
                    break
                err = _read_ffmpeg_error(proc)
                reason = err or ('ffmpeg_exit' if proc.poll() is not None else 'frame_read_failed')
                if 'Impossible to convert' in reason or 'cuda' in reason.lower():
                    prefer_hwaccel = False
                    logging.warning("检测到GPU拉流格式不兼容，后续自动回退CPU拉流")
                logging.warning(f"FFmpeg拉流中断，准备重连: {reason}")
                _send_status('interrupted', reason=reason)
                break

            frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape((frame_height, frame_width, 3))
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if not _ensure_shm(frame):
                continue

            if first_frame:
                _send_status('connected')
                logging.info("连接成功")
                logging.info(f"首帧 size={frame_width}x{frame_height}")
                first_frame = False

            dst = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
            np.copyto(dst, frame)
            try:
                frame_conn.send({
                    'evt': _EVT_FRAME,
                    'shm_name': shm.name,
                    'shape': frame.shape,
                    'dtype': str(frame.dtype),
                    'ts': time.time(),
                })
            except Exception:
                logging.error("发送帧通知失败")
                break

        _terminate_ffmpeg(proc)
        if not stop_event.is_set():
            _send_status('reconnecting')
            time.sleep(reconnect_delay)

    if shm is not None:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    logging.info(f"拉流子进程退出: stream={stream_id}")


class CaptureProxy:
    def __init__(
        self,
        stream_id: str,
        stream_url: str,
        pull_device: str,
        capture_options: str,
        frame_width: int,
        frame_height: int,
        frame_callback=None,
        status_callback=None,
    ):
        self._stream_id = stream_id
        self._stream_url = stream_url
        self._pull_device = pull_device
        self._capture_options = capture_options
        self._frame_width = max(1, int(frame_width or 1))
        self._frame_height = max(1, int(frame_height or 1))
        self._frame_callback = frame_callback
        self._status_callback = status_callback

        self._ctrl_conn = None
        self._frame_conn = None
        self._process: Optional[multiprocessing.Process] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._dispatch_thread: Optional[threading.Thread] = None
        self._running = False

        self._shm_cache: Optional[shm_mod.SharedMemory] = None
        self._shm_cache_name: Optional[str] = None
        self._latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self._frame_event = threading.Event()

    def start(self):
        ctx = multiprocessing.get_context('forkserver')

        ctrl_child_read, ctrl_parent_write = ctx.Pipe(duplex=False)
        frame_parent_read, frame_child_write = ctx.Pipe(duplex=False)

        self._ctrl_conn = ctrl_parent_write
        self._frame_conn = frame_parent_read

        self._process = ctx.Process(
            target=_capture_worker,
            args=(
                self._stream_id,
                self._stream_url,
                self._pull_device,
                self._capture_options,
                self._frame_width,
                self._frame_height,
                ctrl_child_read,
                frame_child_write,
                os.getpid(),
            ),
            daemon=True,
            name=f"CapProc-{self._stream_id}",
        )
        self._process.start()
        ctrl_child_read.close()
        frame_child_write.close()

        logging.info(f"[{self._stream_id}] 拉流子进程已启动 pid={self._process.pid}")

        try:
            self._ctrl_conn.send({'cmd': _CMD_PING})
            if self._frame_conn.poll(30.0):
                resp = self._frame_conn.recv()
                if resp.get('evt') == _EVT_PONG:
                    logging.info(f"[{self._stream_id}] 拉流子进程就绪")
        except Exception as e:
            logging.error(f"[{self._stream_id}] 拉流子进程握手失败: {e}")

        self._running = True
        self._reader_thread = threading.Thread(
            target=self._read_loop,
            daemon=True,
            name=f"CapReader-{self._stream_id}",
        )
        self._reader_thread.start()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name=f"CapDispatch-{self._stream_id}",
        )
        self._dispatch_thread.start()

    def _read_loop(self):
        while self._running:
            try:
                if not self._frame_conn.poll(1.0):
                    continue
                msg = self._frame_conn.recv()
            except EOFError:
                logging.warning(f"[{self._stream_id}] 拉流子进程 Pipe 已关闭")
                break
            except Exception:
                continue

            evt = msg.get('evt')
            if evt == _EVT_FRAME:
                shm_name = msg.get('shm_name')
                shape = msg.get('shape')
                dtype = msg.get('dtype')
                if shm_name and shape and dtype:
                    frame = self._read_shm_frame(shm_name, shape, dtype)
                    if frame is not None:
                        with self._latest_frame_lock:
                            self._latest_frame = frame
                        self._frame_event.set()
            elif evt == _EVT_STATUS and self._status_callback:
                try:
                    self._status_callback(self._stream_id, msg.get('status', ''))
                except Exception:
                    pass

    def _dispatch_loop(self):
        while self._running:
            self._frame_event.wait(1.0)
            frame = None
            with self._latest_frame_lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame
                    self._latest_frame = None
                self._frame_event.clear()
            if frame is None:
                continue
            if self._frame_callback:
                try:
                    self._frame_callback(self._stream_id, frame)
                except Exception as e:
                    logging.error(f"[{self._stream_id}] 帧回调异常: {e}")

    def _read_shm_frame(self, shm_name: str, shape, dtype) -> Optional[np.ndarray]:
        try:
            if self._shm_cache is not None and self._shm_cache_name != shm_name:
                try:
                    self._shm_cache.close()
                except Exception:
                    pass
                self._shm_cache = None
                self._shm_cache_name = None

            if self._shm_cache is None:
                self._shm_cache = shm_mod.SharedMemory(name=shm_name)
                self._shm_cache_name = shm_name

            return np.ndarray(shape, dtype=dtype, buffer=self._shm_cache.buf).copy()
        except Exception as e:
            logging.error(f"[{self._stream_id}] 读取共享内存帧失败: {e}")
            self._shm_cache = None
            self._shm_cache_name = None
            return None

    def stop(self):
        self._running = False
        self._frame_event.set()

        if self._process is not None and self._process.is_alive():
            try:
                self._ctrl_conn.send({'cmd': _CMD_STOP})
            except Exception:
                pass
            self._process.join(timeout=8.0)
            if self._process.is_alive():
                self._kill_process_descendants()
                self._process.terminate()
                self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._kill_process_descendants()
                self._process.kill()
                self._process.join(timeout=2.0)
            logging.info(f"[{self._stream_id}] 拉流子进程已停止")

        self._process = None

        for conn in [self._ctrl_conn, self._frame_conn]:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        self._ctrl_conn = None
        self._frame_conn = None

        if self._shm_cache is not None:
            try:
                self._shm_cache.close()
            except Exception:
                pass
            self._shm_cache = None
            self._shm_cache_name = None

    def _kill_process_descendants(self):
        if self._process is None:
            return
        try:
            parent = psutil.Process(self._process.pid)
        except Exception:
            return

        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
