# -*- coding: utf-8 -*-
"""
Capture subprocess that uses FFmpeg to decode network streams into raw BGR frames.
Frames are passed back through shared memory and control messages use Pipe.
"""

import logging
import multiprocessing
import multiprocessing.shared_memory as shm_mod
import os
import select
import subprocess
import time
from typing import Optional
from urllib.parse import urlparse

import numpy as np

_CMD_STOP = 'stop'
_CMD_PING = 'ping'

_EVT_FRAME = 'frame'
_EVT_PONG = 'pong'
_EVT_STATUS = 'status'


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
    import threading

    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(levelname)s] [cap-{stream_id}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f"拉流子进程启动: stream={stream_id} pid={os.getpid()}")

    def _watch_parent():
        while True:
            time.sleep(2.0)
            try:
                os.kill(main_pid, 0)
            except (ProcessLookupError, PermissionError):
                logging.warning(f"主进程 {main_pid} 已退出，拉流子进程自动退出")
                os._exit(0)
            except Exception:
                pass

    threading.Thread(target=_watch_parent, daemon=True).start()

    shm: Optional[shm_mod.SharedMemory] = None
    shm_shape = None
    stop_flag = False
    reconnect_delay = 5.0
    frame_timeout = 10.0
    frame_width = max(1, int(frame_width or 1))
    frame_height = max(1, int(frame_height or 1))
    frame_bytes = frame_width * frame_height * 3

    def _ensure_shm(frame: np.ndarray):
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
        try:
            shm = shm_mod.SharedMemory(create=True, size=max(needed, 1))
            shm_shape = frame.shape
            return True
        except Exception as e:
            logging.error(f"创建共享内存失败: {e}")
            return False

    def _send_status(status: str, **kwargs):
        try:
            msg = {'evt': _EVT_STATUS, 'status': status}
            msg.update(kwargs)
            frame_conn.send(msg)
        except Exception:
            pass

    def _build_ffmpeg_cmd():
        scheme = (urlparse(stream_url).scheme or '').lower()
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-nostats']
        if scheme == 'rtsp':
            cmd += ['-rtsp_transport', 'tcp']
        cmd += [
            '-i', stream_url,
            '-map', '0:v:0',
            '-an', '-sn', '-dn',
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
            if proc.stderr:
                proc.stderr.close()
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=3.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass

    def _read_ffmpeg_error(proc: Optional[subprocess.Popen]) -> str:
        if proc is None or proc.stderr is None:
            return ''
        try:
            if proc.poll() is None:
                return ''
            return proc.stderr.read().decode('utf-8', errors='ignore').strip()[:500]
        except Exception:
            return ''

    while not stop_flag:
        try:
            while ctrl_conn.poll(0):
                msg = ctrl_conn.recv()
                cmd = msg.get('cmd')
                if cmd == _CMD_STOP:
                    stop_flag = True
                    break
                if cmd == _CMD_PING:
                    try:
                        frame_conn.send({'evt': _EVT_PONG})
                    except Exception:
                        pass
        except EOFError:
            break
        except Exception:
            pass

        if stop_flag:
            break

        _send_status('connecting')
        ffmpeg_cmd = _build_ffmpeg_cmd()
        logging.info(f"FFmpeg拉流命令: {' '.join(ffmpeg_cmd)}")

        proc = None
        try:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=frame_bytes,
            )
        except Exception as e:
            logging.error(f"打开流失败: {e}")
            _send_status('error', reason=str(e))
            time.sleep(reconnect_delay)
            continue

        time.sleep(0.5)
        if proc.poll() is not None:
            err = _read_ffmpeg_error(proc)
            logging.warning(f"无法打开流，等待重连: {err}")
            _send_status('error', reason=err or 'cannot_open')
            _terminate_ffmpeg(proc)
            time.sleep(reconnect_delay)
            continue

        try:
            if proc.stdout is not None:
                os.set_blocking(proc.stdout.fileno(), False)
        except Exception:
            pass

        consecutive_failures = 0
        first_frame = True
        pending = bytearray()
        last_data_ts = time.time()

        while not stop_flag:
            try:
                while ctrl_conn.poll(0):
                    msg = ctrl_conn.recv()
                    if msg.get('cmd') == _CMD_STOP:
                        stop_flag = True
                        break
            except EOFError:
                stop_flag = True
                break
            except Exception:
                pass

            if stop_flag:
                break

            try:
                raw = b''
                if proc and proc.stdout:
                    ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                    if ready:
                        chunk_size = max(frame_bytes - len(pending), 65536)
                        raw = os.read(proc.stdout.fileno(), chunk_size)
            except Exception as e:
                logging.warning(f"ffmpeg.read() 异常: {e}")
                break

            if raw:
                pending.extend(raw)
                last_data_ts = time.time()

            if len(pending) < frame_bytes:
                if proc is not None and proc.poll() is not None:
                    err = _read_ffmpeg_error(proc)
                    logging.warning(f"FFmpeg拉流进程退出，准备重连: {err}")
                    _send_status('interrupted', reason=err or 'ffmpeg_exit')
                    break
                if (time.time() - last_data_ts) >= frame_timeout:
                    logging.warning(f"{frame_timeout:.1f}s 未收到新视频数据，准备重连")
                    _send_status('interrupted', reason='frame_timeout')
                    break
                time.sleep(0.05)
                continue

            frame_buf = bytes(pending[:frame_bytes])
            del pending[:frame_bytes]
            frame = np.frombuffer(frame_buf, dtype=np.uint8).reshape((frame_height, frame_width, 3))

            if first_frame:
                _send_status('connected')
                logging.info("连接成功")
                logging.info(f"首帧 size={frame_width}x{frame_height}")
                first_frame = False

            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if not _ensure_shm(frame):
                continue

            dst = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
            np.copyto(dst, frame)

            try:
                frame_conn.send({
                    'evt': _EVT_FRAME,
                    'shm_name': shm.name,
                    'shape': frame.shape,
                    'dtype': str(frame.dtype),
                })
            except Exception:
                logging.error("发送帧通知失败")
                break

        _terminate_ffmpeg(proc)

        if not stop_flag:
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
        self._reader_thread: Optional[object] = None
        self._running = False

        self._shm_cache: Optional[shm_mod.SharedMemory] = None
        self._shm_cache_name: Optional[str] = None

    def start(self):
        import threading

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
                    if frame is not None and self._frame_callback:
                        try:
                            self._frame_callback(self._stream_id, frame)
                        except Exception as e:
                            logging.error(f"[{self._stream_id}] 帧回调异常: {e}")

            elif evt == _EVT_STATUS:
                status = msg.get('status', '')
                if self._status_callback:
                    try:
                        self._status_callback(self._stream_id, status)
                    except Exception:
                        pass

    def _read_shm_frame(self, shm_name: str, shape, dtype) -> Optional[np.ndarray]:
        try:
            if self._shm_cache is not None and self._shm_cache_name != shm_name:
                try:
                    self._shm_cache.close()
                except Exception:
                    pass
                self._shm_cache = None

            if self._shm_cache is None:
                self._shm_cache = shm_mod.SharedMemory(name=shm_name)
                self._shm_cache_name = shm_name

            return np.ndarray(shape, dtype=dtype, buffer=self._shm_cache.buf).copy()
        except Exception as e:
            logging.error(f"[{self._stream_id}] 读取拉流共享内存失败: {e}")
            self._shm_cache = None
            self._shm_cache_name = None
            return None

    def stop(self):
        self._running = False
        if self._process is not None and self._process.is_alive():
            try:
                self._ctrl_conn.send({'cmd': _CMD_STOP})
            except Exception:
                pass
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=3.0)
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
