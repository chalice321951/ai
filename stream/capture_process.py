# -*- coding: utf-8 -*-
"""
拉流子进程模块 - 每路流独立进程执行 cap.read()，彻底隔离 FFmpeg 全局状态
帧数据通过共享内存传回主进程，控制消息通过 Pipe 传递
"""
import logging
import multiprocessing
import multiprocessing.shared_memory as shm_mod
import numpy as np
import os
import time
import cv2
from urllib.parse import urlparse
from typing import Optional

_CMD_STOP = 'stop'
_CMD_PING = 'ping'

_EVT_FRAME = 'frame'
_EVT_PONG = 'pong'
_EVT_STATUS = 'status'
_EVT_ERROR = 'error'
_EVT_LOG = 'log'


def _capture_worker(
    stream_id: str,
    stream_url: str,
    pull_device: str,
    capture_options: str,
    ctrl_conn,      # 主进程 → 子进程 控制管道 (read端)
    frame_conn,     # 子进程 → 主进程 帧通知管道 (write端)
    main_pid: int,
):
    """拉流子进程入口"""
    import threading

    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(levelname)s] [cap-{stream_id}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.info(f"拉流子进程启动: stream={stream_id} pid={os.getpid()}")

    # 心跳检测主进程
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

    # 共享内存（子进程侧）
    shm: Optional[shm_mod.SharedMemory] = None
    shm_shape = None

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

    stop_flag = False
    reconnect_delay = 5.0

    while not stop_flag:
        # 检查控制管道
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

        # 打开连接
        _send_status('connecting')
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = capture_options
        os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
        os.environ.setdefault('OPENCV_FFMPEG_LOGLEVEL', '0')

        cap = None
        try:
            if hasattr(cv2, 'CAP_FFMPEG'):
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(stream_url)

            if hasattr(cv2, 'CAP_PROP_BUFFERSIZE'):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            logging.error(f"打开流失败: {e}")
            _send_status('error', reason=str(e))
            time.sleep(reconnect_delay)
            continue

        if not cap or not cap.isOpened():
            logging.warning("无法打开流，等待重连")
            _send_status('error', reason='cannot_open')
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            time.sleep(reconnect_delay)
            continue

        _send_status('connected')
        logging.info("连接成功")

        consecutive_failures = 0
        max_failures = 10
        first_frame = True

        # 读帧循环
        while not stop_flag:
            # 检查控制管道
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
                ret, frame = cap.read()
            except Exception as e:
                logging.warning(f"cap.read() 异常: {e}")
                break

            if not ret or frame is None or frame.size == 0:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logging.warning(f"连续 {consecutive_failures} 次读帧失败，重连")
                    _send_status('interrupted')
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            if first_frame:
                h, w = frame.shape[:2]
                logging.info(f"首帧 size={w}x{h}")
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

        # 释放
        if cap:
            try:
                cap.release()
            except Exception:
                pass

        if not stop_flag:
            _send_status('reconnecting')
            time.sleep(reconnect_delay)

    # 清理共享内存
    if shm is not None:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    logging.info(f"拉流子进程退出: stream={stream_id}")


# ──────────────────────────────────────────────
# 主进程侧代理
# ──────────────────────────────────────────────
class CaptureProxy:
    """
    主进程侧代理：管理一个拉流子进程的生命周期。
    帧通过共享内存传递，控制消息通过 Pipe 传递。
    """

    def __init__(self, stream_id: str, stream_url: str, pull_device: str,
                 capture_options: str, frame_callback=None, status_callback=None):
        self._stream_id = stream_id
        self._stream_url = stream_url
        self._pull_device = pull_device
        self._capture_options = capture_options
        self._frame_callback = frame_callback
        self._status_callback = status_callback

        self._ctrl_conn = None
        self._frame_conn = None
        self._process: Optional[multiprocessing.Process] = None
        self._reader_thread: Optional[object] = None
        self._running = False

        # 共享内存读取句柄（主进程侧）
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

        # ping/pong
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
            target=self._read_loop, daemon=True,
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
