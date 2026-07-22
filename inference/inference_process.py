# -*- coding: utf-8 -*-
"""
子进程推理模块 - 每路流独立进程，彻底隔离 CUDA context
帧数据通过共享内存传递，控制消息通过 Pipe 传递（避免 Queue._feed 线程 native 崩溃）
"""
import logging
import multiprocessing
import multiprocessing.shared_memory as shm_mod
import numpy as np
import os
import time
from typing import Dict, Any, List, Optional


# ──────────────────────────────────────────────
# 消息协议
# ──────────────────────────────────────────────
_CMD_INFER = 'infer'
_CMD_STOP  = 'stop'
_CMD_PING  = 'ping'

_STATUS_OK    = 'ok'
_STATUS_ERROR = 'error'
_STATUS_PONG  = 'pong'


def _worker_main(
    stream_name: str,
    model_defs: List[Dict[str, Any]],
    tracking_cfg: Dict[str, Any],
    req_conn,
    res_conn,
    main_pid: int,
):
    """
    子进程入口：加载模型，循环处理推理请求。
    控制消息通过 Pipe 传递，帧数据通过共享内存读取。
    父进程死亡时子进程自动退出（心跳检测）。
    """
    import threading

    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(levelname)s] [infer-{stream_name}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logging.info(f"推理子进程启动: stream={stream_name} pid={os.getpid()}")

    parent_pid = main_pid

    def _watch_parent():
        while True:
            time.sleep(2.0)
            try:
                os.kill(parent_pid, 0)
            except (ProcessLookupError, PermissionError):
                logging.warning(f"父进程 {parent_pid} 已退出，子进程自动退出")
                os._exit(0)
            except Exception:
                pass

    threading.Thread(target=_watch_parent, daemon=True).start()

    models: Dict[str, Any] = {}
    model_cfgs: Dict[str, Dict] = {}
    torch_ref = None

    try:
        from ultralytics import YOLO
        for mdef in model_defs:
            mid   = str(mdef.get('id', 'unknown'))
            mpath = str(mdef.get('path', ''))
            device = str(mdef.get('device', 'cpu'))
            if not mpath or not os.path.exists(mpath):
                logging.warning(f"模型文件不存在，跳过: [{mid}] {mpath}")
                continue
            try:
                logging.info(f"加载模型 [{mid}] device={device}: {mpath}")
                models[mid] = YOLO(mpath)
                model_cfgs[mid] = dict(mdef)
                logging.info(f"模型 [{mid}] 加载成功")
            except Exception as e:
                logging.error(f"模型 [{mid}] 加载失败: {e}")

        try:
            import torch
            torch_ref = torch
        except Exception:
            pass

    except Exception as e:
        logging.error(f"推理子进程初始化失败: {e}")
        try:
            res_conn.send({'status': _STATUS_ERROR, 'error': str(e), 'seq': -1})
        except Exception:
            pass
        return

    if not models:
        logging.warning("未加载任何模型，子进程以空模式运行")

    tracking_enabled  = bool(tracking_cfg.get('tracking_enabled', False))
    tracking_persist  = bool(tracking_cfg.get('tracking_persist', True))
    tracking_tracker  = str(tracking_cfg.get('tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml')
    tracking_conf     = float(tracking_cfg.get('tracking_conf_threshold', 0.3))
    default_conf      = float(tracking_cfg.get('default_conf_threshold', 0.5))
    imgsz             = int(tracking_cfg.get('imgsz', 640) or 640)

    logging.info(f"推理子进程就绪: models={list(models.keys())} tracking={tracking_enabled}")

    while True:
        try:
            if not req_conn.poll(1.0):
                continue
            msg = req_conn.recv()
        except EOFError:
            logging.info("推理子进程 Pipe 已关闭，退出")
            break
        except Exception as e:
            logging.error(f"推理子进程接收消息失败: {e}")
            continue

        cmd = msg.get('cmd')
        seq = msg.get('seq', 0)

        if cmd == _CMD_STOP:
            logging.info("推理子进程收到停止指令，退出")
            break

        if cmd == _CMD_PING:
            try:
                res_conn.send({'status': _STATUS_PONG, 'seq': seq})
            except Exception:
                pass
            continue

        if cmd != _CMD_INFER:
            continue

        shm_name = msg.get('shm_name')
        shape    = msg.get('shape')
        dtype    = msg.get('dtype')
        algo_id  = msg.get('algo_id')

        frame = None
        if shm_name and shape and dtype:
            shm_handle = None
            try:
                shm_handle = shm_mod.SharedMemory(name=shm_name)
                frame = np.ndarray(shape, dtype=dtype, buffer=shm_handle.buf).copy()
            except Exception as e:
                logging.error(f"读取共享内存失败: {e}")
            finally:
                if shm_handle is not None:
                    try:
                        shm_handle.close()
                    except Exception:
                        pass

        if frame is None or not models:
            try:
                res_conn.send({'status': _STATUS_OK, 'results': {}, 'seq': seq})
            except Exception:
                pass
            continue

        model_ids = [str(algo_id)] if (algo_id and str(algo_id) in models) else list(models.keys())

        out: Dict[str, Any] = {}
        for mid in model_ids:
            model = models.get(mid)
            if model is None:
                continue
            mcfg   = model_cfgs.get(mid, {})
            conf   = float(mcfg.get('conf_threshold', default_conf))
            device = str(mcfg.get('device', 'cpu'))
            try:
                if tracking_enabled:
                    res = model.track(
                        frame,
                        conf=max(conf, tracking_conf),
                        device=device,
                        verbose=False,
                        persist=tracking_persist,
                        tracker=tracking_tracker,
                        imgsz=imgsz,
                    )
                else:
                    res = model.predict(frame, conf=conf, device=device, imgsz=imgsz, verbose=False)

                if torch_ref is not None and str(device).lower().startswith('cuda'):
                    torch_ref.cuda.synchronize()

                out[mid] = res
            except Exception as e:
                logging.error(f"推理 [{mid}] 失败: {e}")

        try:
            res_conn.send({'status': _STATUS_OK, 'results': out, 'seq': seq})
        except Exception as e:
            logging.error(f"推理子进程发送结果失败: {e}")

    logging.info(f"推理子进程退出: stream={stream_name}")


# ──────────────────────────────────────────────
# 主进程侧代理
# ──────────────────────────────────────────────
class InferenceProxy:
    """
    主进程侧代理：管理一个推理子进程的生命周期。
    控制消息通过 Pipe 传递，帧通过共享内存传递。
    """

    _ALIVE_CHECK_INTERVAL = 30

    def __init__(self, stream_name: str, config):
        self._stream_name = stream_name
        self._config = config
        self._seq = 0
        self._timeout = float(getattr(config, 'inference_submit_timeout', 30.0) or 30.0)

        self._req_conn = None  # 主进程写端
        self._res_conn = None  # 主进程读端
        self._process: Optional[multiprocessing.Process] = None
        self._loaded = False
        self._alive_cache = False
        self._alive_check_counter = 0

        self._shm: Optional[shm_mod.SharedMemory] = None
        self._shm_shape = None
        self._shm_dtype = None

        self._start_process()

    def _build_model_defs(self) -> List[Dict[str, Any]]:
        get_enabled = getattr(self._config, 'get_enabled_models', None)
        if callable(get_enabled):
            return list(get_enabled())
        return list(getattr(self._config, 'model_definitions', []))

    def _build_tracking_cfg(self) -> Dict[str, Any]:
        return {
            'tracking_enabled':        bool(getattr(self._config, 'tracking_enabled', False)),
            'tracking_persist':        bool(getattr(self._config, 'tracking_persist', True)),
            'tracking_tracker':        str(getattr(self._config, 'tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml'),
            'tracking_conf_threshold': float(getattr(self._config, 'tracking_conf_threshold', 0.3)),
            'default_conf_threshold':  float(getattr(self._config, 'default_conf_threshold', 0.5)),
            'imgsz':                   int(getattr(self._config, 'imgsz', 640) or 640),
        }

    def _start_process(self):
        model_defs   = self._build_model_defs()
        tracking_cfg = self._build_tracking_cfg()

        ctx = multiprocessing.get_context('forkserver')

        # duplex=False: Pipe() 返回 (read_end, write_end)
        # 请求管道：主进程写 → 子进程读
        req_child_read, req_parent_write = ctx.Pipe(duplex=False)
        # 结果管道：子进程写 → 主进程读
        res_parent_read, res_child_write = ctx.Pipe(duplex=False)

        self._req_conn = req_parent_write  # 主进程写端
        self._res_conn = res_parent_read   # 主进程读端

        self._process = ctx.Process(
            target=_worker_main,
            args=(
                self._stream_name,
                model_defs,
                tracking_cfg,
                req_child_read,
                res_child_write,
                os.getpid(),
            ),
            daemon=True,
            name=f"InferProc-{self._stream_name}",
        )
        self._process.start()

        # 子进程启动后关闭主进程不需要的端
        req_child_read.close()
        res_child_write.close()

        logging.info(f"[{self._stream_name}] 推理子进程已启动 pid={self._process.pid}")

        # ping/pong 等待就绪
        self._seq += 1
        try:
            self._req_conn.send({'cmd': _CMD_PING, 'seq': self._seq})
            if self._res_conn.poll(60.0):
                resp = self._res_conn.recv()
                if resp.get('status') == _STATUS_PONG:
                    self._loaded = True
                    self._alive_cache = True
                    logging.info(f"[{self._stream_name}] 推理子进程就绪")
                else:
                    logging.error(f"[{self._stream_name}] 推理子进程启动异常: {resp}")
            else:
                logging.error(f"[{self._stream_name}] 等待推理子进程就绪超时")
        except Exception as e:
            logging.error(f"[{self._stream_name}] 推理子进程就绪握手失败: {e}")

    def _ensure_shm(self, frame: np.ndarray) -> bool:
        needed = frame.nbytes
        if self._shm is not None:
            if self._shm.size >= needed and self._shm_shape == frame.shape and self._shm_dtype == frame.dtype:
                return True
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
        try:
            self._shm = shm_mod.SharedMemory(create=True, size=max(needed, 1))
            self._shm_shape = frame.shape
            self._shm_dtype = frame.dtype
            return True
        except Exception as e:
            logging.error(f"[{self._stream_name}] 创建共享内存失败: {e}")
            return False

    def is_loaded(self) -> bool:
        if not self._loaded:
            return False
        self._alive_check_counter += 1
        if self._alive_check_counter >= self._ALIVE_CHECK_INTERVAL:
            self._alive_check_counter = 0
            self._alive_cache = (self._process is not None and self._process.is_alive())
        return self._alive_cache

    def infer(self, frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        if not self.is_loaded():
            return {}

        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        if not self._ensure_shm(frame):
            return {}

        dst = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self._shm.buf)
        np.copyto(dst, frame)

        self._seq += 1
        seq = self._seq

        try:
            self._req_conn.send({
                'cmd':      _CMD_INFER,
                'shm_name': self._shm.name,
                'shape':    frame.shape,
                'dtype':    str(frame.dtype),
                'algo_id':  algo_id,
                'seq':      seq,
            })
        except Exception:
            logging.warning(f"[{self._stream_name}] 推理请求发送失败，丢弃本帧")
            return {}

        try:
            if not self._res_conn.poll(self._timeout):
                logging.error(f"[{self._stream_name}] 等待推理结果超时")
                return {}
            resp = self._res_conn.recv()
        except Exception as e:
            logging.error(f"[{self._stream_name}] 接收推理结果失败: {e}")
            return {}

        if resp.get('status') != _STATUS_OK:
            logging.error(f"[{self._stream_name}] 推理子进程返回错误: {resp.get('error')}")
            return {}

        return resp.get('results', {}) or {}

    def reset_stream_tracking(self, stream_key: str):
        pass

    def cleanup(self):
        self._loaded = False
        self._alive_cache = False

        if self._process is not None and self._process.is_alive():
            try:
                self._req_conn.send({'cmd': _CMD_STOP, 'seq': 0})
            except Exception:
                pass
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=3.0)
            logging.info(f"[{self._stream_name}] 推理子进程已停止")

        self._process = None

        for conn in [self._req_conn, self._res_conn]:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        self._req_conn = None
        self._res_conn = None

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None
