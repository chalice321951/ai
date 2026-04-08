# -*- coding: utf-8 -*-
"""
子进程推理模块 - 每路流独立进程，彻底隔离 CUDA context
"""
import logging
import multiprocessing
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
    req_queue: multiprocessing.Queue,
    res_queue: multiprocessing.Queue,
):
    """
    子进程入口：加载模型，循环处理推理请求。
    完全独立的 CUDA context，不与其他流共享任何 GPU 资源。
    """
    # 子进程内部配置日志（不继承父进程 handler）
    logging.basicConfig(
        level=logging.INFO,
        format=f'[%(asctime)s] [%(levelname)s] [infer-{stream_name}] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logging.info(f"推理子进程启动: stream={stream_name} pid={os.getpid()}")

    # 加载模型
    models: Dict[str, Any] = {}
    model_cfgs: Dict[str, Dict] = {}
    torch_ref = None

    try:
        from ultralytics import YOLO
        for mdef in model_defs:
            mid = str(mdef.get('id', 'unknown'))
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
        res_queue.put({'status': _STATUS_ERROR, 'error': str(e), 'seq': -1})
        return

    if not models:
        logging.warning("未加载任何模型，子进程以空模式运行")

    tracking_enabled   = bool(tracking_cfg.get('tracking_enabled', False))
    tracking_persist   = bool(tracking_cfg.get('tracking_persist', True))
    tracking_tracker   = str(tracking_cfg.get('tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml')
    tracking_conf      = float(tracking_cfg.get('tracking_conf_threshold', 0.3))
    default_conf       = float(tracking_cfg.get('default_conf_threshold', 0.5))

    logging.info(f"推理子进程就绪: models={list(models.keys())} tracking={tracking_enabled}")

    while True:
        try:
            msg = req_queue.get(timeout=1.0)
        except Exception:
            continue

        cmd = msg.get('cmd')
        seq = msg.get('seq', 0)

        if cmd == _CMD_STOP:
            logging.info("推理子进程收到停止指令，退出")
            break

        if cmd == _CMD_PING:
            res_queue.put({'status': _STATUS_PONG, 'seq': seq})
            continue

        if cmd != _CMD_INFER:
            continue

        frame = msg.get('frame')
        algo_id = msg.get('algo_id')

        if frame is None or not models:
            res_queue.put({'status': _STATUS_OK, 'results': {}, 'seq': seq})
            continue

        if algo_id and str(algo_id) in models:
            model_ids = [str(algo_id)]
        else:
            model_ids = list(models.keys())

        out: Dict[str, Any] = {}
        for mid in model_ids:
            model = models.get(mid)
            if model is None:
                continue
            mcfg  = model_cfgs.get(mid, {})
            conf  = float(mcfg.get('conf_threshold', default_conf))
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
                    )
                else:
                    res = model.predict(frame, conf=conf, device=device, verbose=False)

                if torch_ref is not None and str(device).lower().startswith('cuda'):
                    torch_ref.cuda.synchronize()

                out[mid] = res
            except Exception as e:
                logging.error(f"推理 [{mid}] 失败: {e}")

        res_queue.put({'status': _STATUS_OK, 'results': out, 'seq': seq})

    logging.info(f"推理子进程退出: stream={stream_name}")


# ──────────────────────────────────────────────
# 主进程侧代理
# ──────────────────────────────────────────────
class InferenceProxy:
    """
    主进程侧代理：管理一个推理子进程的生命周期，
    提供与 InferenceEngine 相同的 infer() / is_loaded() / cleanup() 接口。
    """

    def __init__(self, stream_name: str, config):
        self._stream_name = stream_name
        self._config = config
        self._seq = 0
        self._timeout = float(getattr(config, 'inference_submit_timeout', 30.0) or 30.0)

        self._req_queue = None
        self._res_queue = None
        self._process: Optional[multiprocessing.Process] = None
        self._loaded = False

        self._start_process()

    def _build_model_defs(self) -> List[Dict[str, Any]]:
        get_enabled = getattr(self._config, 'get_enabled_models', None)
        if callable(get_enabled):
            return list(get_enabled())
        return list(getattr(self._config, 'model_definitions', []))

    def _build_tracking_cfg(self) -> Dict[str, Any]:
        return {
            'tracking_enabled':       bool(getattr(self._config, 'tracking_enabled', False)),
            'tracking_persist':       bool(getattr(self._config, 'tracking_persist', True)),
            'tracking_tracker':       str(getattr(self._config, 'tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml'),
            'tracking_conf_threshold': float(getattr(self._config, 'tracking_conf_threshold', 0.3)),
            'default_conf_threshold': float(getattr(self._config, 'default_conf_threshold', 0.5)),
        }

    def _start_process(self):
        model_defs   = self._build_model_defs()
        tracking_cfg = self._build_tracking_cfg()

        ctx = multiprocessing.get_context('forkserver')
        self._req_queue = ctx.Queue(maxsize=8)
        self._res_queue = ctx.Queue(maxsize=8)

        self._process = ctx.Process(
            target=_worker_main,
            args=(
                self._stream_name,
                model_defs,
                tracking_cfg,
                self._req_queue,
                self._res_queue,
            ),
            daemon=True,
            name=f"InferProc-{self._stream_name}",
        )
        self._process.start()
        logging.info(f"[{self._stream_name}] 推理子进程已启动 pid={self._process.pid}")

        # 等待子进程就绪（ping/pong）
        self._seq += 1
        self._req_queue.put({'cmd': _CMD_PING, 'seq': self._seq})
        try:
            resp = self._res_queue.get(timeout=60.0)
            if resp.get('status') == _STATUS_PONG:
                self._loaded = True
                logging.info(f"[{self._stream_name}] 推理子进程就绪")
            else:
                logging.error(f"[{self._stream_name}] 推理子进程启动异常: {resp}")
        except Exception as e:
            logging.error(f"[{self._stream_name}] 等待推理子进程就绪超时: {e}")

    def is_loaded(self) -> bool:
        return self._loaded and self._process is not None and self._process.is_alive()

    def infer(self, frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        if not self.is_loaded():
            logging.warning(f"[{self._stream_name}] 推理子进程未就绪，跳过推理")
            return {}

        self._seq += 1
        seq = self._seq

        # 清空过期结果，避免积压
        while not self._res_queue.empty():
            try:
                self._res_queue.get_nowait()
            except Exception:
                break

        try:
            self._req_queue.put_nowait({'cmd': _CMD_INFER, 'frame': frame, 'algo_id': algo_id, 'seq': seq})
        except Exception:
            logging.warning(f"[{self._stream_name}] 推理请求队列已满，丢弃本帧")
            return {}

        try:
            resp = self._res_queue.get(timeout=self._timeout)
        except Exception:
            logging.error(f"[{self._stream_name}] 等待推理结果超时")
            return {}

        if resp.get('status') != _STATUS_OK:
            logging.error(f"[{self._stream_name}] 推理子进程返回错误: {resp.get('error')}")
            return {}

        return resp.get('results', {}) or {}

    def reset_stream_tracking(self, stream_key: str):
        # 子进程内 tracker 状态随进程隔离，重置只需重启进程
        pass

    def cleanup(self):
        if self._process is not None and self._process.is_alive():
            try:
                self._req_queue.put_nowait({'cmd': _CMD_STOP, 'seq': 0})
            except Exception:
                pass
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=3.0)
            logging.info(f"[{self._stream_name}] 推理子进程已停止")
        self._process = None
        self._loaded = False
