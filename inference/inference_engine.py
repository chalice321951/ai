# -*- coding: utf-8 -*-
"""
推理引擎模块 - 基于YOLO的多模型目标检测推理
"""
import logging
import os
import queue
import threading
from typing import Dict, Any, List, Optional


class InferenceEngine:
    """YOLO推理引擎封装，支持多模型和 CPU/GPU 设备选择"""

    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self._models: Dict[str, Any] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._tracking_models_by_stream: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._torch = None
        self._yolo_class = None

        self._single_thread_worker_enabled = bool(getattr(self.config, 'inference_single_thread_worker', False))
        self._submit_timeout = float(getattr(self.config, 'inference_submit_timeout', 30.0) or 30.0)
        self._worker_queue: "queue.Queue" = queue.Queue(maxsize=256)
        self._worker_stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._load_models()
        self._start_worker_if_needed()

    def _start_worker_if_needed(self):
        if not self._single_thread_worker_enabled:
            logging.info("推理串行worker已禁用，沿用调用线程直接推理")
            return
        if not self._loaded:
            return
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="InferenceWorker",
        )
        self._worker_thread.start()
        logging.info("推理串行worker已启动，用于串行化YOLO/Torch/CUDA调用")

    def _resolve_device(self, device_value: Optional[str]) -> str:
        requested = str(device_value or getattr(self.config, 'model_device', 'auto')).strip().lower()
        if requested in ('', 'auto'):
            try:
                import torch
                self._torch = torch
                if torch.cuda.is_available():
                    return 'cuda:0'
            except Exception:
                pass
            return 'cpu'
        if requested == 'gpu':
            return 'cuda:0'
        return requested

    def _create_model_instance(self, model_path: str, runtime_device: str):
        if self._yolo_class is None:
            from ultralytics import YOLO
            self._yolo_class = YOLO
        logging.info(f"创建YOLO模型实例 device={runtime_device}: {model_path}")
        return self._yolo_class(model_path)

    def _load_models(self):
        """加载配置中的所有检测模型"""
        try:
            if self._yolo_class is None:
                from ultralytics import YOLO
                self._yolo_class = YOLO
            model_defs = getattr(self.config, 'get_enabled_models', lambda: [])()

            for model_cfg in model_defs:
                model_id = str(model_cfg.get('id', 'unknown'))
                model_path = str(model_cfg.get('path', ''))
                if not model_path:
                    logging.warning(f"模型路径为空，跳过: [{model_id}]")
                    continue
                if not os.path.exists(model_path):
                    logging.warning(f"模型文件不存在，跳过: [{model_id}] {model_path}")
                    continue

                runtime_device = self._resolve_device(model_cfg.get('device'))
                try:
                    logging.info(f"加载检测模型 [{model_id}] device={runtime_device}: {model_path}")
                    model = self._create_model_instance(model_path, runtime_device)
                    self._models[model_id] = model
                    self._model_configs[model_id] = {
                        'id': model_id,
                        'name': model_cfg.get('name', model_id),
                        'task': model_cfg.get('task', 'detection'),
                        'path': model_path,
                        'conf_threshold': float(model_cfg.get('conf_threshold', getattr(self.config, 'default_conf_threshold', 0.5))),
                        'device': runtime_device,
                    }
                    logging.info(f"模型 [{model_id}] 加载成功")
                except Exception as e:
                    logging.error(f"模型 [{model_id}] 加载失败: {e}")

            if self._models:
                self._loaded = True
                logging.info(f"共加载 {len(self._models)} 个模型: {list(self._models.keys())}")
            else:
                logging.warning("未加载任何模型，将以透传模式运行（仅推流，不检测）")
        except ImportError:
            logging.error("ultralytics 未安装，无法加载YOLO模型")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")

    def _get_models_for_inference(self, tracking_enabled: bool, stream_key: Optional[str]) -> Dict[str, Any]:
        if not tracking_enabled:
            return self._models

        stream_name = str(stream_key or '').strip()
        if not stream_name:
            return self._models

        stream_models = self._tracking_models_by_stream.get(stream_name)
        if stream_models is not None:
            return stream_models

        stream_models = {}
        for model_id, model_cfg in self._model_configs.items():
            model_path = str(model_cfg.get('path', ''))
            if not model_path:
                continue
            try:
                stream_models[model_id] = self._create_model_instance(model_path, model_cfg.get('device', 'cpu'))
            except Exception as e:
                logging.error(f"为流 [{stream_name}] 创建跟踪模型 [{model_id}] 失败: {e}")

        if stream_models:
            self._tracking_models_by_stream[stream_name] = stream_models
            logging.info(f"流 [{stream_name}] 已创建独立跟踪器状态，模型数={len(stream_models)}")
            return stream_models

        return self._models

    def _trace_infer_stage(self, stream_key: Optional[str], model_id: str, stage: str, **kwargs):
        extras = []
        if stream_key:
            extras.append(f"stream={stream_key}")
        extras.append(f"model={model_id}")
        extras.append(f"stage={stage}")
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
        logging.info("[InferTrace] " + ' '.join(extras))
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass

    def _run_inference_internal(self, frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if not self._models:
            return results

        tracking_enabled = bool(getattr(self.config, 'tracking_enabled', False))
        tracking_persist = bool(getattr(self.config, 'tracking_persist', True))
        tracking_tracker = str(getattr(self.config, 'tracking_tracker', 'bytetrack.yaml') or 'bytetrack.yaml')
        tracking_conf = float(getattr(self.config, 'tracking_conf_threshold', getattr(self.config, 'default_conf_threshold', 0.5)))

        with self._lock:
            model_store = self._get_models_for_inference(tracking_enabled, stream_key)
            if algo_id and str(algo_id) in model_store:
                model_ids = [str(algo_id)]
            else:
                model_ids = list(model_store.keys())

            for model_id in model_ids:
                model = model_store.get(model_id)
                model_cfg = self._model_configs.get(model_id, {})
                conf = float(model_cfg.get('conf_threshold', getattr(self.config, 'default_conf_threshold', 0.5)))
                device = model_cfg.get('device', 'cpu')
                try:
                    infer_mode = 'track' if tracking_enabled else 'predict'
                    self._trace_infer_stage(stream_key, model_id, 'call_enter', mode=infer_mode, device=device)
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
                    self._trace_infer_stage(stream_key, model_id, 'call_return', mode=infer_mode)
                    if self._torch is not None and str(device).lower().startswith('cuda'):
                        self._trace_infer_stage(stream_key, model_id, 'cuda_sync_start', mode=infer_mode)
                        self._torch.cuda.synchronize()
                        self._trace_infer_stage(stream_key, model_id, 'cuda_sync_end', mode=infer_mode)
                    self._trace_infer_stage(stream_key, model_id, 'end', mode=infer_mode)
                    results[model_id] = res
                except TypeError:
                    try:
                        fallback_mode = 'track_fallback' if tracking_enabled else 'predict_fallback'
                        self._trace_infer_stage(stream_key, model_id, 'fallback_call_enter', mode=fallback_mode, device=device)
                        if tracking_enabled:
                            res = model.track(
                                frame,
                                conf=max(conf, tracking_conf),
                                verbose=False,
                                persist=tracking_persist,
                                tracker=tracking_tracker,
                            )
                        else:
                            res = model(frame, conf=conf, verbose=False)
                        self._trace_infer_stage(stream_key, model_id, 'fallback_call_return', mode=fallback_mode)
                        if self._torch is not None and str(device).lower().startswith('cuda'):
                            self._trace_infer_stage(stream_key, model_id, 'fallback_cuda_sync_start', mode=fallback_mode)
                            self._torch.cuda.synchronize()
                            self._trace_infer_stage(stream_key, model_id, 'fallback_cuda_sync_end', mode=fallback_mode)
                        self._trace_infer_stage(stream_key, model_id, 'fallback_end', mode=fallback_mode)
                        results[model_id] = res
                    except Exception as e:
                        logging.error(f"推理 [{model_id}] 失败: {e}")
                except Exception as e:
                    logging.error(f"推理 [{model_id}] 失败: {e}")

        return results

    def _worker_loop(self):
        while not self._worker_stop_event.is_set():
            try:
                task = self._worker_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if task is None:
                self._worker_queue.task_done()
                break

            response_queue = task.get('response_queue')
            stream_key = task.get('stream_key')
            algo_id = task.get('algo_id')
            model_label = str(algo_id) if algo_id else 'all'
            self._trace_infer_stage(stream_key, model_label, 'worker_dequeue')
            try:
                result = self._run_inference_internal(
                    frame=task.get('frame'),
                    algo_id=algo_id,
                    stream_key=stream_key,
                )
                if response_queue is not None:
                    response_queue.put({'ok': True, 'result': result})
            except Exception as e:
                logging.error(f"串行推理worker执行失败: {e}")
                if response_queue is not None:
                    response_queue.put({'ok': False, 'error': e})
            finally:
                self._worker_queue.task_done()

    def infer(self, frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        """对单帧执行多模型推理，返回 {algo_id: results, ...}"""
        if not self._models:
            return {}

        if not self._single_thread_worker_enabled:
            return self._run_inference_internal(frame=frame, algo_id=algo_id, stream_key=stream_key)

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_worker_if_needed()
            if self._worker_thread is None or not self._worker_thread.is_alive():
                logging.error("推理串行worker未启动，回退直接推理")
                return self._run_inference_internal(frame=frame, algo_id=algo_id, stream_key=stream_key)

        response_queue: "queue.Queue" = queue.Queue(maxsize=1)
        task = {
            'frame': frame,
            'algo_id': algo_id,
            'stream_key': stream_key,
            'response_queue': response_queue,
        }

        model_label = str(algo_id) if algo_id else 'all'
        try:
            self._trace_infer_stage(stream_key, model_label, 'worker_enqueue')
            self._worker_queue.put(task, timeout=max(1.0, self._submit_timeout))
        except queue.Full:
            logging.error("推理任务提交超时，worker队列已满")
            return {}

        try:
            response = response_queue.get(timeout=max(1.0, self._submit_timeout))
        except queue.Empty:
            logging.error("推理任务等待结果超时，回退空结果")
            return {}

        if not response.get('ok', False):
            logging.error(f"串行推理worker返回失败: {response.get('error')}")
            return {}
        return response.get('result', {}) or {}

    def reset_stream_tracking(self, stream_key: str):
        stream_name = str(stream_key or '').strip()
        if not stream_name:
            return
        with self._lock:
            stream_models = self._tracking_models_by_stream.pop(stream_name, None)
        if stream_models is not None:
            logging.info(f"流 [{stream_name}] 跟踪状态已重置")

    def is_loaded(self) -> bool:
        return self._loaded

    def get_model_ids(self) -> List[str]:
        return list(self._models.keys())

    def get_model_runtime_configs(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._model_configs)

    def cleanup(self):
        self._worker_stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            try:
                self._worker_queue.put_nowait(None)
            except Exception:
                pass
            self._worker_thread.join(timeout=5.0)
        with self._lock:
            self._models.clear()
            self._model_configs.clear()
            self._tracking_models_by_stream.clear()
        logging.info("推理引擎已清理")
