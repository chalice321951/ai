# -*- coding: utf-8 -*-
"""
Inference engine based on Ultralytics YOLO.

All streams share the same loaded model instances. Tracking is handled outside
the engine so batch inference can stay fully shared across streams.
"""
import logging
import os
import queue
import threading
from typing import Any, Dict, List, Optional


class InferenceEngine:
    """YOLO inference wrapper with shared scheduling support."""

    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self._models: Dict[str, Any] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._torch = None
        self._yolo_class = None
        self._trace_enabled = bool(getattr(self.config, 'crash_trace_enabled', False))

        self._single_thread_worker_enabled = bool(getattr(self.config, 'inference_single_thread_worker', False))
        self._submit_timeout = float(getattr(self.config, 'inference_submit_timeout', 30.0) or 30.0)
        self._worker_queue: "queue.Queue" = queue.Queue(maxsize=256)
        self._worker_stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        self._load_models()
        self._start_worker_if_needed()

    def _start_worker_if_needed(self):
        if not self._single_thread_worker_enabled:
            logging.info("推理串行 worker 已禁用，调用线程将直接执行推理")
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
        logging.info("推理串行 worker 已启动，用于串行化 YOLO/Torch/CUDA 调用")

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
        logging.info(f"创建 YOLO 模型实例 device={runtime_device}: {model_path}")
        return self._yolo_class(model_path)

    def _load_models(self):
        try:
            if self._yolo_class is None:
                from ultralytics import YOLO
                self._yolo_class = YOLO
            model_defs = getattr(self.config, 'get_enabled_models', lambda: [])()

            for model_cfg in model_defs:
                model_id = str(model_cfg.get('id', 'unknown'))
                model_path = str(model_cfg.get('path', ''))
                if not model_path:
                    logging.warning(f"模型路径为空，跳过 [{model_id}]")
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
                        'conf_threshold': float(
                            model_cfg.get('conf_threshold', getattr(self.config, 'default_conf_threshold', 0.5))
                        ),
                        'device': runtime_device,
                    }
                    logging.info(f"模型 [{model_id}] 加载成功")
                except Exception as e:
                    logging.error(f"模型 [{model_id}] 加载失败: {e}")

            if self._models:
                self._loaded = True
                logging.info(f"共加载 {len(self._models)} 个模型: {list(self._models.keys())}")
            else:
                logging.warning("未加载任何模型，将以透传模式运行，仅推流不检测")
        except ImportError:
            logging.error("ultralytics 未安装，无法加载 YOLO 模型")
        except Exception as e:
            logging.error(f"加载模型失败: {e}")

    def _trace_infer_stage(self, stream_key: Optional[str], model_id: str, stage: str, **kwargs):
        if not self._trace_enabled:
            return
        extras = []
        if stream_key:
            extras.append(f"stream={stream_key}")
        extras.append(f"model={model_id}")
        extras.append(f"stage={stage}")
        for key, value in kwargs.items():
            extras.append(f"{key}={value}")
        logging.info("[InferTrace] " + ' '.join(extras))

    def _infer_with_model_store(self, model_store: Dict[str, Any], frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

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
                self._trace_infer_stage(stream_key, model_id, 'call_enter', mode='predict', device=device)
                res = model.predict(frame, conf=conf, device=device, verbose=False)
                self._trace_infer_stage(stream_key, model_id, 'call_return', mode='predict')
                if self._torch is not None and str(device).lower().startswith('cuda'):
                    self._trace_infer_stage(stream_key, model_id, 'cuda_sync_start', mode='predict')
                    self._torch.cuda.synchronize()
                    self._trace_infer_stage(stream_key, model_id, 'cuda_sync_end', mode='predict')
                self._trace_infer_stage(stream_key, model_id, 'end', mode='predict')
                results[model_id] = res
            except TypeError:
                try:
                    self._trace_infer_stage(stream_key, model_id, 'fallback_call_enter', mode='predict_fallback', device=device)
                    res = model(frame, conf=conf, verbose=False)
                    self._trace_infer_stage(stream_key, model_id, 'fallback_call_return', mode='predict_fallback')
                    if self._torch is not None and str(device).lower().startswith('cuda'):
                        self._trace_infer_stage(stream_key, model_id, 'fallback_cuda_sync_start', mode='predict_fallback')
                        self._torch.cuda.synchronize()
                        self._trace_infer_stage(stream_key, model_id, 'fallback_cuda_sync_end', mode='predict_fallback')
                    self._trace_infer_stage(stream_key, model_id, 'fallback_end', mode='predict_fallback')
                    results[model_id] = res
                except Exception as e:
                    logging.error(f"推理 [{model_id}] 失败: {e}")
            except Exception as e:
                logging.error(f"推理 [{model_id}] 失败: {e}")

        return results

    def _run_inference_internal(self, frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        if not self._models:
            return {}

        with self._lock:
            return self._infer_with_model_store(self._models, frame, algo_id=algo_id, stream_key=stream_key)

    def infer_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not self._models or not tasks:
            return {}

        outputs: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            model_store = self._models
            first_algo = tasks[0].get('algo_id')
            if first_algo and str(first_algo) in model_store:
                model_ids = [str(first_algo)]
            else:
                model_ids = list(model_store.keys())

            frames = [task.get('frame') for task in tasks]
            stream_keys = [str(task.get('stream_key', '') or '') for task in tasks]
            for stream_key in stream_keys:
                outputs[stream_key] = {}

            for model_id in model_ids:
                model = model_store.get(model_id)
                model_cfg = self._model_configs.get(model_id, {})
                conf = float(model_cfg.get('conf_threshold', getattr(self.config, 'default_conf_threshold', 0.5)))
                device = model_cfg.get('device', 'cpu')
                try:
                    self._trace_infer_stage('batch', model_id, 'batch_call_enter', size=len(frames), device=device)
                    res_batch = model.predict(frames, conf=conf, device=device, verbose=False)
                    if self._torch is not None and str(device).lower().startswith('cuda'):
                        self._torch.cuda.synchronize()
                    self._trace_infer_stage('batch', model_id, 'batch_call_return', size=len(frames))
                except TypeError:
                    try:
                        res_batch = model(frames, conf=conf, verbose=False)
                    except Exception as e:
                        logging.error(f"批量推理 [{model_id}] 失败: {e}")
                        continue
                except Exception as e:
                    logging.error(f"批量推理 [{model_id}] 失败: {e}")
                    continue

                if not isinstance(res_batch, (list, tuple)):
                    res_batch = [res_batch]
                for idx, stream_key in enumerate(stream_keys):
                    if idx < len(res_batch):
                        outputs[stream_key][model_id] = res_batch[idx]

        return outputs

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
                logging.error(f"串行推理 worker 执行失败: {e}")
                if response_queue is not None:
                    response_queue.put({'ok': False, 'error': e})
            finally:
                self._worker_queue.task_done()

    def infer(self, frame, algo_id: str = None, stream_key: str = None) -> Dict[str, Any]:
        if not self._models:
            return {}

        if not self._single_thread_worker_enabled:
            return self._run_inference_internal(frame=frame, algo_id=algo_id, stream_key=stream_key)

        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._start_worker_if_needed()
            if self._worker_thread is None or not self._worker_thread.is_alive():
                logging.error("推理串行 worker 未启动，回退到直接推理")
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
            logging.error("推理任务提交超时，worker 队列已满")
            return {}

        try:
            response = response_queue.get(timeout=max(1.0, self._submit_timeout))
        except queue.Empty:
            logging.error("等待推理结果超时，返回空结果")
            return {}

        if not response.get('ok', False):
            logging.error(f"串行推理 worker 返回失败: {response.get('error')}")
            return {}
        return response.get('result', {}) or {}

    def reset_stream_tracking(self, stream_key: str):
        return

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
        logging.info("推理引擎已清理")
