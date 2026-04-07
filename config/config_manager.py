# -*- coding: utf-8 -*-
import json
import os
import threading


class ConfigManager:
    """配置管理器单例类"""
    _instance = None
    _lock = threading.Lock()
    _config = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                print(f"警告: 配置文件不存在: {config_path}")
                self._config = {}
        except Exception as e:
            print(f"错误: 加载配置文件失败: {e}")
            self._config = {}

    def get(self, key, default=None):
        return self._config.get(key, default)

    def get_streams(self):
        """获取所有流配置"""
        return self._config.get('streams', [])

    def get_enabled_streams(self):
        """获取所有启用的流配置"""
        streams = []
        for stream_cfg in self.get_streams():
            if not stream_cfg.get('enabled', True):
                continue
            normalized = dict(stream_cfg)
            if not normalized.get('input_url'):
                normalized['input_url'] = normalized.get('rtsp_url') or normalized.get('rtmp_url') or ''
            if not normalized.get('output_url'):
                normalized['output_url'] = normalized.get('output_rtsp') or normalized.get('output_rtmp') or ''
            streams.append(normalized)
        return streams
