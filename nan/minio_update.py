# -*- coding: utf-8 -*-
"""
MinIO 上传模块
"""
import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import nan.constants

logger = logging.getLogger(__name__)


def _resolve_minio_settings(minio_config: Optional[Any] = None) -> dict:
    if minio_config is not None:
        endpoint = str(getattr(minio_config, 'minio_endpoint', '') or '')
        access_key = str(getattr(minio_config, 'minio_access_key', '') or '')
        secret_key = str(getattr(minio_config, 'minio_secret_key', '') or '')
        secure = bool(getattr(minio_config, 'minio_secure', False))
        bucket_name = str(getattr(minio_config, 'minio_bucket_name', '') or '')
        if endpoint and bucket_name:
            return {
                'endpoint': endpoint,
                'access_key': access_key,
                'secret_key': secret_key,
                'secure': secure,
                'bucket_name': bucket_name,
            }

    return {
        'endpoint': nan.constants.minioInfo.endpoint,
        'access_key': nan.constants.minioInfo.access_key,
        'secret_key': nan.constants.minioInfo.secret_key,
        'secure': nan.constants.minioInfo.secure,
        'bucket_name': nan.constants.minioInfo.bucket_name,
    }


def _get_client(minio_config: Optional[Any] = None):
    settings = _resolve_minio_settings(minio_config)
    try:
        from minio import Minio
        return Minio(
            endpoint=settings['endpoint'],
            access_key=settings['access_key'],
            secret_key=settings['secret_key'],
            secure=settings['secure'],
        )
    except Exception as e:
        logger.error(f"MinIO客户端初始化失败: {e}")
        return None


def object_name_get(stream_cfg: dict, file_type: str, file_name: str) -> str:
    """生成 MinIO 存储路径"""
    try:
        task_id = stream_cfg.get('taskId', stream_cfg.get('name', 'unknown'))
        camera_name = stream_cfg.get('name', 'camera')
        date_str = datetime.now().strftime("%Y%m%d")
        base = f"{date_str}/task-{task_id}/vendor-{nan.constants.aiVendorName}/camera-{camera_name}/{file_type}"
        return f"{base}/{file_name}"
    except Exception as e:
        logger.error(f"object_name_get 失败: {e}")
        return f"unknown/{file_type}/{file_name}"


def upload_to_minio(object_name: str, file_path: str, max_retries: int = 3, minio_config: Optional[Any] = None) -> str:
    """上传文件到 MinIO，返回 object_name 或空字符串"""
    if not os.path.isfile(file_path):
        logger.error(f"文件不存在: {file_path}")
        return ""
    if os.path.getsize(file_path) == 0:
        logger.error(f"文件为空: {file_path}")
        return ""

    settings = _resolve_minio_settings(minio_config)
    client = _get_client(minio_config)
    if client is None:
        return ""

    for attempt in range(max_retries + 1):
        try:
            bucket = settings['bucket_name']
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
            client.fput_object(bucket_name=bucket, object_name=object_name, file_path=file_path)
            return object_name
        except Exception as e:
            logger.error(f"MinIO上传失败(尝试{attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                time.sleep(2)
    return ""


def minio_interface(stream_cfg: dict, file_type: str, file_name: str, file_path: str, minio_config: Optional[Any] = None) -> str:
    """上传并返回完整 HTTP URL"""
    settings = _resolve_minio_settings(minio_config)
    object_name = object_name_get(stream_cfg, file_type, file_name)
    result = upload_to_minio(object_name, file_path, minio_config=minio_config)
    if result:
        scheme = 'https' if settings['secure'] else 'http'
        endpoint = settings['endpoint']
        bucket = settings['bucket_name']
        return f"{scheme}://{endpoint}/{bucket}/{result}"
    return ""
