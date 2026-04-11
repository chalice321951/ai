# -*- coding: utf-8 -*-
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler


def setup_logging(log_dir='log', log_level=logging.INFO):
    """配置日志系统。"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'ai_camera.log')

    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if root_logger.handlers:
        return

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    file_handler = TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=30,
        encoding='utf-8'
    )
    file_handler.suffix = '%Y-%m-%d'
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    logging.info("日志系统初始化完成")
