# -*- coding: utf-8 -*-
"""
日志配置模块
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """设置全局日志配置

    Args:
        level: 日志级别
        log_file: 日志文件路径 (可选)
        format_string: 日志格式
    """
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """获取模块日志器

    Args:
        name: 日志器名称

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)
