import os
import sys
from pathlib import Path

import logging
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def get_logger(
    name: Optional[str] = None,
    log_level: int = logging.INFO,
    enable_stream: bool = True,
    enable_file: bool = True,
    fmt: str = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt: str = '%Y-%m-%d %H:%M:%S',
    propagate: bool = False
) -> logging.Logger:
    """
    Creates and configures a logger.

    Args:
        name: Logger name (e.g., module name)
        log_file: Path to log file (default is '../../log/app.log' if not provided)
        log_level: Logging level (e.g., logging.DEBUG, logging.INFO)
        enable_stream: Whether to log to stdout
        enable_file: Whether to log to a file
        fmt: Log message format
        datefmt: Log date format
        propagate: Whether to propagate to parent loggers

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        if enable_stream:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        if enable_file:
            default_dir = os.path.join(ROOT_DIR, "log",)
            os.makedirs(default_dir, exist_ok=True)
            log_file = os.path.join(default_dir, "ratansunpy.log")

            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = propagate

    return logger
