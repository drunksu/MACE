"""
Logging configuration.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
) -> None:
    """
    Setup logging configuration.

    Args:
        level: Logging level.
        log_file: Optional file path to write logs.
        console: Whether to output logs to console.
    """
    handlers = []
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        handlers.append(console_handler)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger with given name."""
    return logging.getLogger(name)