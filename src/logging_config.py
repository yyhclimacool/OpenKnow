from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = _PROJECT_ROOT / "logs"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_BUFFER_FORMAT = "[%(levelname)s] %(message)s"  # 简短格式，供 2 行区域显示

# hourly rotation, keep 30 days (30 * 24 = 720 files)
ROTATION_WHEN = "H"
BACKUP_COUNT = 30 * 24


class LogBufferHandler(logging.Handler):
    """将日志写入环形缓冲区，供 Rich Live 显示。"""

    def __init__(self, buffer: list[str], max_lines: int = 2) -> None:
        super().__init__()
        self._buffer = buffer
        self._max_lines = max_lines

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self._buffer.append(msg)
        if len(self._buffer) > self._max_lines:
            self._buffer.pop(0)


def setup_logging(
    level: int = logging.INFO,
    *,
    console: bool = True,
    log_buffer: list[str] | None = None,
    log_buffer_lines: int = 2,
) -> list[str] | None:
    """配置日志。若提供 log_buffer 或 log_buffer 由调用方创建后传入，则同时写入缓冲供控制台 2 行区域显示。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # avoid duplicate handlers on repeated calls
    root.handlers.clear()

    file_handler = TimedRotatingFileHandler(
        filename=LOG_DIR / "openknow.log",
        when=ROTATION_WHEN,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    file_handler.suffix = "%Y-%m-%d_%H"
    root.addHandler(file_handler)

    # 始终保留 stderr handler，避免 HuggingFace 等库在无 stderr 时出现 Bad file descriptor
    if console or log_buffer is not None:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(stream_handler)

    if log_buffer is not None:
        buffer_handler = LogBufferHandler(log_buffer, max_lines=log_buffer_lines)
        buffer_handler.setLevel(level)
        buffer_handler.setFormatter(logging.Formatter(LOG_BUFFER_FORMAT))
        root.addHandler(buffer_handler)

    return log_buffer
