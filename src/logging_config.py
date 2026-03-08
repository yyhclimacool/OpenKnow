from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = _PROJECT_ROOT / "logs"

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# hourly rotation, keep 30 days (30 * 24 = 720 files)
ROTATION_WHEN = "H"
BACKUP_COUNT = 30 * 24


def setup_logging(level: int = logging.INFO, *, console: bool = True) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # avoid duplicate handlers on repeated calls
    root.handlers.clear()

    file_handler = TimedRotatingFileHandler(
        filename=LOG_DIR / "forrest.log",
        when=ROTATION_WHEN,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
        utc=False,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    file_handler.suffix = "%Y-%m-%d_%H"
    root.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        root.addHandler(stream_handler)
