import logging
import sys
from pathlib import Path
from typing import Optional, Sequence, Union


class _FlushFileHandler(logging.FileHandler):
    """Append to disk after every record so logs survive abrupt process death when possible."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    overwrite_file: bool = False,
    quiet_loggers: Optional[Sequence[str]] = None,
) -> None:
    """
    Configure root logger: stdout + optional file (UTF-8, flushed each line).
    Safe to call once per process; avoids duplicating stream handlers.
    """
    root = logging.getLogger()
    root.setLevel(level)
    fmt = logging.Formatter(_DEFAULT_FORMAT)

    has_stream = any(
        type(h) is logging.StreamHandler and getattr(h, "stream", None) is sys.stdout
        for h in root.handlers
    )
    if not has_stream:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path.resolve())
        if overwrite_file and path.exists():
            path.unlink()
        exists_file = any(
            isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == target
            for h in root.handlers
        )
        if not exists_file:
            fh = _FlushFileHandler(path, mode="w" if overwrite_file else "a", encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)

    for logger_name in quiet_loggers or ("httpx", "httpcore", "huggingface_hub"):
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.WARNING)
        lg.propagate = True
