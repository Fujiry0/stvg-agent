"""Configure the shared logger."""

import logging
import os

logging.getLogger("qwen_vl_utils").setLevel(logging.ERROR)

_LOGGER_NAME = "anygroundbench"


def set_logger(log_path: str | None = None):
    """Create or reuse the shared logger for inference runs."""
    resolved_log_path = log_path
    if resolved_log_path is not None:
        log_dir = os.path.dirname(resolved_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(_LOGGER_NAME)
    if getattr(logger, "_configured_log_path", None) == resolved_log_path and logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    if resolved_log_path is not None:
        file_handler = logging.FileHandler(resolved_log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger._configured_log_path = resolved_log_path
    return logger
