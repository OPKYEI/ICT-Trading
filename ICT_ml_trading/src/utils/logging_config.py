# src/utils/logging_config.py

import logging
import os
from typing import Optional

def configure_logging(
    name: str,
    level: str = 'INFO',
    log_file: Optional[str] = None,
    fmt: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Configure and return a logger.

    Args:
        name: Logger name (e.g., __name__).
        level: Logging level as string ('DEBUG', 'INFO', etc.).
        log_file: Optional filepath to write logs to.
        fmt: Format string for log messages.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())

    # If no handlers have been added to this logger yet, add them
    if not logger.handlers:
        formatter = logging.Formatter(fmt)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level.upper())
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(level.upper())
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
