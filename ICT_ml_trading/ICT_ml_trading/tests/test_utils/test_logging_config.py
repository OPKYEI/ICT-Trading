# tests/test_utils/test_logging_config.py

import os
import logging
import pytest

from src.utils.logging_config import configure_logging

def test_configure_console_logger(caplog):
    caplog.set_level(logging.DEBUG)
    logger = configure_logging('test_console_logger', level='DEBUG')
    assert isinstance(logger, logging.Logger)
    logger.debug('debug message')
    assert 'debug message' in caplog.text

def test_configure_file_logger(tmp_path):
    log_path = tmp_path / 'logs' / 'app.log'
    logger = configure_logging('test_file_logger', level='INFO', log_file=str(log_path))
    logger.info('info message')
    # Ensure file was created
    assert os.path.exists(str(log_path))
    # Verify content
    with open(log_path, 'r') as f:
        content = f.read()
    assert 'info message' in content
