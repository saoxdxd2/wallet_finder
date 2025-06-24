import logging
import sys
import os
from logging.handlers import RotatingFileHandler
import finder.config as config

_logging_configured = False

def setup_logging():
    global _logging_configured
    if _logging_configured:
        return

    log_level_str = getattr(config, "APP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s (%(module)s.%(funcName)s:%(lineno)d) - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    checked_logger = logging.getLogger('CheckedWalletsLogger')
    checked_logger.setLevel(logging.INFO)
    checked_logger.propagate = False

    checked_file_dir = os.path.dirname(config.CHECKED_WALLETS_FILE_PATH)
    if checked_file_dir and not os.path.exists(checked_file_dir):
         os.makedirs(checked_file_dir, exist_ok=True)

    checked_file_handler = RotatingFileHandler(
        filename=config.CHECKED_WALLETS_FILE_PATH,
        maxBytes=config.CHECKED_LOG_MAX_BYTES,
        backupCount=config.CHECKED_LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    checked_file_handler.setFormatter(logging.Formatter('%(message)s'))
    checked_logger.addHandler(checked_file_handler)

    found_logger = logging.getLogger('FoundWalletsLogger')
    found_logger.setLevel(logging.INFO)
    found_logger.propagate = False

    found_file_dir = os.path.dirname(config.FOUND_WALLETS_FILE_PATH)
    if found_file_dir and not os.path.exists(found_file_dir):
        os.makedirs(found_file_dir, exist_ok=True)

    found_file_handler = logging.FileHandler(config.FOUND_WALLETS_FILE_PATH, mode='a', encoding='utf-8')
    found_file_handler.setFormatter(logging.Formatter('%(message)s'))
    found_logger.addHandler(found_file_handler)

    _logging_configured = True
    logging.info(f"Logging configured. Root level: {log_level_str}.")
    logging.info(f"CheckedWalletsLogger writing to: {config.CHECKED_WALLETS_FILE_PATH}")
    logging.info(f"FoundWalletsLogger writing to: {config.FOUND_WALLETS_FILE_PATH}")

```
