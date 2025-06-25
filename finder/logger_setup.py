import logging
import sys
import os
from logging.handlers import RotatingFileHandler
# import finder.config as config # Config will be passed as an argument

_logging_configured = False
# Standard format for detailed logs
DETAILED_FORMAT = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s (%(module)s.%(funcName)s:%(lineno)d) - %(message)s'
SIMPLE_FORMAT = '%(message)s'

def setup_logging(config_obj): # Accept config object
    global _logging_configured
    if _logging_configured:
        return

    log_level_str = getattr(config_obj, "APP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Ensure LOG_DIR from config exists (used by file handlers)
    # This is now done in config.py itself, but double check doesn't hurt.
    os.makedirs(config_obj.LOG_DIR, exist_ok=True)

    # --- Root Logger Configuration ---
    # Configure root logger with console and main app file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console Handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(DETAILED_FORMAT, datefmt='%Y-%m-%d %H:%M:%S'))
    root_logger.addHandler(console_handler)

    # Main App File Handler (app.log)
    app_main_log_file = config_obj.APP_MAIN_LOG_FILE_PATH
    os.makedirs(os.path.dirname(app_main_log_file), exist_ok=True)
    app_file_handler = RotatingFileHandler(
        filename=app_main_log_file,
        maxBytes=config_obj.APP_MAIN_LOG_MAX_BYTES,
        backupCount=config_obj.APP_MAIN_LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    app_file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT, datefmt='%Y-%m-%d %H:%M:%S'))
    root_logger.addHandler(app_file_handler)

    # --- CheckedWalletsLogger (checked.txt) ---
    checked_logger = logging.getLogger('CheckedWalletsLogger')
    checked_logger.setLevel(logging.INFO) # Typically INFO for data logs
    checked_logger.propagate = False # Do not pass to root logger's handlers

    checked_file_path = config_obj.CHECKED_WALLETS_FILE_PATH
    os.makedirs(os.path.dirname(checked_file_path), exist_ok=True)
    checked_file_handler = RotatingFileHandler(
        filename=checked_file_path,
        maxBytes=config_obj.CHECKED_LOG_MAX_BYTES,
        backupCount=config_obj.CHECKED_LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    checked_file_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT)) # Just the message
    checked_logger.addHandler(checked_file_handler)

    # --- FoundWalletsLogger (found.txt) ---
    found_logger = logging.getLogger('FoundWalletsLogger')
    found_logger.setLevel(logging.INFO)
    found_logger.propagate = False

    found_file_path = config_obj.FOUND_WALLETS_FILE_PATH
    os.makedirs(os.path.dirname(found_file_path), exist_ok=True)
    found_file_handler = RotatingFileHandler( # Changed to RotatingFileHandler
        filename=found_file_path,
        maxBytes=config_obj.FOUND_LOG_MAX_BYTES, # Use new config values
        backupCount=config_obj.FOUND_LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    found_file_handler.setFormatter(logging.Formatter(SIMPLE_FORMAT)) # Just the message
    found_logger.addHandler(found_file_handler)

    # --- RL Debug Logger (debug_rl_agent.log) ---
    # This logger can be used by PPOAgent or other RL components for verbose debug messages
    rl_debug_logger = logging.getLogger('RLDebugLogger')
    # Set its level potentially lower if needed, e.g., DEBUG, controlled by a new config var if desired
    rl_debug_logger.setLevel(getattr(logging, config_obj.APP_LOG_LEVEL, logging.INFO)) # Or a specific RL_LOG_LEVEL
    rl_debug_logger.propagate = False # Keep RL debug separate

    rl_debug_log_file = config_obj.DEBUG_RL_AGENT_LOG_FILE_PATH
    os.makedirs(os.path.dirname(rl_debug_log_file), exist_ok=True)
    rl_debug_file_handler = RotatingFileHandler(
        filename=rl_debug_log_file,
        maxBytes=config_obj.DEBUG_RL_LOG_MAX_BYTES,
        backupCount=config_obj.DEBUG_RL_LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    rl_debug_file_handler.setFormatter(logging.Formatter(DETAILED_FORMAT, datefmt='%Y-%m-%d %H:%M:%S'))
    rl_debug_logger.addHandler(rl_debug_file_handler)

    _logging_configured = True
    root_logger.info(f"Logging configured. Root level: {log_level_str}.")
    root_logger.info(f"Main application log: {app_main_log_file}")
    root_logger.info(f"CheckedWalletsLogger (checked.txt): {checked_file_path}")
    root_logger.info(f"FoundWalletsLogger (found.txt): {found_file_path}")
    root_logger.info(f"RLDebugLogger (debug_rl_agent.log): {rl_debug_log_file}")

# Example of how other modules should get their logger:
# import logging
# logger = logging.getLogger(__name__)
# This ensures they use the root configuration set up by setup_logging().

```
