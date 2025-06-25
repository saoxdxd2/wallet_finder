import logging
import sys
import os
from logging.handlers import RotatingFileHandler
# No direct import of config here; it will be passed as an argument.

_logging_configured = False # Module-level flag to prevent multiple configurations

# Standardized Log Formats
DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s (%(module)s.%(funcName)s:%(lineno)d) - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
SIMPLE_FORMATTER = logging.Formatter(
    '%(asctime)s - %(message)s', # Keep timestamp for simple format as well for data logs
    datefmt='%Y-%m-%d %H:%M:%S'
)
# For JSONL logs, the message itself is the JSON string, so formatter might just be '%(message)s'
JSON_FORMATTER = logging.Formatter('%(message)s')


def setup_logging(config_obj, app_name="App"): # Accept config object and optional app_name
    global _logging_configured
    if _logging_configured:
        logging.getLogger(__name__).warning("Logging already configured. Skipping re-configuration.")
        return

    # Determine log level from config object
    log_level_str = getattr(config_obj, "APP_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Ensure LOG_DIR from config exists (config.py should also create it, but good practice)
    # All log file paths from config_obj are expected to be absolute or resolvable.
    # The RotatingFileHandler will create the file, but not the directory.
    # So, we ensure directories for all configured log files.
    log_files_paths = [
        config_obj.APP_MAIN_LOG_FILE_PATH,
        config_obj.CHECKED_WALLETS_FILE_PATH,
        config_obj.FOUND_WALLETS_FILE_PATH,
        config_obj.DEBUG_RL_AGENT_LOG_FILE_PATH,
    ]
    for log_file_path in log_files_paths:
        log_file_dir = os.path.dirname(log_file_path)
        if not os.path.exists(log_file_dir):
            try:
                os.makedirs(log_file_dir, exist_ok=True)
            except OSError as e:
                # Fallback to console logging if directory creation fails for some reason
                sys.stderr.write(f"Error creating log directory {log_file_dir}: {e}. File logging for this path might fail.\n")


    # --- Root Logger Configuration ---
    # Configure root logger. Other loggers will inherit its level if not set explicitly.
    # Add console handler to root by default for general visibility.
    root_logger = logging.getLogger() # Get the root logger
    root_logger.setLevel(log_level) # Set its level

    # Console Handler (stdout) for general application messages
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(DETAILED_FORMATTER)
    # Set level for console handler (e.g., could be different from file log level if desired)
    console_handler.setLevel(log_level) # Use same level as root for now
    root_logger.addHandler(console_handler)

    # --- Main Application File Handler (e.g., app_main.log) ---
    # This handler is also added to the root logger, so all logs (unless propagated=False) go here.
    try:
        app_main_file_handler = RotatingFileHandler(
            filename=config_obj.APP_MAIN_LOG_FILE_PATH,
            maxBytes=config_obj.APP_MAIN_LOG_MAX_BYTES,
            backupCount=config_obj.APP_MAIN_LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        app_main_file_handler.setFormatter(DETAILED_FORMATTER)
        app_main_file_handler.setLevel(log_level) # Main app log captures at root's level
        root_logger.addHandler(app_main_file_handler)
    except Exception as e:
        root_logger.error(f"Failed to initialize main app file handler for {config_obj.APP_MAIN_LOG_FILE_PATH}: {e}", exc_info=True)


    # --- CheckedWalletsLogger (e.g., checked_wallets.jsonl) ---
    # This is a dedicated logger for specific data, not for general app messages.
    checked_logger = logging.getLogger('CheckedWalletsLogger') # Specific name for this logger
    checked_logger.setLevel(logging.INFO) # Data logs are usually INFO
    checked_logger.propagate = False # Do not pass these logs to root logger's handlers (console, app_main.log)
    try:
        checked_file_handler = RotatingFileHandler(
            filename=config_obj.CHECKED_WALLETS_FILE_PATH,
            maxBytes=config_obj.CHECKED_LOG_MAX_BYTES,
            backupCount=config_obj.CHECKED_LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        checked_file_handler.setFormatter(JSON_FORMATTER) # Message is expected to be a JSON string
        checked_logger.addHandler(checked_file_handler)
    except Exception as e:
        root_logger.error(f"Failed to initialize CheckedWalletsLogger file handler for {config_obj.CHECKED_WALLETS_FILE_PATH}: {e}", exc_info=True)


    # --- FoundWalletsLogger (e.g., found_wallets.jsonl) ---
    found_logger = logging.getLogger('FoundWalletsLogger')
    found_logger.setLevel(logging.INFO)
    found_logger.propagate = False
    try:
        found_file_handler = RotatingFileHandler(
            filename=config_obj.FOUND_WALLETS_FILE_PATH,
            maxBytes=config_obj.FOUND_LOG_MAX_BYTES,
            backupCount=config_obj.FOUND_LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        found_file_handler.setFormatter(JSON_FORMATTER) # Message is expected to be a JSON string
        found_logger.addHandler(found_file_handler)
    except Exception as e:
        root_logger.error(f"Failed to initialize FoundWalletsLogger file handler for {config_obj.FOUND_WALLETS_FILE_PATH}: {e}", exc_info=True)

    # --- RL Debug Logger (e.g., debug_rl_agent.log) ---
    # For verbose PPO/SB3 internal messages or detailed RL agent decision logs.
    rl_debug_logger = logging.getLogger('RLDebugLogger')
    # RL Debug Logger level can be controlled independently, e.g., DEBUG if APP_LOG_LEVEL is INFO
    # For now, let's use the main app log level, but could be a specific config var.
    rl_debug_logger_level_str = getattr(config_obj, "RL_DEBUG_LOG_LEVEL", log_level_str) # Example: New config option
    rl_debug_logger_level = getattr(logging, rl_debug_logger_level_str, log_level)
    rl_debug_logger.setLevel(rl_debug_logger_level)
    rl_debug_logger.propagate = False # Keep RL debug logs separate from main app log if desired
                                     # Or set to True if you want them in app_main.log as well.
    try:
        rl_debug_file_handler = RotatingFileHandler(
            filename=config_obj.DEBUG_RL_AGENT_LOG_FILE_PATH,
            maxBytes=config_obj.DEBUG_RL_LOG_MAX_BYTES,
            backupCount=config_obj.DEBUG_RL_LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        rl_debug_file_handler.setFormatter(DETAILED_FORMATTER) # Detailed format for debug logs
        rl_debug_logger.addHandler(rl_debug_file_handler)
    except Exception as e:
        root_logger.error(f"Failed to initialize RLDebugLogger file handler for {config_obj.DEBUG_RL_AGENT_LOG_FILE_PATH}: {e}", exc_info=True)

    _logging_configured = True
    root_logger.info(f"Logging configured for '{app_name}'. Root level: {log_level_str}.")
    root_logger.info(f"Main application log: {config_obj.APP_MAIN_LOG_FILE_PATH}")
    root_logger.info(f"Checked Wallets log (JSONL): {config_obj.CHECKED_WALLETS_FILE_PATH}")
    root_logger.info(f"Found Wallets log (JSONL): {config_obj.FOUND_WALLETS_FILE_PATH}")
    root_logger.info(f"RL Debug log: {config_obj.DEBUG_RL_AGENT_LOG_FILE_PATH} (Level: {rl_debug_logger_level_str})")

# How other modules should get their logger:
# import logging
# logger = logging.getLogger(__name__) # Uses parent's (root) config if not specifically configured
# checked_specific_logger = logging.getLogger('CheckedWalletsLogger') # Gets the specific logger
# rl_debug_specific_logger = logging.getLogger('RLDebugLogger')

```
