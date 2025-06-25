import asyncio
import argparse
import multiprocessing
import os
import sys
import logging

# --- Path Setup ---
# Ensure the 'finder' module can be imported.
# This assumes 'run_scanner.py' is in the project root, and 'finder' is a subdirectory.
# If 'run_scanner.py' is inside 'finder', this might not be necessary or different.
# For now, let's assume project_root/run_scanner.py and project_root/finder/
# Adding project root to sys.path to allow `from finder import ...`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# If run_scanner.py is intended to be in the project root:
PROJECT_ROOT = CURRENT_DIR
# If run_scanner.py is one level above 'finder' (e.g. in a 'scripts' folder, and finder is sibling)
# then PROJECT_ROOT might be os.path.dirname(CURRENT_DIR)
# If finder is a direct subdir of where run_scanner.py is, then finder path is fine.
# Let's assume run_scanner.py is in the project root, same level as the 'finder' directory.
# So, direct imports like `from finder.app import main_entry_point` should work if Python's
# current working directory is the project root when `python run_scanner.py` is called.
# To be safe, one could add `finder`'s parent to sys.path if structure is complex.
# For now, relying on standard Python import mechanisms assuming CWD is project root.
# If `finder` is not found, this might need adjustment:
# sys.path.insert(0, os.path.join(CURRENT_DIR, '..')) # If run_scanner is in a subdir like 'scripts'
# Or if run_scanner.py is in project root and finder is ./finder
# sys.path.insert(0, CURRENT_DIR) # Project root


# --- Logger for this entry script ---
# Basic logger setup here for messages from run_scanner.py itself before app's logging takes over.
# The application's full logging is configured inside `main_entry_point` via `logger_setup`.
entry_logger = logging.getLogger("run_scanner_entry_point")
logging.basicConfig(level=logging.INFO, # Basic level for this script's own messages
                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])


def main():
    # freeze_support() is necessary for multiprocessing programs frozen into executables (e.g., with PyInstaller)
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Crypto Wallet Scanner Application.")
    parser.add_argument(
        "--profile-cpu",
        action="store_true",
        help="Enable CPU profiling with cProfile for the application.",
    )
    parser.add_argument(
        "--profile-mem",
        action="store_true",
        help="Enable memory profiling with tracemalloc for the application.",
    )
    # Future: Add arguments to override config values, e.g.,
    # parser.add_argument("--config-file", type=str, help="Path to a custom YAML/JSON config file.")
    # parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Override application log level.")

    cli_args = parser.parse_args()

    entry_logger.info(f"Starting {APP_NAME_PLACEHOLDER}...") # App name from config later
    entry_logger.info(f"Command line arguments: {cli_args}")

    try:
        # Dynamically import application components after path setup (if any was needed)
        # and after basic arg parsing. This allows cleaner separation.
        from finder.app import main_entry_point
        from finder.config import (
            APP_NAME, # For logging app name
            # Other specific configs could be loaded here if needed before passing app_config object
        )
        import finder.config as app_config_module # Import the whole module to pass as object

        # Update logger message with actual app name from config
        entry_logger.info(f"Starting {APP_NAME}...")


        # TODO: Implement config override logic here if --config-file or other overrides are used.
        # For example, load a YAML file and merge its settings into app_config_module.
        # For now, app_config_module is used directly as loaded from finder.config.

        # Run the main asynchronous application logic
        asyncio.run(main_entry_point(args=cli_args, app_config_obj=app_config_module))

    except ImportError as e:
        entry_logger.critical(f"Failed to import application modules. Ensure 'finder' package is in PYTHONPATH or script is run from project root. Error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        entry_logger.critical(f"An unexpected error occurred at the entry point: {e}", exc_info=True)
        sys.exit(1)

    entry_logger.info(f"{APP_NAME_PLACEHOLDER} has finished.")


if __name__ == "__main__":
    # A placeholder for APP_NAME until config is loaded, or define a default here.
    APP_NAME_PLACEHOLDER = "Crypto Wallet Scanner"
    main()
