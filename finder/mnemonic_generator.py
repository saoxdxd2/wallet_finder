import os
import sys # For sys.frozen and sys._MEIPASS (PyInstaller)
import time
import logging
import multiprocessing
from mnemonic import Mnemonic as MnemonicGeneratorLib # Using 'mnemonic' library

from bip_utils import (
    Bip44Coins, Bip39SeedGenerator, Bip44, Bip44Changes, Bip39Languages,
    Bip44Levels # For path validation/construction if needed
)

# Get a logger instance for this module. Configuration is handled by the main application.
logger = logging.getLogger(__name__) # Standard way to get logger

# Helper functions for loading/saving generator state (index)
# These are used by MnemonicGeneratorManager for its *raw* generation index,
# and by app.py (BalanceChecker) for its *processed* mnemonic index.
# Ensure distinct file paths are used.

def load_generator_index_from_file(state_file_path: str, default_index: int = 0) -> int:
    """Loads an index from the specified state file, returning default_index if not found or invalid."""
    try:
        # Ensure directory for state file exists before trying to read
        # This is more relevant for saving, but good practice if file might be moved.
        # os.makedirs(os.path.dirname(state_file_path), exist_ok=True) # Not strictly needed for load
        if not os.path.exists(state_file_path):
            logger.info(f"Generator state file {state_file_path} not found, starting from index {default_index}.")
            return default_index
        with open(state_file_path, "r") as f:
            content = f.read().strip()
            idx = int(content)
            logger.info(f"Loaded generator index {idx} from {state_file_path}")
            return idx
    except ValueError:
        logger.error(f"Invalid content in generator state file {state_file_path}, using index {default_index}.")
    except Exception as e:
        logger.error(f"Error loading generator state from {state_file_path}: {e}. Using index {default_index}.")
    return default_index

def save_generator_index_to_file(index: int, state_file_path: str):
    """Saves the current index to the specified state file."""
    try:
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True) # Ensure directory exists
        with open(state_file_path, "w") as f:
            f.write(str(index))
        logger.debug(f"Saved generator index {index} to {state_file_path}")
    except IOError as e:
        logger.error(f"Failed to save generator state to {state_file_path}: {e}", exc_info=True)


# Primary address derivation logic (seems robust, used by app.py)
def generate_addresses_with_paths(
    mnemonic: str,
    coins_to_check: list[Bip44Coins],
    num_children: int,
    account_idx: int,
    change_enum_val: Bip44Changes, # Expecting Bip44Changes enum directly
    mnemonic_language_str: str = "english" # Added language parameter
    ) -> dict:
    """
    Generates multiple derived addresses for specified coins from a mnemonic.
    Args:
        mnemonic: The BIP39 mnemonic phrase.
        coins_to_check: List of Bip44Coins enum members.
        num_children: Number of child addresses to derive for each coin.
        account_idx: The account index for derivation.
        change_enum_val: Bip44Changes enum member (e.g., Bip44Changes.CHAIN_EXT).
        mnemonic_language_str: The language of the mnemonic (e.g., "english").
    Returns:
        A dictionary where keys are coin names (str from enum) and values are lists of address info dicts.
        Each address info dict: {"path": "m/44'/...", "address": "1ABC..."}
    """
    addresses_by_coin_name = {}
    try:
        # Convert string language to Bip39Languages enum
        try:
            bip39_lang_enum = Bip39Languages[mnemonic_language_str.upper()]
        except KeyError:
            logger.warning(f"Invalid mnemonic language string '{mnemonic_language_str}', defaulting to ENGLISH.")
            bip39_lang_enum = Bip39Languages.ENGLISH

        seed_bytes = Bip39SeedGenerator(mnemonic, bip39_lang_enum).Generate()

        for coin_type_enum in coins_to_check:
            coin_addresses_list = []
            # Create Bip44 object from seed and coin type
            bip44_mst_ctx = Bip44.FromSeed(seed_bytes, coin_type_enum)
            # Derive up to AddressIndex level: m / purpose' / coin_type' / account' / change / address_index
            # Purpose is fixed (44) by Bip44 class. Coin is from coin_type_enum.
            bip44_acc_ctx = bip44_mst_ctx.Purpose().Coin().Account(account_idx)
            bip44_chg_ctx = bip44_acc_ctx.Change(change_enum_val) # Use the enum directly

            for i in range(num_children):
                bip44_addr_ctx = bip44_chg_ctx.AddressIndex(i)
                address_str = bip44_addr_ctx.PublicKey().ToAddress()
                derivation_path = bip44_addr_ctx.DerivationPath().ToStr() # Get full path string
                coin_addresses_list.append({"path": derivation_path, "address": address_str})

            # Use coin_type_enum.name as key for better consistency with Bip44Coins usage in app.py
            addresses_by_coin_name[coin_type_enum.name] = coin_addresses_list

    except Exception as e:
        logger.error(f"Address generation failed for mnemonic (first 15 chars: '{mnemonic[:15]}...'): {e}", exc_info=True)
        # Return partially filled dict or empty if error was early
    return addresses_by_coin_name


class MnemonicGeneratorManager:
    def __init__(self, config, num_workers: int = -1, load_state: bool = True): # config is the app config object
        self.config_obj = config
        self.output_queue = None # Set by start_generation

        # Use configured number of workers, or default based on CPU cores
        if num_workers <= 0: # Use config's default if num_workers is not positive
            self.num_workers = getattr(self.config_obj, "MNEMONIC_GENERATOR_WORKERS", max(1, (os.cpu_count() or 2) // 2) )
        else:
            self.num_workers = num_workers # Use provided num_workers if positive

        self.processes = []
        self._shutdown_event = multiprocessing.Event()
        self.is_running = False

        # This manager does not save/load its own raw generation index.
        # The `load_generator_index_from_file` and `save_generator_index_to_file` helpers
        # are for use by app.py (BalanceChecker) to manage the *processed* mnemonic index
        # using `config_obj.GENERATOR_STATE_FILE_APP`.
        # If this MnemonicGeneratorManager needed to track its *own* raw generated count
        # (e.g., if it were producing a finite number of mnemonics and needed to resume),
        # then it would use `config_obj.GENERATOR_STATE_FILE_RAW` here.
        # However, its current design is to generate mnemonics continuously.

        logger.info(f"MnemonicGeneratorManager initialized with {self.num_workers} worker process(es).")


    def start_generation(self, output_queue: multiprocessing.Queue): # Removed start_index, not used by this manager
        if self.is_running:
            logger.warning("Mnemonic generator processes already started.")
            return

        self.output_queue = output_queue
        self._shutdown_event.clear()
        self.is_running = True

        strength = self.config_obj.MNEMONIC_STRENGTH
        worker_sleep = self.config_obj.MNEMONIC_GENERATOR_WORKER_SLEEP
        # Get language string from config, defaulting to "english"
        language_str = getattr(self.config_obj, "MNEMONIC_LANGUAGE", "english").lower()


        for i in range(self.num_workers):
            process = multiprocessing.Process(
                target=self._generation_worker_loop_static, # Use static method
                name=f"MnemonicGen-{i+1}",
                args=(self.output_queue, self._shutdown_event, strength, worker_sleep, language_str)
            )
            self.processes.append(process)
            process.start()
        logger.info(f"Started {len(self.processes)} mnemonic generator worker processes. Strength: {strength} bits. Language: {language_str}.")

    @staticmethod
    def _generation_worker_loop_static(output_q: multiprocessing.Queue, shutdown_ev: multiprocessing.Event,
                                strength: int, worker_sleep: float, language: str): # Made static
        """
        Internal loop for each worker process to generate mnemonics continuously.
        """
        # Get a logger for the worker process. It will inherit root config from main process.
        # Or, if main app uses logger_setup, this worker's logger will also use that.
        worker_logger = logging.getLogger(f"{__name__}.Worker-{multiprocessing.current_process().name}")
        process_name = multiprocessing.current_process().name

        # Handle PyInstaller's _MEIPASS for bip_utils wordlists if necessary
        # This needs to be done inside the worker process.
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = sys._MEIPASS
            bip_utils_wordlist_path = os.path.join(bundle_dir, "bip_utils", "bip", "bip39", "wordlist")
            if os.path.exists(bip_utils_wordlist_path):
                 os.environ["BIP39_WORDLISTS_PATH"] = bip_utils_wordlist_path
                 worker_logger.debug(f"Set BIP39_WORDLISTS_PATH for PyInstaller: {os.environ['BIP39_WORDLISTS_PATH']}")
            else:
                 worker_logger.warning(f"bip_utils wordlist path not found in PyInstaller bundle: {bip_utils_wordlist_path}")

        try:
            # MnemonicGeneratorLib expects 'english', 'french', etc.
            mgen = MnemonicGeneratorLib(language=language)
            worker_logger.info(f"Starting generation. Language: {language}, Strength: {strength} bits.")
        except Exception as e: # Catch potential error if language wordlist file is missing
            worker_logger.error(f"Failed to initialize MnemonicGeneratorLib with language '{language}': {e}. Defaulting to English.")
            mgen = MnemonicGeneratorLib(language="english") # Fallback to English
            worker_logger.info(f"Using fallback English MnemonicGeneratorLib. Strength: {strength} bits.")

        generated_this_worker = 0
        log_interval = 50000 # Log progress every N mnemonics per worker

        try:
            while not shutdown_ev.is_set():
                mnemonic_phrase = mgen.generate(strength=strength)
                try:
                    # Put only the phrase; index is handled by consumer (BalanceChecker)
                    output_q.put(mnemonic_phrase, timeout=0.2) # Timeout to prevent indefinite block
                    generated_this_worker += 1
                    if generated_this_worker % log_interval == 0:
                        worker_logger.debug(f"Has put {generated_this_worker} mnemonics to queue.")
                except multiprocessing.queues.Full:
                    # Queue is full, wait briefly and check shutdown_event
                    if shutdown_ev.wait(timeout=0.1): # Check shutdown event while waiting
                        break # Exit loop if shutdown is signaled
                    continue # Try putting again

                if worker_sleep > 0:
                    time.sleep(worker_sleep) # Yield CPU slightly if configured

        except KeyboardInterrupt: # Should be caught by main process, but handle defensively
            worker_logger.info(f"Received KeyboardInterrupt directly, exiting loop.")
        except Exception as e:
            worker_logger.error(f"Critical error in generation loop: {e}", exc_info=True)
        finally:
            worker_logger.info(f"Stopping generation loop. Total mnemonics put by this worker: {generated_this_worker}.")


    def stop_generation(self): # Renamed from shutdown for consistency with app.py
        if not self.is_running:
            logger.info("Mnemonic generator manager is not running or already stopped.")
            return

        logger.info("Shutting down mnemonic generator manager and its worker processes...")
        self._shutdown_event.set()

        join_timeout = self.config_obj.MNEMONIC_WORKER_JOIN_TIMEOUT_SECONDS
        active_processes = []
        for p in self.processes:
            try:
                p.join(timeout=join_timeout)
                if p.is_alive():
                    logger.warning(f"Process {p.name} did not terminate gracefully after {join_timeout}s, forcing termination.")
                    p.terminate() # Force if stuck
                    p.join(timeout=2) # Wait for termination to complete
                    if p.is_alive():
                        logger.error(f"Process {p.name} could not be terminated.")
                    else:
                        logger.info(f"Process {p.name} terminated forcefully.")
                else:
                     logger.info(f"Process {p.name} joined gracefully.")
            except Exception as e:
                logger.error(f"Error shutting down process {p.name}: {e}", exc_info=True)
            if p.is_alive(): # Keep track of still active ones if terminate failed
                active_processes.append(p)

        self.processes = active_processes # Update list to any stubborn processes
        if not self.processes:
            self.is_running = False
            logger.info("All mnemonic generator workers shut down.")
        else:
            logger.error(f"{len(self.processes)} mnemonic generator worker(s) could not be shut down.")

# Removed old standalone functions like `_mnemonic_generation_worker` (integrated into manager's static method),
# `generate_mnemonics_process`, `load_initial_index`, `save_current_index` as they are no longer used
# in this revised structure. The load/save helpers are kept as they are used by app.py.

# Example Usage (for testing this module directly - requires a mock config)
if __name__ == "__main__":
    # This example needs a mock config object that mimics the expected attributes.
    class MockConfig:
        # Paths
        LOG_DIR = "temp_logs_mnemonic_gen" # Use a specific temp dir for this test
        # These paths are for logger_setup.py if it were called, not directly used by MnemonicGeneratorManager here.
        GENERATOR_STATE_FILE_RAW = os.path.join(LOG_DIR, "test_gen_raw_state.txt") # Not used by MGM itself
        APP_MAIN_LOG_FILE_PATH = os.path.join(LOG_DIR, "test_app_main.log")
        APP_MAIN_LOG_MAX_BYTES = 1024*1024
        APP_MAIN_LOG_BACKUP_COUNT = 1
        CHECKED_WALLETS_FILE_PATH = os.path.join(LOG_DIR, "test_checked.jsonl")
        CHECKED_LOG_MAX_BYTES = 1024*1024
        CHECKED_LOG_BACKUP_COUNT = 1
        FOUND_WALLETS_FILE_PATH = os.path.join(LOG_DIR, "test_found.jsonl")
        FOUND_LOG_MAX_BYTES = 1024*1024
        FOUND_LOG_BACKUP_COUNT = 1
        DEBUG_RL_AGENT_LOG_FILE_PATH = os.path.join(LOG_DIR, "test_debug_rl.log")
        DEBUG_RL_LOG_MAX_BYTES = 1024*1024
        DEBUG_RL_LOG_BACKUP_COUNT = 1

        # Mnemonic settings used by MnemonicGeneratorManager
        MNEMONIC_GENERATOR_WORKERS = 2
        MNEMONIC_STRENGTH = 128 # Faster for testing
        MNEMONIC_LANGUAGE = "english" # Test with english
        MNEMONIC_GENERATOR_WORKER_SLEEP = 0.005 # Sleep a bit to see context switching
        MNEMONIC_WORKER_JOIN_TIMEOUT_SECONDS = 3 # Timeout for workers to join
        APP_LOG_LEVEL = "DEBUG" # For verbose test output

    mock_config_instance = MockConfig()
    if not os.path.exists(mock_config_instance.LOG_DIR):
        os.makedirs(mock_config_instance.LOG_DIR, exist_ok=True)

    # Basic logging setup for the test (simulates what main entry point would do)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - [%(processName)s:%(threadName)s] - %(name)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    main_logger = logging.getLogger(__name__) # Logger for this __main__ block
    main_logger.info("--- Testing MnemonicGeneratorManager ---")

    # Multiprocessing queue for workers to put mnemonics into
    mp_output_queue = multiprocessing.Queue(maxsize=1000) # Test with a reasonably sized queue

    # Test MnemonicGeneratorManager
    # num_workers can be overridden here, or it will use MNEMONIC_GENERATOR_WORKERS from config
    manager = MnemonicGeneratorManager(config=mock_config_instance, load_state=False) # load_state=False as MGM is stateless for index
    manager.start_generation(output_queue=mp_output_queue)

    generated_count_main = 0
    max_to_pull = 200 # Pull a few mnemonics for testing
    pull_start_time = time.time()
    main_logger.info(f"Main process will try to pull {max_to_pull} mnemonics...")

    try:
        for i in range(max_to_pull):
            mnemonic = mp_output_queue.get(timeout=10) # Wait up to 10s for a mnemonic
            generated_count_main += 1
            if (generated_count_main % 20 == 0) or (i == max_to_pull -1) : # Log periodically and the last one
                main_logger.info(f"Main: Pulled {generated_count_main}/{max_to_pull}. Last: '{mnemonic[:20]}...'")
    except multiprocessing.queues.Empty:
        main_logger.warning("Main: Queue empty after timeout during test pull.")
    except KeyboardInterrupt:
        main_logger.info("Main: Test interrupted by user.")
    finally:
        pull_duration = time.time() - pull_start_time
        main_logger.info(f"Main: Finished pulling. Total pulled: {generated_count_main} mnemonics in {pull_duration:.2f} seconds.")

        main_logger.info("Main: Shutting down MnemonicGeneratorManager...")
        manager.stop_generation() # Call the renamed method
        main_logger.info("Main: MnemonicGeneratorManager shutdown sequence complete.")

        # Attempt to clear the queue to prevent main process hanging on exit if workers put more items
        # while shutting down or if the queue was not fully drained by the test.
        main_logger.info("Main: Clearing any remaining items from multiprocessing queue post-shutdown...")
        cleared_count = 0
        while not mp_output_queue.empty():
            try:
                mp_output_queue.get_nowait() # Non-blocking get
                cleared_count +=1
            except multiprocessing.queues.Empty: # Should not happen if not empty, but good practice
                break
            except Exception: # Catch other potential errors like EOFError if queue is already closed
                break
        if cleared_count > 0:
            main_logger.info(f"Main: Cleared an additional {cleared_count} items from queue after workers stopped.")

        main_logger.info("--- MnemonicGeneratorManager Test Complete ---")
```
