import os
import time
import logging
import multiprocessing
from mnemonic import Mnemonic as MnemonicGeneratorLib # Using 'mnemonic' library

from bip_utils import Bip44Coins, Bip39SeedGenerator, Bip44, Bip44Changes, Bip39Languages # Added for new address gen function

# Get a logger instance for this module. Configuration is handled by the main application.
logger = logging.getLogger(__name__)


# This state file is for the raw mnemonic generator's own tracking if it were to run independently
# and save its own progress of *generated* mnemonics.
# BalanceChecker in app.py uses a different state file for *processed* mnemonics.
# To avoid confusion, ensure these are distinct if both are used.
# The MnemonicGeneratorManager here is now passed config, so it can use a path from there.
# GENERATOR_STATE_FILE = "generator_state.txt" # From config now

def load_generator_index_from_file(state_file_path: str): # Renamed for clarity
    """Loads the initial generation index from the specified state file."""
    try:
        with open(state_file_path, "r") as f:
            content = f.read().strip()
            idx = int(content)
            logger.info(f"Loaded generator index {idx} from {state_file_path}")
            return idx
    except FileNotFoundError:
        logger.info(f"Generator state file {state_file_path} not found, starting from index 0.")
    except ValueError:
        logger.error(f"Invalid content in generator state file {state_file_path}, starting from index 0.")
    return 0

def save_generator_index_to_file(index: int, state_file_path: str): # Renamed for clarity
    """Saves the current generation index to the specified state file."""
    try:
        os.makedirs(os.path.dirname(state_file_path), exist_ok=True) # Ensure dir exists
        with open(state_file_path, "w") as f:
            f.write(str(index))
        logger.debug(f"Saved generator index {index} to {state_file_path}")
    except IOError as e:
        logger.error(f"Failed to save generator state to {state_file_path}: {e}")


# This function is now the primary address derivation logic
def generate_addresses_with_paths(mnemonic: str, coins_to_check: list[Bip44Coins], num_children: int, account_idx: int, change_val: int) -> dict:
    """
    Generates multiple derived addresses for specified coins from a mnemonic.
    Uses bip_utils for derivation.
    Args:
        mnemonic: The BIP39 mnemonic phrase.
        coins_to_check: List of Bip44Coins enum members (e.g., [Bip44Coins.BITCOIN, Bip44Coins.ETHEREUM]).
        num_children: Number of child addresses to derive for each coin (e.g., 10).
        account_idx: The account index for derivation (e.g., 0).
        change_val: The integer value for change type (0 for external, 1 for internal).
    Returns:
        A dictionary where keys are Bip44Coins members and values are lists of address info dictionaries.
        Each address info dict: {"path": "m/44'/...", "address": "1ABC..."}
    """
    addresses_by_coin = {}
    try:
        seed_bytes = Bip39SeedGenerator(mnemonic, Bip39Languages.ENGLISH).Generate()
        actual_change_type = Bip44Changes(change_val)

        for coin_type in coins_to_check:
            coin_addresses = []
            bip44_mst_ctx = Bip44.FromSeed(seed_bytes, coin_type)
            bip44_acc_ctx = bip44_mst_ctx.Purpose().Coin().Account(account_idx)
            bip44_chg_ctx = bip44_acc_ctx.Change(actual_change_type)

            for i in range(num_children):
                bip44_addr_ctx = bip44_chg_ctx.AddressIndex(i)
                address_str = bip44_addr_ctx.PublicKey().ToAddress()
                derivation_path = bip44_addr_ctx.DerivationPath().ToStr()
                coin_addresses.append({"path": derivation_path, "address": address_str})

            addresses_by_coin[coin_type] = coin_addresses

    except Exception as e:
        # Log to the module's logger instance
        logger.error(f"Address generation failed for mnemonic '{mnemonic[:15]}...': {e}", exc_info=True)
    return addresses_by_coin


def _mnemonic_generation_worker(output_queue: multiprocessing.Queue,
                                start_index: int,
                                mnemonics_per_worker_chunk: int, # How many this worker should try to generate
                                shutdown_event: multiprocessing.Event,
                                strength: int = 256):
    """
    Worker process function to generate a chunk of mnemonics.
    Puts (index, mnemonic_phrase) tuples onto the output_queue.
    """
    process_name = multiprocessing.current_process().name
    logger.info(f"{process_name} starting generation from index {start_index}, aiming for {mnemonics_per_worker_chunk} mnemonics.")

    mgen = MnemonicGeneratorLib("english")
    generated_count = 0
    current_idx = start_index

    try:
        for i in range(mnemonics_per_worker_chunk):
            if shutdown_event.is_set():
                logger.info(f"{process_name} received shutdown signal, stopping early.")
                break

            mnemonic_phrase = mgen.generate(strength=strength)
            # This worker puts (index, phrase) which BalanceChecker might not expect.
            # The MnemonicGeneratorManager._generation_worker_loop (older version) put only phrase.
            # Let's make this worker put only the phrase, and the manager handles indexing/state.
            # output_queue.put((current_idx, mnemonic_phrase))
            output_queue.put(mnemonic_phrase) # Simpler: just put phrase

            current_idx += 1
            generated_count += 1
            if generated_count % 10000 == 0: # Log progress less frequently for workers
                logger.debug(f"{process_name} generated {generated_count} mnemonics. Current local index: {current_idx}")

    except KeyboardInterrupt: # Should be handled by main process signal to shutdown_event
        logger.info(f"{process_name} received KeyboardInterrupt directly, stopping.")
    except Exception as e:
        logger.error(f"{process_name} encountered an error: {e}", exc_info=True)
    finally:
        logger.info(f"{process_name} finished. Generated in this run: {generated_count}. Final local index: {current_idx}")


class MnemonicGeneratorManager:
    def __init__(self, config, num_workers: int = 0, load_state: bool = True): # Added config, load_state
        self.config_obj = config # Store the config object
        self.output_queue = None # Will be set in start_generation or passed if always same
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count() or 1
        self.processes = []

        self.state_file_path = self.config_obj.GENERATOR_STATE_FILE_RAW # Path for raw generated index
        self.current_raw_generated_index = 0
        if load_state:
            self.current_raw_generated_index = load_generator_index_from_file(self.state_file_path)

        self._shutdown_event = multiprocessing.Event()
        self.is_running = False

        logger.info(f"MnemonicGeneratorManager initialized with {self.num_workers} worker(s).")
        logger.info(f"Raw generator index starts at: {self.current_raw_generated_index} (from {self.state_file_path if load_state else 'default 0'})")

    def start_generation(self, output_queue: multiprocessing.Queue, start_index: int = 0): # start_index for overall progress
        """
        Starts the mnemonic generation worker processes.
        `start_index` here refers to the index of *processed* mnemonics, used by BalanceChecker.
        This manager's internal `current_raw_generated_index` tracks mnemonics *pushed to queue*.
        """
        if self.is_running:
            logger.warning("Generator processes already started.")
            return

        self.output_queue = output_queue # Set the queue to be used
        self._shutdown_event.clear()
        self.is_running = True

        # The `start_index` passed by BalanceChecker is for its processed count.
        # This MnemonicGeneratorManager should try to generate ahead if its own raw index is behind.
        # For simplicity, let its workers run and fill the queue.
        # The crucial part is that BalanceChecker pulls and knows *which* mnemonic it's processing.
        # If `_generation_worker_loop` puts only phrases, BalanceChecker assigns indices.

        for i in range(self.num_workers):
            process = multiprocessing.Process(
                target=self._generation_worker_loop_internal, # Use the manager's internal loop
                name=f"MnemonicGen-{i}",
                args=(self.config_obj.MNEMONIC_STRENGTH,) # Pass strength from config
            )
            self.processes.append(process)
            process.start()
        logger.info(f"Started {len(self.processes)} mnemonic generator worker processes.")

    def _generation_worker_loop_internal(self, strength: int): # Renamed from _generation_worker_loop
        """Internal loop for each worker process to generate mnemonics."""
        process_name = multiprocessing.current_process().name
        # Ensure BIP39 wordlists are found, especially when bundled by PyInstaller
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            bundle_dir = sys._MEIPASS
            os.environ["BIP39_WORDLISTS_PATH"] = os.path.join(bundle_dir, "bip_utils", "bip", "bip39", "wordlist")
            logger.debug(f"{process_name}: Set BIP39_WORDLISTS_PATH for PyInstaller: {os.environ['BIP39_WORDLISTS_PATH']}")

        mgen = MnemonicGeneratorLib("english")
        logger.info(f"{process_name} starting generation with strength {strength}.")

        generated_this_worker = 0
        try:
            while not self._shutdown_event.is_set():
                mnemonic_phrase = mgen.generate(strength=strength)
                try:
                    self.output_queue.put(mnemonic_phrase, timeout=0.1) # Small timeout to prevent indefinite block
                    generated_this_worker +=1
                    if generated_this_worker % 20000 == 0: # Log less frequently from workers
                        logger.debug(f"{process_name} has put {generated_this_worker} mnemonics to queue.")
                except multiprocessing.queues.Full:
                    # Queue is full, wait a bit before trying again
                    if self._shutdown_event.wait(timeout=0.05): break # Check shutdown event while waiting
                    continue
                # Optional: add a small sleep if CPU usage is too high and generation is too fast
                # if self.config_obj.MNEMONIC_GENERATOR_WORKER_SLEEP > 0:
                #    time.sleep(self.config_obj.MNEMONIC_GENERATOR_WORKER_SLEEP)
        except KeyboardInterrupt: # Should be handled by main process signal
            logger.info(f"{process_name} received KeyboardInterrupt directly, exiting loop.")
        except Exception as e:
            logger.error(f"{process_name} error in generation loop: {e}", exc_info=True)
        finally:
            logger.info(f"{process_name} stopping generation loop. Put {generated_this_worker} mnemonics this run.")

    def stop_generation(self): # Renamed from shutdown for consistency
        """Signals all worker processes to shut down and waits for them."""
        if not self.is_running:
            logger.info("Mnemonic generator manager already stopped or not started.")
            return

        logger.info("Shutting down mnemonic generator manager...")
        self._shutdown_event.set()

        # Help drain the queue by putting sentinels if workers are blocked on put
        # This is tricky; workers might exit before seeing sentinel if queue was full.
        # Relying on timeout in put and shutdown_event check is primary.

        for p in self.processes:
            try:
                p.join(timeout=self.config_obj.MNEMONIC_WORKER_JOIN_TIMEOUT_SECONDS)
                if p.is_alive():
                    logger.warning(f"Process {p.name} did not terminate gracefully, forcing termination.")
                    p.terminate() # Force if stuck
                    p.join(timeout=2) # Wait for termination
            except Exception as e:
                logger.error(f"Error shutting down process {p.name}: {e}")

        self.processes = []
        self.is_running = False
        # The manager itself doesn't save the raw generated index here.
        # If that state is needed, it would be managed by an entity that tracks items *put* by workers.
        # For now, BalanceChecker manages the *processed* index from what it *gets*.
        logger.info("Mnemonic generator manager shut down complete.")


# Example Usage (for testing this module directly)
# Note: This example usage needs to be updated if MnemonicGeneratorManager changes significantly
# For instance, it now requires a config object.
# if __name__ == "__main__":
#     # This example needs a mock config or a way to instantiate it.
#     # For simplicity, commenting out for now as direct execution isn't the primary use.
#     # from finder.config import AppConfig # Assuming AppConfig can be imported
#     # test_config = AppConfig()
لا
#     mp_queue = multiprocessing.Queue(maxsize=10000)
#     manager = MnemonicGeneratorManager(config=test_config, num_workers=2, load_state=False)
#     manager.start_generation(output_queue=mp_queue, start_index=0)

#     generated_count = 0
#     try:
#         while True:
#             mnemonic = mp_queue.get(timeout=5)
#             generated_count += 1
#             if generated_count % 100 == 0:
#                 print(f"Main: Pulled {generated_count} mnemonics from queue.")
#             if generated_count >= 500:
#                 print("Main: Reached test limit, stopping.")
#                 break
#     except multiprocessing.queues.Empty:
#         print("Main: Queue empty after timeout, stopping.")
#     except KeyboardInterrupt:
#         print("Main: Interrupted. Shutting down.")
#     finally:
#         manager.stop_generation()
#         print(f"Main: Total mnemonics processed by example: {generated_count}")
    """Loads the initial generation index from the state file."""
    try:
        with open(GENERATOR_STATE_FILE, "r") as f:
            content = f.read().strip()
            return int(content)
    except FileNotFoundError:
        logging.info("Generator state file not found, starting from index 0.")
    except ValueError:
        logging.error(f"Invalid content in generator state file, starting from index 0.")
    return 0

def save_current_index(index):
    """Saves the current generation index to the state file."""
    try:
        with open(GENERATOR_STATE_FILE, "w") as f:
            f.write(str(index))
    except IOError as e:
        logging.error(f"Failed to save generator state: {e}")

def generate_mnemonics_process(output_queue: multiprocessing.Queue, num_mnemonics_to_generate: int = 0, start_index: int = 0):
    """
    Worker process function to generate mnemonics.
    Generates a specific number of mnemonics or runs indefinitely if num_mnemonics_to_generate is 0.
    """
    process_name = multiprocessing.current_process().name
    logging.info(f"{process_name} starting generation from index {start_index}.")

    mgen = MnemonicGeneratorLib("english")
    count = 0
    current_idx = start_index

    try:
        while num_mnemonics_to_generate == 0 or count < num_mnemonics_to_generate:
            # Strength: 128 bits for 12 words, 256 bits for 24 words.
            # app.go used 256 bits.
            mnemonic_phrase = mgen.generate(strength=256)
            output_queue.put((current_idx, mnemonic_phrase))
            current_idx += 1
            count += 1
            if count % 1000 == 0: # Log progress periodically
                logging.info(f"{process_name} generated {count} mnemonics. Current overall index: {current_idx}")
    except KeyboardInterrupt:
        logging.info(f"{process_name} received KeyboardInterrupt, stopping.")
    except Exception as e:
        logging.error(f"{process_name} encountered an error: {e}")
    finally:
        logging.info(f"{process_name} finished generating. Total generated in this run: {count}. Final index: {current_idx}")
        # Signal completion by putting None or a special marker if needed by consumer,
        # or rely on process termination. For now, just log.
        # The main process will be responsible for saving the overall index.

class MnemonicGeneratorManager:
    def __init__(self, output_queue: multiprocessing.Queue, num_workers: int = 0):
        self.output_queue = output_queue
        self.num_workers = num_workers if num_workers > 0 else os.cpu_count() or 1
        self.processes = []
        self.current_global_index = load_initial_index()
        self._shutdown_event = multiprocessing.Event()

        logging.info(f"MnemonicGeneratorManager initialized with {self.num_workers} worker(s).")
        logging.info(f"Starting generation from global index: {self.current_global_index}")

    def start(self):
        """Starts the mnemonic generation worker processes."""
        if self.processes:
            logging.warning("Generator processes already started.")
            return

        for i in range(self.num_workers):
            # For simplicity, each process runs indefinitely until shutdown.
            # If we wanted to distribute a fixed total number, logic would be more complex.
            # Each process gets a unique starting index based on global index,
            # but this is tricky if they run at different speeds.
            # A simpler model for now: all contribute to the output_queue,
            # and the manager tracks the global index based on items *taken* from queue.
            # However, the current `generate_mnemonics_process` logs its own index.
            # Let's stick to a model where the manager is mostly for starting/stopping
            # and the global index is managed by the consumer of the queue or a dedicated saver.
            # For now, `generate_mnemonics_process` doesn't need a start_index from here.
            # It will be simpler if the state is managed by the main thread that pulls from queue.

            # Revised: The process itself won't manage file state. Manager will.
            # The process just generates and puts to queue.
            # The consumer of the queue will be responsible for saving state.
            # This makes the generator process dumber and more focused.

            # Let's refine the generate_mnemonics_process to not handle state directly.
            # The manager will tell it where to start if needed or it just generates.
            # For now, let it generate continuously. The BalanceChecker will handle state.

            process = multiprocessing.Process(
                target=self._generation_worker_loop,
                name=f"MnemonicGen-{i}"
            )
            self.processes.append(process)
            process.start()
        logging.info(f"Started {len(self.processes)} mnemonic generator worker processes.")

    def _generation_worker_loop(self):
        """Internal loop for each worker process to generate mnemonics."""
        process_name = multiprocessing.current_process().name
        mgen = MnemonicGeneratorLib("english")
        logging.info(f"{process_name} starting generation.")

        try:
            while not self._shutdown_event.is_set():
                mnemonic_phrase = mgen.generate(strength=256)
                self.output_queue.put(mnemonic_phrase)
                # Optional: add a small sleep if CPU usage is too high and generation is too fast
                # time.sleep(0.001)
        except KeyboardInterrupt:
            logging.info(f"{process_name} received KeyboardInterrupt, stopping generation loop.")
        except Exception as e:
            logging.error(f"{process_name} error in generation loop: {e}", exc_info=True)
        finally:
            logging.info(f"{process_name} stopping generation loop.")

    def shutdown(self):
        """Signals all worker processes to shut down and waits for them."""
        logging.info("Shutting down mnemonic generator manager...")
        self._shutdown_event.set()
        for p in self.processes:
            try:
                # Wait for a bit, then terminate if stuck
                p.join(timeout=5)
                if p.is_alive():
                    logging.warning(f"Process {p.name} did not terminate gracefully, forcing termination.")
                    p.terminate()
                    p.join(timeout=2) # Wait for termination
            except Exception as e:
                logging.error(f"Error shutting down process {p.name}: {e}")
        self.processes = []
        logging.info("Mnemonic generator manager shut down complete.")

# Example Usage (for testing this module directly)
if __name__ == "__main__":
    mp_queue = multiprocessing.Queue(maxsize=10000) # Maxsize similar to asyncio.Queue
    manager = MnemonicGeneratorManager(output_queue=mp_queue, num_workers=2)
    manager.start()

    generated_count = 0
    try:
        while True:
            mnemonic = mp_queue.get(timeout=5) # Wait for 5 seconds
            # print(f"Got mnemonic: {mnemonic}")
            generated_count += 1
            if generated_count % 100 == 0:
                print(f"Main: Pulled {generated_count} mnemonics from queue.")
            if generated_count >= 5000: # Stop after some count for testing
                print("Main: Reached test limit, stopping.")
                break
            # Simulate work
            # time.sleep(0.01)
    except multiprocessing.queues.Empty:
        print("Main: Queue empty after timeout, stopping.")
    except KeyboardInterrupt:
        print("Main: Interrupted. Shutting down.")
    finally:
        manager.shutdown()
        print(f"Main: Total mnemonics processed: {generated_count}")
        # In a real app, save current_global_index here.
        # For this test, it's just a local count.
```
