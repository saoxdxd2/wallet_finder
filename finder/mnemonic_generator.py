import os
import time
import logging
import multiprocessing
from mnemonic import Mnemonic as MnemonicGeneratorLib # Using 'mnemonic' library

# Configure basic logging for the generator module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s')

GENERATOR_STATE_FILE = "generator_state.txt"

def load_initial_index():
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
