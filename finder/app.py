import os
import sys
import asyncio
import aiohttp
import json
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from bip_utils import Bip39SeedGenerator, Bip44, Bip44Coins, Bip39Languages # For Bip39SeedGenerator
from aiolimiter import AsyncLimiter
# import aiofiles # No longer needed for checked.txt, only if input_file was async
from concurrent.futures import ThreadPoolExecutor
import multiprocessing # For MnemonicGeneratorManager
import time # For DQN cycle timing & profiling
import numpy as np # For DQN states
import tracemalloc # For memory profiling
import cProfile # For performance profiling
import pstats # For processing cProfile stats
import io # For capturing cProfile output to string
import argparse # For command-line arguments

# Import the new mnemonic generator
# Modular imports from the previous refactoring (assuming they are now in place)
import finder.config as config
import finder.logger_setup as logger_setup
from finder.mnemonic_generator import MnemonicGeneratorManager, save_current_index as save_generator_index, load_initial_index as load_generator_index
from finder.dqn_agent import DQNAgent # Will be replaced for PPO where applicable
from finder.ppo_sb3_agent import PPOAgentSB3 # Import the new PPO agent
from finder.api_handler import APIHandler


if getattr(sys, 'frozen', False):
    # PyInstaller bundle path
    base_path = sys._MEIPASS
    os.environ["BIP39_WORDLISTS_PATH"] = os.path.join(base_path, "bip_utils", "bip", "bip39", "wordlist")

# Setup main application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(module)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Setup dedicated logger for results (checked.txt)
results_logger = logging.getLogger('ResultsLogger')
results_logger.setLevel(logging.INFO)
results_logger.propagate = False  # Don't pass to root logger

# Configure RotatingFileHandler for results_logger
# Rotate when checked.txt reaches 10MB, keep 5 backup files
# Ensure the 'finder' directory exists or specify full path if needed
checked_file_path = "finder/checked.txt" # Ensure this path is correct
os.makedirs(os.path.dirname(checked_file_path), exist_ok=True)

checked_file_handler = RotatingFileHandler(
    checked_file_path,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5  # Number of backup files
)
checked_file_handler.setFormatter(logging.Formatter('%(message)s')) # Raw message format
results_logger.addHandler(checked_file_handler)


API_ENDPOINTS = {
    Bip44Coins.BITCOIN: [
        "https://blockchain.info/balance?active={address}",
        "https://chain.api.btc.com/v3/address/{address}",
    ],
    Bip44Coins.ETHEREUM: [
        "https://api.etherscan.io/api?module=account&action=balance&address={address}",
        # "https://eth-mainnet.alchemyapi.io/v2/demo/balance?address={address}", # Often rate limited
    ],
    "USDT": [ # USDT on Ethereum (ERC20)
        "https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress=0xdac17f958d2ee523a2206206994597c13d831ec7&address={address}&tag=latest",
    ]
}

# Path for generator state file, ensure it's in a writable location
GENERATOR_STATE_FILE = "finder/generator_state.txt" # Note: This is also defined in mnemonic_generator.py

# DQN Related Constants
DQN_RL_CYCLE_INTERVAL_SECONDS = 30  # Rate Limiter DQN
DQN_WC_CYCLE_INTERVAL_SECONDS = 45  # Worker Count DQN (can be different)

# Rate Limiter DQN
RL_AGENT_STATE_SIZE = 4
RL_AGENT_ACTION_SIZE = 3 # Decrease, Maintain, Increase rate
MIN_RATE_LIMIT = 1.0
MAX_RATE_LIMIT = 50.0 # Sensible maximum for public APIs
RATE_ADJUSTMENT_STEP = 1.0

# Worker Count DQN
WC_AGENT_STATE_SIZE = 3 # current_workers_norm, queue_fill, avg_proc_time_norm (or throughput per worker)
WC_AGENT_ACTION_SIZE = 3 # Decrease, Maintain, Increase workers
MIN_PROCESSING_WORKERS = 1
MAX_PROCESSING_WORKERS = (os.cpu_count() or 1) * 4 # Example max: 4x CPU cores
WORKER_ADJUSTMENT_STEP = 1


class BalanceChecker:
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        self.found_file = "finder/found.txt"

        # --- Rate Limiter Attributes ---
        self.current_api_rate = 10.0
        self.limiter = AsyncLimiter(self.current_api_rate, 1)
        self.rl_agent_rate_limiter = DQNAgent(
            state_size=RL_AGENT_STATE_SIZE, action_size=RL_AGENT_ACTION_SIZE,
            agent_name="rate_limiter_agent", seed=42)
        self.rl_previous_state = None
        self.rl_previous_action = None
        self.api_calls_total_since_last_rl_cycle = 0
        self.api_errors_429_since_last_rl_cycle = 0
        self.api_timeouts_since_last_rl_cycle = 0
        self.last_rl_cycle_time = time.time()

        # --- Worker Count Attributes ---
        self.current_num_processing_workers = (os.cpu_count() or 1) * 2
        self.processing_worker_tasks = []
        self.wc_agent_worker_count = DQNAgent(
            state_size=WC_AGENT_STATE_SIZE, action_size=WC_AGENT_ACTION_SIZE,
            agent_name="worker_count_agent", seed=123)
        self.wc_previous_state = None
        self.wc_previous_action = None
        self.mnemonics_processed_last_wc_cycle = 0
        self.last_wc_cycle_time = time.time() # Corrected variable name

        # --- API Endpoint Stats ---
        self.api_endpoint_stats = {} # {coin_type: {url_template: {stats}}}
        self._initialize_api_endpoint_stats()
        self.api_stats_log_interval = 600 # Log API stats every 10 minutes
        self.last_api_stats_log_time = time.time()

        # --- Common Attributes ---
        self.async_mnemonic_queue = asyncio.Queue(maxsize=10000)
        self.coins = [Bip44Coins.BITCOIN, Bip44Coins.ETHEREUM, "USDT"]
        self.processed_mnemonic_index = load_generator_index()
        self.mnemonics_processed_in_session = 0
        self.mp_queue = multiprocessing.Queue(maxsize=10000)
        self.mnemonic_generator_manager = MnemonicGeneratorManager(output_queue=self.mp_queue)
        self._stop_event = asyncio.Event()


    def _initialize_api_endpoint_stats(self):
        for coin_type, endpoints in API_ENDPOINTS.items():
            self.api_endpoint_stats[coin_type] = {}
            for endpoint_url_template in endpoints:
                self.api_endpoint_stats[coin_type][endpoint_url_template] = {
                    "successes": 0,
                    "failures": 0, # General failures (non-200, parse errors)
                    "timeouts": 0,
                    "errors_429": 0, # Specific for rate limits
                    "total_latency_ms": 0, # Sum of latencies for successful calls
                    "latency_count": 0,    # Number of calls included in total_latency_ms
                    "score": 100.0 # Initial optimistic score
                }
        logging.info("API endpoint stats initialized.")

    async def _update_api_call_stats(self, status_code=None, is_timeout=False):
        # This function is for the Rate Limiter DQN's perspective
        self.api_calls_total_since_last_rl_cycle += 1
        if status_code == 429:
            self.api_errors_429_since_last_rl_cycle += 1
        if is_timeout: # This is timeout from the perspective of the limiter DQN
            self.api_timeouts_since_last_rl_cycle +=1

    async def _update_specific_endpoint_stats(self, coin_type, url_template, success, is_timeout, is_429, latency_ms=0):
        """Updates stats for a specific endpoint after an API call."""
        if coin_type not in self.api_endpoint_stats or \
           url_template not in self.api_endpoint_stats[coin_type]:
            logging.warning(f"Attempted to update stats for unknown endpoint: {coin_type} - {url_template}")
            return

        stats = self.api_endpoint_stats[coin_type][url_template]
        if success:
            stats["successes"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["latency_count"] += 1
        else:
            stats["failures"] += 1
            if is_timeout:
                stats["timeouts"] += 1
            if is_429:
                stats["errors_429"] += 1

        # Simple scoring: +1 for success, -2 for failure/timeout, -5 for 429
        # More sophisticated scoring can be added later.
        score_change = 0
        if success: score_change = 1
        else: score_change = -2
        if is_429: score_change = -5 # Heavier penalty for being rate-limited by this specific endpoint

        stats["score"] = max(0, stats["score"] + score_change) # Score doesn't go below 0

    async def _log_api_endpoint_stats_periodically(self):
        """Periodically logs the collected API endpoint statistics."""
        if (time.time() - self.last_api_stats_log_time) > self.api_stats_log_interval:
            logging.info("--- API Endpoint Statistics ---")
            for coin_type, endpoints_data in self.api_endpoint_stats.items():
                logging.info(f"Coin: {coin_type}")
                # Sort endpoints by score descending for readability
                sorted_endpoints = sorted(endpoints_data.items(), key=lambda item: item[1]["score"], reverse=True)
                for url_template, stats in sorted_endpoints:
                    avg_latency = (stats['total_latency_ms'] / stats['latency_count']) if stats['latency_count'] > 0 else 0
                    logging.info(
                        f"  URL: {url_template} | Score: {stats['score']:.1f} | "
                        f"S: {stats['successes']}, F: {stats['failures']}, T: {stats['timeouts']}, 429: {stats['errors_429']} | "
                        f"Avg Latency: {avg_latency:.0f}ms"
                    )
            logging.info("--- End API Endpoint Statistics ---")
            self.last_api_stats_log_time = time.time()


    async def _mp_to_asyncio_queue_adapter(self):
        logging.info("Starting multiprocessing to asyncio queue adapter.")
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                # Use run_in_executor to get from mp.Queue without blocking asyncio loop
                mnemonic_phrase = await loop.run_in_executor(self.executor, self.mp_queue.get, True, 0.1) # Timeout 0.1s
                if mnemonic_phrase:
                    await self.async_mnemonic_queue.put(mnemonic_phrase)
            except multiprocessing.queues.Empty: # Expected on timeout
                continue
            except Exception as e:
                logging.error(f"Error in mp_to_asyncio_queue_adapter: {e}")
                await asyncio.sleep(0.1) # Avoid busy loop on error
        logging.info("Multiprocessing to asyncio queue adapter stopped.")


    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit_per_host=5), # Be kinder to APIs
            headers={"User-Agent": "Mozilla/5.0"} # Standard User-Agent
        )

        # Start the Python-based mnemonic generator manager
        self.mnemonic_generator_manager.start()
        adapter_task = asyncio.create_task(self._mp_to_asyncio_queue_adapter())

        # Initialize processing workers
        self.processing_worker_tasks = []
        for _ in range(self.current_num_processing_workers):
            task = asyncio.create_task(self.process_queue())
            self.processing_worker_tasks.append(task)
        logging.info(f"Starting with {self.current_num_processing_workers} processing workers.")

        # Start DQN control loops
        dqn_rl_control_task = asyncio.create_task(self._dqn_control_loop_rate_limiter())
        dqn_wc_control_task = asyncio.create_task(self._dqn_control_loop_worker_count())

        # Consolidate all main tasks to await their completion
        # Note: self.processing_worker_tasks are managed (added/removed) dynamically by DQN
        # So, they are not directly awaited in the main gather like fixed tasks.
        # Instead, their lifecycle is tied to _stop_event and cancellations by DQN.
        # The main gather awaits tasks that run for the lifetime of the app.
        core_tasks = [adapter_task, dqn_rl_control_task, dqn_wc_control_task]

        try:
            # We also need to await the initial set of worker tasks,
            # or handle their completion/cancellation within the shutdown logic.
            # For now, let's add them to the gather, but dynamic adjustment needs care.
            # This might be problematic if workers are frequently added/removed.
            # A better approach for workers might be to manage them outside the main gather,
            # and ensure they are all stopped during the finally block.
            # For now, let's gather all tasks that are initially started.

            await asyncio.gather(*(core_tasks + self.processing_worker_tasks))

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received, shutting down BalanceChecker...")
        except Exception as e:
            logging.critical(f"Critical error in BalanceChecker main gather: {e}", exc_info=True)
        finally:
            logging.info("BalanceChecker shutting down gracefully...")
            self._stop_event.set() # This should signal all loops and workers to stop

            self.mnemonic_generator_manager.shutdown()

            # Cancel all processing worker tasks explicitly
            logging.info(f"Cancelling {len(self.processing_worker_tasks)} processing workers...")
            for task in self.processing_worker_tasks:
                if not task.done():
                    task.cancel()
            # Await their cancellation
            await asyncio.gather(*self.processing_worker_tasks, return_exceptions=True)
            logging.info("Processing workers shut down.")

            # Wait for core tasks (adapter, DQN loops) with a timeout
            # These tasks should respond to _stop_event
            done, pending = await asyncio.wait(core_tasks, timeout=DQN_RL_CYCLE_INTERVAL_SECONDS + 5.0) # Ensure DQN has time for one last save
            for task in pending:
                task_name = task.get_name() if hasattr(task, 'get_name') else "Unknown Core Task"
                logging.warning(f"Core task {task_name} did not finish in time, cancelling.")
                task.cancel()
            if pending:
                 await asyncio.gather(*pending, return_exceptions=True)

            if self.session:
                await self.session.close()
                logging.info("aiohttp session closed.")

            # Save DQN agent models
            try:
                self.rl_agent_rate_limiter.save_model()
                self.wc_agent_worker_count.save_model()
                logging.info("DQN agent models saved.")
            except Exception as e:
                logging.error(f"Failed to save one or more DQN agent models: {e}")

            if self.executor: # Shutdown executor after all tasks that might use it are done
                self.executor.shutdown(wait=True)
                logging.info("ThreadPoolExecutor shut down.")

            save_generator_index(self.processed_mnemonic_index)
            logging.info(f"Saved generator state. Last processed index: {self.processed_mnemonic_index}")
            logging.info(f"Mnemonics processed this session: {self.mnemonics_processed_in_session}")

    # --- DQN Rate Limiter Methods ---
    def _get_rate_limiter_state_rl(self) -> np.ndarray:
        # Normalize current rate
        norm_current_rate = (self.current_api_rate - MIN_RATE_LIMIT) / (MAX_RATE_LIMIT - MIN_RATE_LIMIT)
        norm_current_rate = np.clip(norm_current_rate, 0, 1)

        # Calculate error rates (since last RL cycle)
        if self.api_calls_total_since_last_rl_cycle > 0:
            error_rate_429 = self.api_errors_429_since_last_rl_cycle / self.api_calls_total_since_last_rl_cycle
            timeout_rate = self.api_timeouts_since_last_rl_cycle / self.api_calls_total_since_last_rl_cycle
        else:
            error_rate_429 = 0.0
            timeout_rate = 0.0

        # Queue fill percentage
        queue_fill = self.async_mnemonic_queue.qsize() / self.async_mnemonic_queue.maxsize if self.async_mnemonic_queue.maxsize > 0 else 0.0

        state = np.array([
            norm_current_rate,
            error_rate_429,
            timeout_rate,
            queue_fill
        ], dtype=np.float32)
        return state

    def _apply_rate_limiter_action_rl(self, action: int):
        """ Applies the action chosen by the DQN agent to the rate limiter. """
        previous_rate = self.current_api_rate
        if action == 0: # Decrease rate
            self.current_api_rate = max(MIN_RATE_LIMIT, self.current_api_rate - RATE_ADJUSTMENT_STEP)
        elif action == 1: # Maintain rate
            pass
        elif action == 2: # Increase rate
            self.current_api_rate = min(MAX_RATE_LIMIT, self.current_api_rate + RATE_ADJUSTMENT_STEP)

        if self.current_api_rate != previous_rate:
            self.limiter = AsyncLimiter(self.current_api_rate, 1) # Re-initialize limiter with new rate
            logging.info(f"RL Agent adjusted API rate from {previous_rate:.2f} to {self.current_api_rate:.2f} rps.")
        else:
            logging.info(f"RL Agent decided to maintain API rate at {self.current_api_rate:.2f} rps.")
        return self.current_api_rate # Return new rate for logging or confirmation

    def _calculate_rate_limiter_reward_rl(self, time_delta_seconds: float) -> float:
        """ Calculates reward for the rate limiter agent. """
        mnemonics_in_cycle = self.mnemonics_processed_in_session - self.mnemonics_processed_last_rl_cycle

        throughput = mnemonics_in_cycle / time_delta_seconds if time_delta_seconds > 0 else 0.0

        # Penalties should be negative
        penalty_429 = -50.0 * self.api_errors_429_since_last_rl_cycle  # Heavy penalty for 429s
        penalty_timeout = -10.0 * self.api_timeouts_since_last_rl_cycle # Moderate penalty for timeouts

        # Reward for throughput, scaled (e.g., aim for 10 processed/sec as a baseline reward of 1)
        throughput_reward = throughput * 0.1

        # Queue status consideration (small penalty for near-empty queue if trying to speed up)
        queue_penalty = 0.0
        queue_fill = self.async_mnemonic_queue.qsize() / self.async_mnemonic_queue.maxsize if self.async_mnemonic_queue.maxsize > 0 else 0
        if queue_fill < 0.1 and self.rl_previous_action == 2 : # If tried to increase rate with empty queue
             queue_penalty = -0.5
        if queue_fill > 0.95 : # Penalize nearly full queue
             queue_penalty = -1.0


        reward = throughput_reward + penalty_429 + penalty_timeout + queue_penalty
        logging.debug(f"RL Reward Calc: Throughput={throughput:.2f} ({throughput_reward:.2f}), "
                     f"429s={self.api_errors_429_since_last_rl_cycle} ({penalty_429:.2f}), "
                     f"Timeouts={self.api_timeouts_since_last_rl_cycle} ({penalty_timeout:.2f}), "
                     f"QueuePen={queue_penalty:.2f}. Total Reward: {reward:.2f}")
        return float(reward)

    async def _dqn_control_loop_rate_limiter(self):
        logging.info("DQN Rate Limiter control loop started.")
        await asyncio.sleep(DQN_CYCLE_INTERVAL_SECONDS) # Initial delay before first action

        while not self._stop_event.is_set():
            loop_start_time = time.time()

            # 1. Get current state
            current_state = self._get_rate_limiter_state_rl()

            # 2. If there was a previous state/action, calculate reward and learn
            if self.rl_previous_state is not None and self.rl_previous_action is not None:
                time_delta = loop_start_time - self.last_rl_cycle_time
                reward = self._calculate_rate_limiter_reward_rl(time_delta)

                # Run learning step in executor
                # done flag is False as this is a continuous task
                await asyncio.get_running_loop().run_in_executor(
                    self.executor,
                    self.rl_agent_rate_limiter.step,
                    self.rl_previous_state,
                    self.rl_previous_action,
                    reward,
                    current_state, # This is next_state from perspective of previous action
                    False
                )
                logging.debug(f"RL Agent learning step completed. Reward: {reward:.3f}, Epsilon: {self.rl_agent_rate_limiter.eps:.3f}")


            # 3. Agent selects an action based on current state
            # Run action selection in executor as it involves model inference
            action = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.rl_agent_rate_limiter.get_action,
                current_state
            )

            # 4. Apply action
            self._apply_rate_limiter_action_rl(action) # This updates self.limiter

            # 5. Store current state and action for next iteration's learning step
            self.rl_previous_state = current_state
            self.rl_previous_action = action

            # Reset counters for the next cycle
            self.mnemonics_processed_last_rl_cycle = self.mnemonics_processed_in_session
            self.api_calls_total_since_last_rl_cycle = 0
            self.api_errors_429_since_last_rl_cycle = 0
            self.api_timeouts_since_last_rl_cycle = 0
            self.last_rl_cycle_time = loop_start_time # Record time for next delta calculation

            # Wait for the next cycle
            # Adjust sleep time if the loop itself took significant time, though unlikely here
            await asyncio.sleep(DQN_CYCLE_INTERVAL_SECONDS)

        logging.info("DQN Rate Limiter control loop stopped.")
        # Model saving is now handled in the main finally block of BalanceChecker.start()


    # --- DQN Worker Count Methods ---
    def _get_worker_count_state_wc(self) -> np.ndarray:
        norm_worker_count = (self.current_num_processing_workers - MIN_PROCESSING_WORKERS) / \
                            (MAX_PROCESSING_WORKERS - MIN_PROCESSING_WORKERS)
        norm_worker_count = np.clip(norm_worker_count, 0, 1)

        queue_fill = self.async_mnemonic_queue.qsize() / self.async_mnemonic_queue.maxsize \
            if self.async_mnemonic_queue.maxsize > 0 else 0.0

        # Placeholder for processing time per mnemonic or throughput per worker
        # For now, using overall throughput as a rough guide.
        # A more direct measure of worker efficiency would be better.
        # Let's use queue_fill again as a simple third state for now.
        # TODO: Improve this state feature with actual worker efficiency metric.
        avg_proc_time_norm_placeholder = queue_fill # Re-using queue_fill as placeholder for 3rd state feature

        state = np.array([
            norm_worker_count,
            queue_fill,
            avg_proc_time_norm_placeholder # Placeholder
        ], dtype=np.float32)
        return state

    async def _apply_worker_count_action_wc(self, action: int):
        previous_worker_count = self.current_num_processing_workers

        if action == 0: # Decrease workers
            target_workers = max(MIN_PROCESSING_WORKERS, self.current_num_processing_workers - WORKER_ADJUSTMENT_STEP)
        elif action == 1: # Maintain workers
            target_workers = self.current_num_processing_workers
        elif action == 2: # Increase workers
            target_workers = min(MAX_PROCESSING_WORKERS, self.current_num_processing_workers + WORKER_ADJUSTMENT_STEP)
        else: # Should not happen
            logging.warning(f"WC DQN: Unknown action {action}")
            return

        tasks_to_add = target_workers - len(self.processing_worker_tasks)

        if tasks_to_add > 0:
            logging.info(f"WC DQN: Increasing workers by {tasks_to_add}. Current: {len(self.processing_worker_tasks)}, Target: {target_workers}")
            for _ in range(tasks_to_add):
                if len(self.processing_worker_tasks) < MAX_PROCESSING_WORKERS: # Double check limit
                    task = asyncio.create_task(self.process_queue())
                    self.processing_worker_tasks.append(task)
                else:
                    logging.warning("WC DQN: Max worker limit reached during increase.")
                    break
        elif tasks_to_add < 0: # Need to remove tasks
            num_to_remove = abs(tasks_to_add)
            logging.info(f"WC DQN: Decreasing workers by {num_to_remove}. Current: {len(self.processing_worker_tasks)}, Target: {target_workers}")
            for _ in range(num_to_remove):
                if self.processing_worker_tasks:
                    task_to_cancel = self.processing_worker_tasks.pop()
                    if not task_to_cancel.done():
                        task_to_cancel.cancel()
                        # Optionally await cancellation with a timeout if critical
                        # try:
                        #     await asyncio.wait_for(task_to_cancel, timeout=1.0)
                        # except asyncio.CancelledError:
                        #     logging.debug("Worker task cancelled successfully by WC DQN.")
                        # except asyncio.TimeoutError:
                        #     logging.warning("Worker task did not respond to cancellation by WC DQN quickly.")
                else:
                    logging.warning("WC DQN: Min worker limit reached during decrease or no tasks to remove.")
                    break

        self.current_num_processing_workers = len(self.processing_worker_tasks) # Update actual count
        if self.current_num_processing_workers != previous_worker_count:
             logging.info(f"WC DQN: Adjusted processing workers from {previous_worker_count} to {self.current_num_processing_workers}.")
        else:
             logging.info(f"WC DQN: Maintained processing workers at {self.current_num_processing_workers}.")


    def _calculate_worker_count_reward_wc(self, time_delta_seconds: float) -> float:
        mnemonics_in_cycle = self.mnemonics_processed_in_session - self.mnemonics_processed_last_wc_cycle
        throughput = mnemonics_in_cycle / time_delta_seconds if time_delta_seconds > 0 else 0.0

        reward = throughput * 0.1 # Basic reward for throughput

        queue_fill = self.async_mnemonic_queue.qsize() / self.async_mnemonic_queue.maxsize if self.async_mnemonic_queue.maxsize > 0 else 0

        if queue_fill > 0.85 and self.wc_previous_action == 0 : # Penalize if decreased workers when queue was full
            reward -= 2.0
        elif queue_fill < 0.15 and self.wc_previous_action == 2: # Penalize if increased workers when queue was empty
            reward -= 1.0
        elif 0.2 <= queue_fill <= 0.8: # Bonus for keeping queue in healthy range
            reward += 0.5

        # Penalty for too many or too few workers relative to some ideal (hard to define without CPU load)
        # For now, rely on queue fill and throughput.

        logging.debug(f"WC Reward Calc: Throughput={throughput:.2f} (reward component: {throughput * 0.1:.2f}), "
                     f"QueueFill={queue_fill:.2f}. Total Reward: {reward:.2f}")
        return float(reward)

    async def _dqn_control_loop_worker_count(self):
        logging.info("DQN Worker Count control loop started.")
        await asyncio.sleep(DQN_WC_CYCLE_INTERVAL_SECONDS) # Initial delay

        while not self._stop_event.is_set():
            loop_start_time = time.time()
            current_state = self._get_worker_count_state_wc()

            if self.wc_previous_state is not None and self.wc_previous_action is not None:
                time_delta = loop_start_time - self.last_wc_cycle_time
                reward = self._calculate_worker_count_reward_wc(time_delta)

                await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.wc_agent_worker_count.step,
                    self.wc_previous_state, self.wc_previous_action, reward, current_state, False)
                logging.debug(f"WC DQN Agent learning step. Reward: {reward:.3f}, Epsilon: {self.wc_agent_worker_count.eps:.3f}")

            action = await asyncio.get_running_loop().run_in_executor(
                self.executor, self.wc_agent_worker_count.get_action, current_state)

            await self._apply_worker_count_action_wc(action) # Apply action (might change self.processing_worker_tasks)

            self.wc_previous_state = current_state
            self.wc_previous_action = action
            self.mnemonics_processed_last_wc_cycle = self.mnemonics_processed_in_session
            self.last_wc_cycle_time = loop_start_time

            await asyncio.sleep(DQN_WC_CYCLE_INTERVAL_SECONDS)

        logging.info("DQN Worker Count control loop stopped.")
        # Model saving handled in main shutdown


    async def process_queue(self):
        task_name = asyncio.current_task().get_name() if hasattr(asyncio.current_task(), 'get_name') else "ProcessQueueTask"
        logging.info(f"Worker {task_name} starting.")
        while not self._stop_event.is_set():
            try:
                mnemonic = await asyncio.wait_for(self.async_mnemonic_queue.get(), timeout=1.0)
                if mnemonic is None: # Sentinel for shutdown
                    self.async_mnemonic_queue.put_nowait(None) # Put back for other workers
                    break

                current_index = self.processed_mnemonic_index + 1 # Tentative index

                addresses = await self.generate_addresses(mnemonic)
                balances = {}
                found_any_balance = False
                for coin_type in self.coins:
                    address = addresses.get(coin_type)
                    if address:
                        balance = await self.check_balance(address, coin_type)
                        balances[str(coin_type)] = balance
                        if balance > 0:
                            found_any_balance = True
                    else: # Should not happen if generate_addresses is correct
                        balances[str(coin_type)] = 0.0

                await self.log_result(current_index, mnemonic, balances, found_any_balance)

                # Update processed index and counter
                self.processed_mnemonic_index = current_index
                self.mnemonics_processed_in_session += 1
                if self.mnemonics_processed_in_session % 100 == 0: # Log progress
                    logging.info(f"Processed {self.mnemonics_processed_in_session} mnemonics this session. Current overall index: {self.processed_mnemonic_index}")

            except asyncio.TimeoutError: # Expected if queue is empty
                continue
            except asyncio.CancelledError:
                logging.info("process_queue task cancelled.")
                break
            except Exception as e:
                # Log mnemonic if available, otherwise a placeholder
                mnemonic_str = mnemonic if 'mnemonic' in locals() and isinstance(mnemonic, str) else "unknown_mnemonic"
                logging.error(f"Error processing {mnemonic_str}: {str(e)}", exc_info=True)
            finally:
                if 'mnemonic' in locals() and mnemonic is not None: # Check if mnemonic was successfully retrieved
                    self.async_mnemonic_queue.task_done()
        logging.info(f"process_queue worker {asyncio.current_task().get_name()} stopping.")


    async def generate_addresses(self, mnemonic: str):
        """Generates addresses for supported coins from a mnemonic."""
        loop = asyncio.get_running_loop()
        try:
            # bip_utils recommends using its own Bip39SeedGenerator for converting mnemonic to seed
            # It also needs the language for the mnemonic. Assuming English.
            seed_bytes = await loop.run_in_executor(
                self.executor,
                Bip39SeedGenerator(mnemonic, Bip39Languages.ENGLISH).Generate
            )

            addresses = {}
            # Bitcoin (BTC) - BIP44 path m/44'/0'/0'/0/0
            bip44_btc_acc = Bip44.FromSeed(seed_bytes, Bip44Coins.BITCOIN).Purpose().Coin().Account(0)
            bip44_btc_addr = bip44_btc_acc.Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
            addresses[Bip44Coins.BITCOIN] = await loop.run_in_executor(self.executor, bip44_btc_addr.PublicKey().ToAddress)

            # Ethereum (ETH) - BIP44 path m/44'/60'/0'/0/0
            bip44_eth_acc = Bip44.FromSeed(seed_bytes, Bip44Coins.ETHEREUM).Purpose().Coin().Account(0)
            bip44_eth_addr = bip44_eth_acc.Change(Bip44Changes.CHAIN_EXT).AddressIndex(0)
            addresses[Bip44Coins.ETHEREUM] = await loop.run_in_executor(self.executor, bip44_eth_addr.PublicKey().ToAddress)

            return addresses
        except Exception as e:
            logging.error(f"Address generation failed for mnemonic '{mnemonic[:15]}...': {str(e)}", exc_info=True)
            return {}


    async def check_balance(self, address: str, coin_type):
        """
        Checks balance for a given address and coin type using multiple API fallbacks.
        Updates API endpoint statistics.
        """
        endpoint_url_templates = API_ENDPOINTS.get(coin_type, [])
        if not endpoint_url_templates:
            logging.warning(f"No API endpoints configured for coin type: {coin_type}")
            return 0.0

        # Create tasks for all configured endpoints for this coin_type
        # Store url_template with the task to identify it later for stats update
        tasks_with_ids = []
        for url_template in endpoint_url_templates:
            task = asyncio.create_task(
                self.fetch_balance(url_template, coin_type, address),
                name=f"fetch_{coin_type}_{url_template}" # Optional name for debugging
            )
            tasks_with_ids.append({"task": task, "url_template": url_template})

        final_balance = 0.0
        first_success_achieved = False

        # Process tasks as they complete
        for task_with_id in asyncio.as_completed([item["task"] for item in tasks_with_ids]):
            # Find the original url_template associated with this completed task
            # This is a bit clunky; could also pass url_template into fetch_balance's result tuple
            completed_task_obj = None
            url_template_for_task = None
            for item in tasks_with_ids:
                if item["task"] == task_with_id: # task_with_id is the coroutine result from as_completed
                    completed_task_obj = item["task"]
                    url_template_for_task = item["url_template"]
                    break

            if not completed_task_obj or not url_template_for_task:
                logging.error("Could not map completed task to its URL template. Skipping stats update for this task.")
                continue

            try:
                balance, success, is_timeout, is_429, latency_ms = await completed_task_obj

                # Update stats for this specific endpoint
                await self._update_specific_endpoint_stats(coin_type, url_template_for_task, success, is_timeout, is_429, latency_ms)

                if success and balance > 0 and not first_success_achieved:
                    final_balance = balance
                    first_success_achieved = True
                    # Don't break here; allow other tasks to complete for stats collection, but we have our primary result.
                    # However, we should cancel remaining tasks if we are satisfied.
                    # For now, let's let them all run to gather full stats, then refine.
                    # This means we might make more API calls than strictly necessary if first one succeeds.
            except Exception as e:
                # This exception would be from awaiting the task itself, not from within fetch_balance's try/except
                logging.error(f"Error processing task for {url_template_for_task}: {e}", exc_info=True)
                # Still attempt to update stats as a failure
                await self._update_specific_endpoint_stats(coin_type, url_template_for_task, False, False, False, 0)
        
        # Log API stats periodically (could also be a separate recurring task)
        await self._log_api_endpoint_stats_periodically()

        return final_balance


    async def fetch_balance(self, url_template: str, coin_type, address: str):
        """
        Fetches balance from a single API endpoint.
        Returns a tuple: (balance: float, success: bool, is_timeout: bool, is_429: bool, latency_ms: int)
        url_template is used as the key for stats.
        """
        is_timeout_local = False
        is_429_local = False
        status_code_local = None
        latency_ms = 0
        url_formatted = url_template.format(address=address)

        start_time = time.perf_counter()

        try:
            async with self.limiter: # Overall rate limiter
                async with self.session.get(url_formatted, timeout=10) as resp:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    status_code_local = resp.status
                    resp_text = await resp.text()

                    if status_code_local == 429:
                        is_429_local = True
                        logging.debug(f"API 429 error for {address} at {url_formatted}")
                        return 0.0, False, is_timeout_local, is_429_local, latency_ms

                    if status_code_local != 200:
                        logging.debug(f"API {url_formatted} returned status {status_code_local}. Response: {resp_text[:200]}")
                        return 0.0, False, is_timeout_local, is_429_local, latency_ms
                    
                    try:
                        data = json.loads(resp_text)
                    except json.JSONDecodeError:
                        logging.debug(f"Invalid JSON from {url_formatted}. Response: {resp_text[:200]}")
                        return 0.0, False, is_timeout_local, is_429_local, latency_ms
                    
                    if 'error' in data or ('message' in data and isinstance(data.get('message'), str) and
                                           "error" in data.get('message','').lower() and
                                           not "rate limit" in data.get('message','').lower()):
                        logging.debug(f"API error in payload for {address} at {url_formatted}: {data.get('error', data.get('message'))}")
                        return 0.0, False, is_timeout_local, is_429_local, latency_ms
                    
                    balance = self.parse_balance(data, coin_type, url_formatted)
                    return balance, True, is_timeout_local, is_429_local, latency_ms
                    
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000) # Capture latency up to timeout
            logging.debug(f"API timeout for {address} at {url_formatted}")
            is_timeout_local = True
            return 0.0, False, is_timeout_local, is_429_local, latency_ms
        except aiohttp.ClientError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logging.debug(f"API client error for {address} at {url_formatted}: {type(e).__name__} - {str(e)}")
            if hasattr(e, 'status') and e.status:
                status_code_local = e.status
                if status_code_local == 429: is_429_local = True
            return 0.0, False, is_timeout_local, is_429_local, latency_ms
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logging.warning(f"Unexpected API error for {address} at {url_formatted}: {type(e).__name__} - {str(e)}", exc_info=True)
            return 0.0, False, is_timeout_local, is_429_local, latency_ms
        finally:
            # Update general RL stats (for rate limiter DQN)
            await self._update_api_call_stats(status_code=status_code_local, is_timeout=is_timeout_local)
            # Specific endpoint stats are updated in check_balance after all attempts for a coin.


    def parse_balance(self, data, coin_type, url_for_context=""):
        """Parses balance from API response data."""
        try:
            if coin_type == Bip44Coins.BITCOIN:
                if 'final_balance' in data: # blockchain.info
                    return float(data['final_balance']) / 1e8
                # btc.com specific parsing (example, needs actual structure)
                if isinstance(data.get('data'), dict) and 'balance' in data['data']:
                    return float(data['data']['balance']) / 1e8
                # Add more parsers for other Bitcoin APIs if needed
                logging.debug(f"Unknown Bitcoin balance format from {url_for_context}: {str(data)[:200]}")
                return 0.0
                
            elif coin_type == Bip44Coins.ETHEREUM:
                result = data.get('result', '0')
                if isinstance(result, str) and result.startswith('0x'): # Hex value
                    return float(int(result, 16)) / 1e18
                if isinstance(result, str) and result.isdigit(): # String decimal
                    return float(result) / 1e18
                if isinstance(result, (int, float)): # Already a number
                     return float(result) / 1e18
                logging.debug(f"Unknown Ethereum balance format or error in result from {url_for_context}: {result}")
                return 0.0
                
            elif coin_type == "USDT": # ERC20 USDT on Ethereum
                result = data.get('result', '0')
                if isinstance(result, str) and result.isdigit():
                    return float(result) / 1e6 # USDT has 6 decimal places typically
                if isinstance(result, (int, float)):
                     return float(result) / 1e6
                logging.debug(f"Unknown USDT balance format or error in result from {url_for_context}: {result}")
                return 0.0
                
        except (ValueError, TypeError, KeyError) as e:
            logging.error(f"Parse error for {coin_type} from {url_for_context} with data {str(data)[:200]}: {e}", exc_info=True)
        return 0.0

    async def log_result(self, index: int, mnemonic: str, balances: dict, found_balance: bool):
        """Logs the result using the dedicated results_logger."""
        # Format: index|datetime|mnemonic|json_balances
        log_message = f"{index}|{datetime.now().isoformat()}|{mnemonic}|{json.dumps(balances)}"
        
        loop = asyncio.get_running_loop()
        # Run synchronous logging in executor to avoid blocking asyncio loop
        await loop.run_in_executor(self.executor, results_logger.info, log_message)

        if found_balance:
            found_log_message = f"{index}|{datetime.now().isoformat()}|{mnemonic}|{json.dumps(balances)}\n"
            try:
                # found.txt can also be managed by a logger if complex rotation is needed
                # For now, simple append. Ensure directory exists.
                os.makedirs(os.path.dirname(self.found_file), exist_ok=True)
                # Synchronous write to found.txt, executed in the ThreadPoolExecutor
                await loop.run_in_executor(self.executor, self._write_to_found_file, found_log_message)
            except Exception as e:
                logging.error(f"Failed to write to found.txt: {e}")

    def _write_to_found_file(self, content: str):
        """Helper synchronous method to write content to the found_file."""
        try:
            with open(self.found_file, "a") as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Synchronous write to {self.found_file} failed: {e}")

async def main():
    # Ensure necessary directories exist
    os.makedirs("finder", exist_ok=True)

    checker = BalanceChecker()
    try:
        await checker.start()
    except KeyboardInterrupt:
        logging.info("Main: KeyboardInterrupt caught, initiating shutdown sequence.")
        # The checker.start() finally block should handle graceful shutdown.
    except Exception as e:
        logging.critical(f"Main: Unhandled exception in BalanceChecker: {e}", exc_info=True)
    finally:
        logging.info("Main: Application shutdown complete.")


if __name__ == "__main__":
    # Important for multiprocessing on Windows and macOS with spawn/forkserver
    multiprocessing.freeze_support()
    asyncio.run(main())