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

# Modular imports
import config as app_config
import logger_setup # For main_entry_point call
from mnemonic_generator import (
    MnemonicGeneratorManager,
    generate_addresses_with_paths, # Use this for address derivation
    load_generator_index_from_file, # For loading processed index state
    save_generator_index_to_file    # For saving processed index state
)
from ppo_sb3_agent import PPOAgentSB3
from api_handler import APIHandler
from features import extract_mnemonic_features # extract_address_features is not used yet
from task import Task # Import the Task class


# Configure a logger for this module
logger = logging.getLogger(__name__)


class BalanceChecker:
    def __init__(self, cfg): # Pass config object
        self.config = cfg
        self.api_handler: APIHandler = None # Will be initialized in start()
        self.executor = ThreadPoolExecutor(max_workers=self.config.APP_MAX_EXECUTOR_WORKERS)

        # --- Rate Limiter Agent (PPO) ---
        if self.config.ENABLE_RL_AGENT_RATE_LIMITER:
            self.rl_agent_rate_limiter = PPOAgentSB3(
                state_dim=self.config.RL_AGENT_STATE_SIZE,
                action_dim=self.config.RL_AGENT_ACTION_SIZE,
                lr=self.config.PPO_LR_RATE_LIMITER,
                gamma=self.config.PPO_GAMMA_RATE_LIMITER,
                agent_name=self.config.RL_AGENT_NAME,
                models_dir=self.config.MODELS_DIR,
                log_dir=self.config.LOG_DIR_PPO_RATE_LIMITER,
                seed=self.config.SEED_RL_AGENT_RATE_LIMITER
            )
            logger.info("PPO Rate Limiter Agent enabled and initialized.")
        else:
            self.rl_agent_rate_limiter = None
            logger.info("PPO Rate Limiter Agent disabled.")
        self.current_api_rate = self.config.INITIAL_OVERALL_API_RATE # Initial rate
        self.limiter = AsyncLimiter(self.current_api_rate, 1) # Will be updated by PPO

        # --- Worker Count Agent (PPO) ---
        if self.config.ENABLE_RL_AGENT_WORKER_COUNT:
            self.wc_agent_worker_count = PPOAgentSB3(
                state_dim=self.config.WC_AGENT_STATE_SIZE,
                action_dim=self.config.WC_AGENT_ACTION_SIZE,
                lr=self.config.PPO_LR_WORKER_COUNT,
                gamma=self.config.PPO_GAMMA_WORKER_COUNT,
                agent_name=self.config.WC_AGENT_NAME,
                models_dir=self.config.MODELS_DIR,
                log_dir=self.config.LOG_DIR_PPO_WORKER_COUNT,
                seed=self.config.SEED_RL_AGENT_WORKER_COUNT
            )
            logger.info("PPO Worker Count Agent enabled and initialized.")
        else:
            self.wc_agent_worker_count = None
            logger.info("PPO Worker Count Agent disabled.")
        self.current_num_processing_workers = self.config.INITIAL_PROCESSING_WORKERS
        self.processing_worker_tasks = []

        # --- Common Attributes ---
        self.async_mnemonic_queue = asyncio.Queue(maxsize=self.config.ASYNC_MNEMONIC_QUEUE_SIZE)
        self.coins_to_check = self.config.COINS_TO_CHECK # List of Bip44Coins enums

        self.processed_mnemonic_index = 0 # Initialized from state file in start()
        self.mnemonics_processed_in_session = 0

        # Multiprocessing queue for receiving mnemonics from workers
        self.mp_mnemonic_input_queue = multiprocessing.Queue(maxsize=self.config.MP_MNEMONIC_QUEUE_SIZE)
        self.mnemonic_generator_manager = MnemonicGeneratorManager( # Manager from mnemonic_generator.py
            config=self.config,
            num_workers=self.config.MNEMONIC_GENERATOR_WORKERS,
            # load_state=True here means MnemonicGeneratorManager loads its *raw* generation index.
            # BalanceChecker loads its *processed* index separately.
            load_state=True
        )
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event() # For pause/resume functionality
        self._pause_event.set() # Start in unpaused state (event is clear = paused)

        # --- Classifier Model ---
        self.classifier_history_model = None
        if self.config.ENABLE_CLASSIFIER_SCORING:
            try:
                import joblib # Lazy import for classifier dependency
                model_path = self.config.CLASSIFIER_MODEL_HISTORY_PATH
                if os.path.exists(model_path):
                    self.classifier_history_model = joblib.load(model_path)
                    logger.info(f"History classifier model loaded from {model_path}")
                else:
                    logger.warning(f"History classifier model not found at {model_path}. Scoring will be disabled.")
                    self.config.ENABLE_CLASSIFIER_SCORING = False
            except Exception as e:
                logger.error(f"Failed to load history classifier model: {e}", exc_info=True)
                self.config.ENABLE_CLASSIFIER_SCORING = False

        # --- Web Server Attributes ---
        self.websocket_message_queue = asyncio.Queue(maxsize=100) # For sending updates to UI
        self.app_stats = { # For UI display
            "mnemonics_checked_session": 0,
            "mnemonics_per_second_session": 0,
            "total_mnemonics_checked_all_time": 0,
            "wallets_found_session": 0,
            "current_api_rate_limit": self.current_api_rate,
            "active_processing_workers": self.current_num_processing_workers,
            "input_queue_size": 0,
            "output_queue_size": 0, # Not directly used, but async_mnemonic_queue.qsize()
            "proxy_enabled": self.config.ENABLE_PROXY,
            "proxy_url": self.config.PROXY_URL if self.config.ENABLE_PROXY else "N/A",
            "status": "Initializing", # Initializing, Running, Paused, Stopping, Stopped
            "last_found_wallet_details": None, # Store details of the last found wallet
            "errors_encountered": 0,
            "api_429_errors_total": 0, # From APIHandler
            "api_timeout_errors_total": 0, # From APIHandler
            "api_other_errors_total": 0, # From APIHandler
        }
        self.session_start_time = time.time()
        self.found_wallets_count_session = 0


    async def _mp_to_asyncio_queue_adapter(self):
        logger.info("Starting multiprocessing to asyncio queue adapter.")
        loop = asyncio.get_running_loop()
        temp_mnemonic_list = [] # Buffer for batch put
        while not self._stop_event.is_set():
            await self._pause_event.wait() # Respect pause signal
            try:
                # Batch get from mp.Queue if possible, or single get
                # For simplicity, using single get with timeout in executor
                mnemonic_data = await loop.run_in_executor(self.executor, self.mp_queue.get, True, 0.1) # Timeout 0.1s
                if mnemonic_data: # Expecting (index, mnemonic_phrase) from new generator
                    # The new MnemonicGeneratorManager from mnemonic_generator.py puts (index, phrase)
                    # but the one in this app.py was modified to put just phrase.
                    # Let's assume the external one puts (index, phrase) and we use the phrase.
                    # If it's just phrase, then current_index logic in process_queue needs care.
                    # For now, assuming mnemonic_data is just the phrase as per current MnemonicGeneratorManager in app.py.

                    # If MnemonicGeneratorManager from finder.mnemonic_generator.py is used,
                    # it puts (index, phrase). We should use that index.
                    # Let's switch to that assumption, as it's more robust.
                    # current_idx, mnemonic_phrase = mnemonic_data # If (idx, phrase)
                    # For now, this BalanceChecker assumes its own MnemonicGeneratorManager from app.py
                    # which puts only the phrase. So self.processed_mnemonic_index is managed here.

                    # Reconciling with `finder.mnemonic_generator.MnemonicGeneratorManager`:
                    # That manager is simpler and its workers put only `mnemonic_phrase`.
                    # So, this adapter is fine as is, `BalanceChecker` manages the overall index.
                    await self.async_mnemonic_queue.put(mnemonic_data) # mnemonic_data is just the phrase

            except multiprocessing.queues.Empty: # Expected on timeout
                continue
            except Exception as e:
                logger.error(f"Error in mp_to_asyncio_queue_adapter: {e}", exc_info=True)
                await asyncio.sleep(0.1) # Avoid busy loop on error
        logger.info("Multiprocessing to asyncio queue adapter stopped.")

    async def _handle_api_stats_for_rl(self, status_code, is_timeout, is_429, is_other_failure):
        """Callback for APIHandler to update RL-relevant stats."""
        if self.rl_agent_rate_limiter: # Only if agent is enabled
            self.rl_agent_rate_limiter.increment_api_calls() # Assuming PPOAgent has such a method
            if is_429: self.rl_agent_rate_limiter.increment_429_errors()
            if is_timeout: self.rl_agent_rate_limiter.increment_timeout_errors()
            # Update app_stats for UI
            if is_429: self.app_stats["api_429_errors_total"] +=1
            if is_timeout: self.app_stats["api_timeout_errors_total"] +=1
            if is_other_failure and not is_429 and not is_timeout:
                self.app_stats["api_other_errors_total"] +=1


    async def start(self, args): # Pass command line args
        logger.info("Initializing BalanceChecker...")
        self.app_stats["status"] = "Initializing"
        self.update_websocket_message_queue() # Initial status update

        # Load initial *processed* mnemonic index from app's state file
        self.processed_mnemonic_index = load_generator_index_from_file(self.config.GENERATOR_STATE_FILE_APP)
        self.app_stats["total_mnemonics_checked_all_time"] = self.processed_mnemonic_index
        logger.info(f"Loaded initial processed mnemonic index: {self.processed_mnemonic_index} from {self.config.GENERATOR_STATE_FILE_APP}")

        # Initialize APIHandler with aiohttp.ClientSession
        # The session should be created here and passed to APIHandler
        # Proxy configuration should be handled by APIHandler based on config
        connector_settings = {'limit_per_host': self.config.TCP_CONNECTOR_LIMIT_PER_HOST}
        if self.config.ENABLE_PROXY and self.config.PROXY_URL:
            logger.info(f"Proxy enabled, using: {self.config.PROXY_URL}")
            # Proxy will be passed to session.get by APIHandler
        else:
            logger.info("Proxy disabled.")

        # Initialize aiohttp.ClientSession here to be managed by BalanceChecker
        self.client_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(**connector_settings),
            headers={"User-Agent": self.config.DEFAULT_USER_AGENT}
        )

        self.api_handler = APIHandler(
            config_obj=self.config, # Pass full config object
            session=self.client_session, # Pass the session
            limiter=self.limiter, # Pass the rate limiter to be used by APIHandler's _make_api_request
                                  # APIHandler will use this limiter internally.
            stats_callback=self._handle_api_stats_for_rl # Callback for RL agent
        )
        # The APIHandler itself will manage its internal session if not provided one,
        # but better to provide one for centralized lifecycle management.
        # The APIHandler's init_session will be called implicitly or explicitly.
        # Let's assume APIHandler uses the provided session.
        # No, APIHandler creates its own session if one isn't passed.
        # If we want APIHandler to use *this* session, it needs to accept it.
        # And APIHandler's overall_limiter is distinct from self.limiter here.
        # This needs careful thought.
        # For PPO, the self.limiter rate is what the PPO agent controls.
        # APIHandler should use *that* limiter.

        # Revised APIHandler integration:
        # BalanceChecker creates and manages the aiohttp.ClientSession.
        # BalanceChecker creates and manages the AsyncLimiter (self.limiter) whose rate is PPO controlled.
        # APIHandler is instantiated with this session and this limiter.
        # This seems cleaner. APIHandler's internal limiter becomes unused if one is passed.

        # Start the MnemonicGeneratorManager (from finder.mnemonic_generator.py)
        # It manages its own raw generation index. We pass it the queue it should use.
        self.mnemonic_generator_manager.start_generation(output_queue=self.mp_mnemonic_input_queue)

        adapter_task = asyncio.create_task(self._mp_to_asyncio_queue_adapter(self.mp_mnemonic_input_queue))

        # Initialize processing workers (coroutines)
        self.processing_worker_tasks = []
        for _ in range(self.current_num_processing_workers): # Use initial worker count from config
            task = asyncio.create_task(self.process_queue())
            self.processing_worker_tasks.append(task)
        logger.info(f"Starting with {self.current_num_processing_workers} processing workers.")
        self.app_stats["active_processing_workers"] = self.current_num_processing_workers

        # Start PPO control loops if enabled
        control_loop_tasks = []
        if self.rl_agent_rate_limiter:
            control_loop_tasks.append(asyncio.create_task(self._ppo_control_loop_rate_limiter()))
        if self.wc_agent_worker_count:
            control_loop_tasks.append(asyncio.create_task(self._ppo_control_loop_worker_count()))

        # Start periodic stats update task
        stats_update_task = asyncio.create_task(self._periodic_stats_updater())

        # Start web server task
        web_server_task = asyncio.create_task(self.run_web_server())

        core_tasks = [adapter_task, stats_update_task, web_server_task] + control_loop_tasks
        self.app_stats["status"] = "Running"
        self.update_websocket_message_queue()

        try:
            await asyncio.gather(*core_tasks) # Removed self.processing_worker_tasks from main gather
                                            # as they are managed by PPO agent / shutdown logic
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down BalanceChecker...")
        except Exception as e:
            logger.critical(f"Critical error in BalanceChecker main gather: {e}", exc_info=True)
            self.app_stats["errors_encountered"] +=1
        finally:
            logger.info("BalanceChecker shutting down gracefully...")
            self.app_stats["status"] = "Stopping"
            self.update_websocket_message_queue()
            self._stop_event.set()
            self._pause_event.set() # Ensure it's unpaused for shutdown tasks

            # Shutdown MnemonicGeneratorManager (from finder.mnemonic_generator.py)
            self.mnemonic_generator_manager.stop_generation()

            # Cancel all processing worker tasks explicitly (these are BalanceChecker's workers)
            logger.info(f"Cancelling {len(self.processing_worker_tasks)} processing workers...")
            for task in self.processing_worker_tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*self.processing_worker_tasks, return_exceptions=True)
            logger.info("Processing workers shut down.")

            # Wait for core tasks with a timeout
            # PPO loops should handle _stop_event and save models if needed within their loop or on exit
            # Let's ensure PPO agents save their models here explicitly if not in their loop.
            done, pending = await asyncio.wait(core_tasks, timeout=max(self.config.RL_CYCLE_INTERVAL_SECONDS, self.config.WC_CYCLE_INTERVAL_SECONDS) + 10.0)
            for task in pending:
                task_name = getattr(task, 'get_name', lambda: "Unknown Core Task")()
                logger.warning(f"Core task {task_name} did not finish in time, cancelling.")
                task.cancel()
            if pending: await asyncio.gather(*pending, return_exceptions=True)

            if self.api_handler and self.api_handler.session: # Close session used by APIHandler
                await self.api_handler.session.close() # APIHandler needs a close_session method or direct access
                logger.info("APIHandler's aiohttp session closed.")
            elif self.client_session: # If APIHandler used our session
                 await self.client_session.close()
                 logger.info("BalanceChecker's main aiohttp session closed.")

            # Save PPO agent models if they are enabled
            if self.rl_agent_rate_limiter: self.rl_agent_rate_limiter.save_model()
            if self.wc_agent_worker_count: self.wc_agent_worker_count.save_model()
            logger.info("PPO agent models saved (if enabled).")

            if self.executor:
                self.executor.shutdown(wait=True)
                logger.info("ThreadPoolExecutor shut down.")

            save_generator_index_to_file(self.processed_mnemonic_index, self.config.GENERATOR_STATE_FILE_APP)
            logger.info(f"Saved processed generator state. Last processed index: {self.processed_mnemonic_index} to {self.config.GENERATOR_STATE_FILE_APP}")
            logger.info(f"Mnemonics processed this session: {self.mnemonics_processed_in_session}")
            logger.info(f"Wallets with balance found this session: {self.found_wallets_count_session}")
            self.app_stats["status"] = "Stopped"
            self.update_websocket_message_queue() # Final status update

    # --- PPO Rate Limiter Methods ---
    def _get_rate_limiter_state_ppo(self) -> np.ndarray:
        # Normalize current rate
        norm_current_rate = (self.current_api_rate - self.config.RL_MIN_RATE_LIMIT) / \
                            (self.config.RL_MAX_RATE_LIMIT - self.config.RL_MIN_RATE_LIMIT)
        norm_current_rate = np.clip(norm_current_rate, 0, 1)

        # Get error rates from the agent itself (which should be updated by APIHandler callback)
        error_rate_429 = self.rl_agent_rate_limiter.get_429_rate() if self.rl_agent_rate_limiter else 0.0
        timeout_rate = self.rl_agent_rate_limiter.get_timeout_rate() if self.rl_agent_rate_limiter else 0.0

        queue_fill = self.async_mnemonic_queue.qsize() / self.config.ASYNC_MNEMONIC_QUEUE_SIZE \
            if self.config.ASYNC_MNEMONIC_QUEUE_SIZE > 0 else 0.0

        state = np.array([norm_current_rate, error_rate_429, timeout_rate, queue_fill], dtype=np.float32)
        return state

    def _apply_rate_limiter_action_ppo(self, action: int):
        previous_rate = self.current_api_rate
        # Action mapping: 0 = decrease, 1 = maintain, 2 = increase
        if action == 0:
            self.current_api_rate = max(self.config.RL_MIN_RATE_LIMIT, self.current_api_rate - self.config.RL_RATE_ADJUSTMENT_STEP)
        elif action == 2:
            self.current_api_rate = min(self.config.RL_MAX_RATE_LIMIT, self.current_api_rate + self.config.RL_RATE_ADJUSTMENT_STEP)
        # action == 1 means maintain, so no change.

        if abs(self.current_api_rate - previous_rate) > 1e-5 : # If rate actually changed
            self.limiter = AsyncLimiter(self.current_api_rate, 1) # Re-initialize limiter
            logger.info(f"PPO RL Agent adjusted API rate from {previous_rate:.2f} to {self.current_api_rate:.2f} rps.")
            self.app_stats["current_api_rate_limit"] = self.current_api_rate
        else:
            logger.info(f"PPO RL Agent decided to maintain API rate at {self.current_api_rate:.2f} rps.")

    def _calculate_rate_limiter_reward_ppo(self, time_delta_seconds: float, previous_action: int) -> float:
        # Throughput: mnemonics processed by BalanceChecker workers
        # This needs to be tracked correctly per cycle for PPO agent.
        # Assume PPOAgentSB3 handles its own mnemonic count or we pass it.
        # Let's use self.rl_agent_rate_limiter.get_processed_count_cycle()

        processed_count_cycle = self.rl_agent_rate_limiter.get_and_reset_processed_count_cycle() # Agent tracks this
        throughput = processed_count_cycle / time_delta_seconds if time_delta_seconds > 0 else 0.0

        error_429_count_cycle = self.rl_agent_rate_limiter.get_and_reset_429_errors_cycle()
        timeout_count_cycle = self.rl_agent_rate_limiter.get_and_reset_timeout_errors_cycle()

        penalty_429 = self.config.RL_PENALTY_429 * error_429_count_cycle
        penalty_timeout = self.config.RL_PENALTY_TIMEOUT * timeout_count_cycle

        throughput_reward = throughput * self.config.RL_REWARD_THROUGHPUT_SCALAR

        queue_penalty = 0.0
        queue_fill = self.async_mnemonic_queue.qsize() / self.config.ASYNC_MNEMONIC_QUEUE_SIZE if self.config.ASYNC_MNEMONIC_QUEUE_SIZE > 0 else 0.0
        if queue_fill < self.config.RL_QUEUE_LOW_THRESHOLD_PENALTY and previous_action == 2: # Increased rate with empty queue
             queue_penalty = self.config.RL_PENALTY_QUEUE_LOW_ON_INCREASE
        if queue_fill > self.config.RL_QUEUE_HIGH_THRESHOLD_PENALTY: # Penalize nearly full queue
             queue_penalty = self.config.RL_PENALTY_QUEUE_HIGH

        reward = throughput_reward + penalty_429 + penalty_timeout + queue_penalty
        logger.debug(f"PPO RL Reward: Throughput={throughput:.2f} ({throughput_reward:.2f}), "
                     f"429s={error_429_count_cycle} ({penalty_429:.2f}), "
                     f"Timeouts={timeout_count_cycle} ({penalty_timeout:.2f}), "
                     f"QueuePen={queue_penalty:.2f}. Total: {reward:.2f}")
        return float(reward)

    async def _ppo_control_loop_rate_limiter(self):
        logger.info("PPO Rate Limiter control loop started.")
        await asyncio.sleep(self.config.RL_CYCLE_INTERVAL_SECONDS)

        last_cycle_time = time.time()

        # PPO often collects a rollout (sequence of experiences) before learning.
        # For simplicity here, we might do a learn step every cycle, which is more like A2C/A3C.
        # Or, PPOAgentSB3 handles its own n_steps internally for rollout collection.

        while not self._stop_event.is_set():
            await self._pause_event.wait() # Respect pause
            loop_start_time = time.time()
            try:
                current_state = self._get_rate_limiter_state_ppo()
                action, _states = self.rl_agent_rate_limiter.predict(current_state, deterministic=not self.config.PPO_TRAIN_MODE)

                self._apply_rate_limiter_action_ppo(action)

                # Wait for the cycle duration to collect experience
                # This sleep should ideally be adjusted if the above steps take significant time
                # For now, fixed sleep assuming agent steps are fast.
                await asyncio.sleep(self.config.RL_CYCLE_INTERVAL_SECONDS) # Target cycle interval

                if self.config.PPO_TRAIN_MODE:
                    time_delta = time.time() - loop_start_time # Actual time passed in cycle
                    reward = self._calculate_rate_limiter_reward_ppo(time_delta, action)
                    next_state = self._get_rate_limiter_state_ppo()
                    done = self._stop_event.is_set()

                    self.rl_agent_rate_limiter.record_experience(current_state, action, reward, next_state, done)
                    await asyncio.get_running_loop().run_in_executor(self.executor, self.rl_agent_rate_limiter.train_on_collected_rollout)
            except Exception as e:
                logger.error(f"Error in PPO Rate Limiter control loop: {e}", exc_info=True)
                # Optional: Implement a cooldown or skip a cycle on error to prevent rapid error loops
                await asyncio.sleep(self.config.RL_CYCLE_INTERVAL_SECONDS) # Ensure we still wait out the cycle

            last_cycle_time = time.time()

        logger.info("PPO Rate Limiter control loop stopped.")

    # --- PPO Worker Count Methods ---
    def _get_worker_count_state_ppo(self) -> np.ndarray:
        norm_worker_count = (len(self.processing_worker_tasks) - self.config.WC_MIN_PROCESSING_WORKERS) / \
                            (self.config.WC_MAX_PROCESSING_WORKERS - self.config.WC_MIN_PROCESSING_WORKERS)
        norm_worker_count = np.clip(norm_worker_count, 0, 1)

        queue_fill = self.async_mnemonic_queue.qsize() / self.config.ASYNC_MNEMONIC_QUEUE_SIZE \
            if self.config.ASYNC_MNEMONIC_QUEUE_SIZE > 0 else 0.0

        # More advanced state: CPU load, actual processing time per item
        # For now, using a placeholder like in DQN version or just two states.
        # Let's use CPU load if psutil is available and configured
        cpu_load = 0.0
        if self.config.WC_INCLUDE_CPU_LOAD_STATE:
            try:
                import psutil # Lazy import for optional dependency
                cpu_load = psutil.cpu_percent(interval=None) / 100.0 # Normalized
            except ImportError:
                logger.warning("psutil not installed, CPU load for WC agent state is disabled.")
            except Exception as e:
                logger.error(f"Error getting CPU load: {e}")

        state_features = [norm_worker_count, queue_fill]
        if self.config.WC_INCLUDE_CPU_LOAD_STATE:
            state_features.append(cpu_load)

        # Ensure state matches WC_AGENT_STATE_SIZE from config
        # This is a simple way, could be more dynamic if state features change often
        if len(state_features) != self.config.WC_AGENT_STATE_SIZE:
            logger.error(f"WC Agent state feature count mismatch: expected {self.config.WC_AGENT_STATE_SIZE}, got {len(state_features)}. Adjust config or state features.")
            # Fallback to a fixed size state if error (e.g. pad with zeros or truncate)
            # For now, this will likely cause PPO to fail if sizes don't match.
            # Best to ensure config matches features defined here.

        return np.array(state_features, dtype=np.float32)


    async def _apply_worker_count_action_ppo(self, action: int):
        previous_worker_count = len(self.processing_worker_tasks)
        target_workers = previous_worker_count

        if action == 0: # Decrease workers
            target_workers = max(self.config.WC_MIN_PROCESSING_WORKERS, previous_worker_count - self.config.WC_WORKER_ADJUSTMENT_STEP)
        elif action == 2: # Increase workers
            target_workers = min(self.config.WC_MAX_PROCESSING_WORKERS, previous_worker_count + self.config.WC_WORKER_ADJUSTMENT_STEP)
        # action == 1 means maintain

        tasks_to_add = target_workers - previous_worker_count

        if tasks_to_add > 0:
            logger.info(f"PPO WC: Increasing workers by {tasks_to_add}. Current: {previous_worker_count}, Target: {target_workers}")
            for _ in range(tasks_to_add):
                if len(self.processing_worker_tasks) < self.config.WC_MAX_PROCESSING_WORKERS:
                    task = asyncio.create_task(self.process_queue())
                    self.processing_worker_tasks.append(task)
                else:
                    logger.warning("PPO WC: Max worker limit reached during increase.")
                    break
        elif tasks_to_add < 0:
            num_to_remove = abs(tasks_to_add)
            logger.info(f"PPO WC: Decreasing workers by {num_to_remove}. Current: {previous_worker_count}, Target: {target_workers}")
            for _ in range(num_to_remove):
                if self.processing_worker_tasks:
                    task_to_cancel = self.processing_worker_tasks.pop()
                    if not task_to_cancel.done(): task_to_cancel.cancel()
                else:
                    logger.warning("PPO WC: Min worker limit reached or no tasks to remove.")
                    break

        self.current_num_processing_workers = len(self.processing_worker_tasks) # Update actual count
        self.app_stats["active_processing_workers"] = self.current_num_processing_workers
        if self.current_num_processing_workers != previous_worker_count:
             logger.info(f"PPO WC: Adjusted processing workers from {previous_worker_count} to {self.current_num_processing_workers}.")
        else:
             logger.info(f"PPO WC: Maintained processing workers at {self.current_num_processing_workers}.")

    def _calculate_worker_count_reward_ppo(self, time_delta_seconds: float, previous_action: int) -> float:
        processed_count_cycle = self.wc_agent_worker_count.get_and_reset_processed_count_cycle()
        throughput = processed_count_cycle / time_delta_seconds if time_delta_seconds > 0 else 0.0

        reward = throughput * self.config.WC_REWARD_THROUGHPUT_SCALAR

        queue_fill = self.async_mnemonic_queue.qsize() / self.config.ASYNC_MNEMONIC_QUEUE_SIZE if self.config.ASYNC_MNEMONIC_QUEUE_SIZE > 0 else 0.0

        if queue_fill > self.config.WC_QUEUE_HIGH_THRESHOLD_PENALTY and previous_action == 0 : # Decreased workers when queue was full
            reward += self.config.WC_PENALTY_QUEUE_HIGH_ON_DECREASE # Penalties are negative in config
        elif queue_fill < self.config.WC_QUEUE_LOW_THRESHOLD_PENALTY and previous_action == 2: # Increased workers when queue was empty
            reward += self.config.WC_PENALTY_QUEUE_LOW_ON_INCREASE

        if self.config.WC_REWARD_QUEUE_OPTIMAL_RANGE_BONUS > 0 and \
           self.config.WC_QUEUE_OPTIMAL_RANGE_LOW <= queue_fill <= self.config.WC_QUEUE_OPTIMAL_RANGE_HIGH:
            reward += self.config.WC_REWARD_QUEUE_OPTIMAL_RANGE_BONUS

        # Optional: Penalty for high CPU load if workers were increased
        if self.config.WC_PENALTY_HIGH_CPU_ON_INCREASE < 0 and previous_action == 2:
            try:
                import psutil
                cpu_load = psutil.cpu_percent(interval=None)
                if cpu_load > self.config.WC_CPU_HIGH_THRESHOLD_PENALTY:
                    reward += self.config.WC_PENALTY_HIGH_CPU_ON_INCREASE
            except Exception: pass # Ignore if psutil fails or not installed

        logger.debug(f"PPO WC Reward: Throughput={throughput:.2f} (rew: {throughput * self.config.WC_REWARD_THROUGHPUT_SCALAR:.2f}), "
                     f"QueueFill={queue_fill:.2f}. Total: {reward:.2f}")
        return float(reward)

    async def _ppo_control_loop_worker_count(self):
        logger.info("PPO Worker Count control loop started.")
        await asyncio.sleep(self.config.WC_CYCLE_INTERVAL_SECONDS)

        last_cycle_time = time.time()

        while not self._stop_event.is_set():
            await self._pause_event.wait() # Respect pause
            loop_start_time = time.time()
            try:
                current_state = self._get_worker_count_state_ppo()
                action, _ = self.wc_agent_worker_count.predict(current_state, deterministic=not self.config.PPO_TRAIN_MODE)

                await self._apply_worker_count_action_ppo(action)

                # Wait for the cycle duration
                await asyncio.sleep(self.config.WC_CYCLE_INTERVAL_SECONDS)

                if self.config.PPO_TRAIN_MODE:
                    time_delta = time.time() - loop_start_time # Actual time passed
                    reward = self._calculate_worker_count_reward_ppo(time_delta, action)
                    next_state = self._get_worker_count_state_ppo()
                    done = self._stop_event.is_set()

                    self.wc_agent_worker_count.record_experience(current_state, action, reward, next_state, done)
                    await asyncio.get_running_loop().run_in_executor(self.executor, self.wc_agent_worker_count.train_on_collected_rollout)
            except Exception as e:
                logger.error(f"Error in PPO Worker Count control loop: {e}", exc_info=True)
                await asyncio.sleep(self.config.WC_CYCLE_INTERVAL_SECONDS) # Ensure wait on error

            last_cycle_time = time.time()

        logger.info("PPO Worker Count control loop stopped.")


    async def process_queue(self):
        """Worker coroutine that processes mnemonics from the async_mnemonic_queue."""
        task_name = getattr(asyncio.current_task(), 'get_name', lambda: "ProcessQueueTask")()
        logger.info(f"Worker {task_name} starting.")

        # Get local references to config values used frequently in loop
        num_child_addrs = self.config.NUM_CHILD_ADDRESSES_TO_CHECK
        bip44_account = self.config.BIP44_ACCOUNT
        bip44_change_val = self.config.BIP44_CHANGE_TYPE.value # Get int value from enum
        coins_to_process = self.coins_to_check # From config

        while not self._stop_event.is_set():
            await self._pause_event.wait() # Respect pause
            current_task_object: Optional[Task] = None
            mnemonic_phrase: Optional[str] = None
            try:
                mnemonic_phrase = await asyncio.wait_for(self.async_mnemonic_queue.get(), timeout=1.0)
                if mnemonic_phrase is None: # Sentinel for shutdown
                    self.async_mnemonic_queue.task_done()
                    break

                # Create a Task object for the current mnemonic
                # The index for the task will be the next sequential processed index.
                current_processed_idx = self.processed_mnemonic_index + 1
                current_task_object = Task(mnemonic_phrase=mnemonic_phrase, index=current_processed_idx)

                # 1. Generate addresses for all configured coins
                derived_addresses_by_coin_enum = await asyncio.get_running_loop().run_in_executor(
                    self.executor,
                    generate_addresses_with_paths,
                    current_task_object.mnemonic_phrase, coins_to_process, num_child_addrs, bip44_account, bip44_change_val
                )
                current_task_object.set_derived_addresses(derived_addresses_by_coin_enum)

                found_any_balance_for_mnemonic = False
                history_score = 0.0 # Default if classifier not used or fails

                # Optional: Score mnemonic with classifier before detailed checks
                if self.config.ENABLE_CLASSIFIER_SCORING and self.classifier_history_model:
                    try:
                        mnemonic_features = await asyncio.get_running_loop().run_in_executor(
                            self.executor, extract_mnemonic_features, current_task_object.mnemonic_phrase
                        )
                        score_proba = await asyncio.get_running_loop().run_in_executor(
                             self.executor, self.classifier_history_model.predict_proba, [mnemonic_features]
                        )
                        history_score = score_proba[0][1]
                        if history_score < self.config.CLASSIFIER_SKIP_THRESHOLD:
                            logger.debug(f"Skipping Task {current_task_object.index} (mnemonic {current_task_object.mnemonic_phrase[:15]}...) due to low history score: {history_score:.3f}")
                            # If skipping, we might want to log it differently or mark task as skipped.
                            # For now, it will proceed but the score is available.
                    except Exception as e:
                        logger.error(f"Error during mnemonic classification for Task {current_task_object.index}: {e}", exc_info=True)
                        current_task_object.error_message = f"Classifier error: {e}"


                # 2. For each coin and its derived addresses, check balance and existence
                # Iterate through the addresses stored in the Task object
                for coin_name_str, child_addresses_info_list in current_task_object.derived_addresses.items():
                    # Need to map coin_name_str back to Bip44Coins enum for APIHandler
                    # This assumes coin_name_str is a valid Bip44Coins member name.
                    try:
                        coin_type_enum = Bip44Coins[coin_name_str]
                    except KeyError:
                        logger.error(f"Invalid coin name '{coin_name_str}' in Task {current_task_object.index}. Skipping checks for this coin.")
                        continue

                    for child_addr_detail in child_addresses_info_list:
                        address_str = child_addr_detail["address"]
                        # path_str = child_addr_detail["path"] # Path already in task object

                        balance, has_funds, has_history = await self.api_handler.get_balance_and_existence(
                            coin_type_enum, address_str
                        )

                        # Update the Task object with the results
                        current_task_object.update_address_details(coin_name_str, address_str, balance, has_funds, has_history)

                        if has_funds:
                            found_any_balance_for_mnemonic = True

                current_task_object.mark_as_completed()

                # 3. Log the comprehensive result for the task
                await self.log_result(current_task_object, found_any_balance_for_mnemonic, history_score if self.config.ENABLE_CLASSIFIER_SCORING else None)

                # 4. Update counters
                self.processed_mnemonic_index = current_task_object.index # Use index from task
                self.mnemonics_processed_in_session += 1
                self.app_stats["mnemonics_checked_session"] = self.mnemonics_processed_in_session

                # Update per-cycle counts for RL agents
                if self.rl_agent_rate_limiter: self.rl_agent_rate_limiter.increment_processed_count()
                if self.wc_agent_worker_count: self.wc_agent_worker_count.increment_processed_count()

                if found_any_balance_for_mnemonic:
                    self.found_wallets_count_session += 1
                    self.app_stats["wallets_found_session"] = self.found_wallets_count_session
                    self.app_stats["last_found_wallet_details"] = current_task_object.to_dict() # Log the full task object dict

            except asyncio.TimeoutError: # Expected if queue is empty
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {task_name} was cancelled.")
                if current_task_object and current_task_object.status != "completed":
                    current_task_object.mark_as_failed("Worker cancelled during processing")
                    # Optionally log the failed task if needed for recovery
                break
            except Exception as e:
                task_idx_str = str(current_task_object.index) if current_task_object else "unknown_task_index"
                mnemonic_str = current_task_object.mnemonic_phrase[:15] if current_task_object else (mnemonic_phrase[:15] if mnemonic_phrase else "unknown_mnemonic")
                logger.error(f"Error processing Task {task_idx_str} (mnemonic {mnemonic_str}...): {e}", exc_info=True)
                self.app_stats["errors_encountered"] +=1
                if current_task_object:
                    current_task_object.mark_as_failed(str(e))
                    # Log the failed task so its state is recorded
                    await self.log_result(current_task_object, False, history_score if 'history_score' in locals() else None)
            finally:
                if mnemonic_phrase is not None: # Check if mnemonic was successfully retrieved
                    self.async_mnemonic_queue.task_done()
        logger.info(f"Worker {task_name} stopping.")

    async def log_result(self, task: Task, found_balance: bool, history_score: float = None):
        """Logs the detailed result for a Task object."""

        log_payload = task.to_dict() # Use Task's own serialization method

        # Determine the timestamp for the log entry
        # The task.to_dict() already includes creation_time and completion_time.
        # The 'timestamp' field in the log_payload should represent when this specific log event occurred.
        # For consistency, we can override the 'timestamp' from task.to_dict() if it had one,
        # or just add it if task.to_dict() doesn't produce a top-level 'timestamp'.
        # Task.to_dict() does not add a 'timestamp' field itself, it has 'creation_time' and 'completion_time'.
        # So, we are adding a new 'log_event_timestamp'.

        if task.status in ["completed", "failed"] and task.completion_time:
            # If task is finalized, use its completion time for the main log event timestamp.
            log_payload["log_event_timestamp"] = task.completion_time.isoformat()
        else:
            # Otherwise, use current time (e.g. for partially processed tasks or if completion_time isn't set yet)
            log_payload["log_event_timestamp"] = datetime.now().isoformat()

        if history_score is not None:
            log_payload["history_score"] = round(history_score, 4) # Add/overwrite if classifier was used

        # Add overall_balance_found_for_mnemonic for clarity at the top level of the log
        log_payload["overall_balance_found_for_mnemonic"] = found_balance


        log_message_json = json.dumps(log_payload)

        # Log to checked.txt (which is now handled by logger_setup.py 'CheckedWalletsLogger')
        # So, get that logger and use it.
        checked_logger = logging.getLogger('CheckedWalletsLogger')
        await asyncio.get_running_loop().run_in_executor(self.executor, checked_logger.info, log_message_json)

        if found_balance:
            found_logger = logging.getLogger('FoundWalletsLogger')
            # For found.txt, we might want a slightly different or more concise format if it's human-readable
            # For now, using the same JSON payload.
            await asyncio.get_running_loop().run_in_executor(self.executor, found_logger.info, log_message_json)

        # Update WebSocket queue for UI
        # Send a summary or the full payload depending on UI needs
        # For now, let's send a summary if it's just for table update, or full if UI can handle it.
        # Let's send a structured summary for the main table, and full details if a row is expanded.
        # For now, just push the full payload to the websocket queue.
        try:
            # Create a distinct object for WebSocket to avoid issues if log_payload is modified
            ws_payload = dict(log_payload)
            ws_payload["type"] = "checked_wallet"
            if found_balance:
                ws_payload["type"] = "found_wallet" # UI can highlight this

            self.websocket_message_queue.put_nowait(ws_payload)
        except asyncio.QueueFull:
            logger.warning("WebSocket message queue full. UI updates may be lagging.")
        except Exception as e:
            logger.error(f"Error putting message to WebSocket queue: {e}")


    # --- Web Server Methods (aiohttp) ---
    async def handle_websocket(self, request):
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        logger.info("WebSocket client connected.")
        request.app['websockets'].add(ws) # Add to app's set of websockets

        # Send current app status immediately on connection
        try:
            await ws.send_json({"type": "app_status", "data": self.app_stats})
        except Exception as e:
            logger.error(f"Error sending initial status to WebSocket: {e}")


        try:
            async for msg in ws: # Listen for messages from client (e.g., control commands)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        logger.info(f"WebSocket received command: {data}")
                        if data.get("command") == "pause":
                            self._pause_event.clear() # Pause processing
                            self.app_stats["status"] = "Paused"
                            logger.info("Processing paused via WebSocket.")
                        elif data.get("command") == "resume":
                            self._pause_event.set() # Resume processing
                            self.app_stats["status"] = "Running"
                            logger.info("Processing resumed via WebSocket.")
                        elif data.get("command") == "stop":
                            logger.info("Stop command received via WebSocket. Initiating shutdown.")
                            self._stop_event.set()
                            self._pause_event.set() # Ensure unpaused for shutdown
                            # Further shutdown logic is handled by main start() loop's finally block
                        # Send updated status back
                        self.update_websocket_message_queue() # This will send app_status via broadcast
                    except json.JSONDecodeError:
                        logger.warning(f"WebSocket received invalid JSON: {msg.data}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket command: {e}", exc_info=True)

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket connection closed with exception {ws.exception()}")
        except Exception as e:
            logger.error(f"Exception in WebSocket handler: {e}", exc_info=True)
        finally:
            logger.info("WebSocket client disconnected.")
            request.app['websockets'].discard(ws)
        return ws

    async def get_status_http(self, request):
        return aiohttp.web.json_response(self.app_stats)

    async def handle_control_http(self, request):
        try:
            data = await request.json()
            command = data.get("command")
            logger.info(f"HTTP control command received: {command}")
            if command == "pause":
                self._pause_event.clear()
                self.app_stats["status"] = "Paused"
                return aiohttp.web.json_response({"status": "success", "message": "Processing paused."})
            elif command == "resume":
                self._pause_event.set()
                self.app_stats["status"] = "Running"
                return aiohttp.web.json_response({"status": "success", "message": "Processing resumed."})
            elif command == "stop":
                self._stop_event.set()
                self._pause_event.set()
                return aiohttp.web.json_response({"status": "success", "message": "Shutdown initiated."})
            else:
                return aiohttp.web.json_response({"status": "error", "message": "Unknown command."}, status=400)
        except json.JSONDecodeError:
            return aiohttp.web.json_response({"status": "error", "message": "Invalid JSON payload."}, status=400)
        except Exception as e:
            logger.error(f"Error in HTTP control handler: {e}", exc_info=True)
            return aiohttp.web.json_response({"status": "error", "message": "Internal server error."}, status=500)


    async def broadcast_to_websockets(self):
        """Periodically sends updates from websocket_message_queue to all connected clients."""
        while not self._stop_event.is_set():
            try:
                # Send app_stats periodically or on change (more efficient)
                # For now, this loop focuses on messages from websocket_message_queue
                message_to_send = await asyncio.wait_for(self.websocket_message_queue.get(), timeout=0.5)
                if message_to_send:
                    # Create a list of sockets to iterate over to avoid issues if set changes during iteration
                    sockets_to_send_to = list(self.web_app['websockets'])
                    for ws_client in sockets_to_send_to:
                        if not ws_client.closed:
                            try:
                                await ws_client.send_json(message_to_send)
                            except ConnectionResetError:
                                logger.warning("WebSocket ConnectionResetError during broadcast. Client likely disconnected.")
                                self.web_app['websockets'].discard(ws_client) # Remove if error
                            except RuntimeError as e: # Eg "RuntimeError: Socket is closed"
                                logger.warning(f"WebSocket RuntimeError during broadcast: {e}. Client likely disconnected.")
                                self.web_app['websockets'].discard(ws_client)
                            except Exception as e:
                                logger.error(f"Error sending message to WebSocket client: {e}")
                                self.web_app['websockets'].discard(ws_client) # Assume problematic
                    self.websocket_message_queue.task_done()
            except asyncio.TimeoutError: # Expected, means queue was empty
                continue
            except Exception as e:
                logger.error(f"Error in WebSocket broadcast loop: {e}", exc_info=True)
                await asyncio.sleep(1) # Avoid fast loop on persistent error

    def update_websocket_message_queue(self):
        """Puts the current app_stats onto the websocket_message_queue."""
        try:
            # Update dynamic stats before sending
            self.app_stats["mnemonics_checked_session"] = self.mnemonics_processed_in_session
            self.app_stats["wallets_found_session"] = self.found_wallets_count_session
            self.app_stats["current_api_rate_limit"] = self.current_api_rate
            self.app_stats["active_processing_workers"] = len(self.processing_worker_tasks)
            self.app_stats["input_queue_size"] = self.mp_mnemonic_input_queue.qsize() if hasattr(self.mp_mnemonic_input_queue, 'qsize') else -1
            self.app_stats["output_queue_size"] = self.async_mnemonic_queue.qsize()

            time_elapsed_session = time.time() - self.session_start_time
            if time_elapsed_session > 0 :
                self.app_stats["mnemonics_per_second_session"] = round(self.mnemonics_processed_in_session / time_elapsed_session, 2)
            else:
                self.app_stats["mnemonics_per_second_session"] = 0

            self.websocket_message_queue.put_nowait({"type": "app_status", "data": self.app_stats})
        except asyncio.QueueFull:
            logger.warning("WebSocket message queue full while trying to update app_status.")
        except Exception as e:
            logger.error(f"Error updating WebSocket queue with app_status: {e}")

    async def _periodic_stats_updater(self):
        """Periodically calls update_websocket_message_queue."""
        while not self._stop_event.is_set():
            await self._pause_event.wait() # Respect pause for this too
            try:
                self.update_websocket_message_queue()
                # Also log API stats from APIHandler periodically if it's not doing it itself
                if self.api_handler:
                    await self.api_handler.log_api_endpoint_stats_periodically()

            except Exception as e:
                logger.error(f"Error in periodic_stats_updater: {e}", exc_info=True)

            # Determine sleep duration based on configured interval (e.g., 1 second for UI updates)
            await asyncio.sleep(self.config.APP_STATS_UI_UPDATE_INTERVAL_SECONDS)


    async def run_web_server(self):
        self.web_app = aiohttp.web.Application()
        self.web_app['websockets'] = set() # Store active WebSocket connections

        self.web_app.router.add_get('/ws', self.handle_websocket)
        self.web_app.router.add_get('/status', self.get_status_http)
        self.web_app.router.add_post('/control', self.handle_control_http)

        # Configure CORS if origins are set
        if self.config.CORS_ALLOWED_ORIGINS:
            import aiohttp_cors
            cors = aiohttp_cors.setup(self.web_app, defaults={
                origin: aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*", allow_headers="*"
                ) for origin in self.config.CORS_ALLOWED_ORIGINS
            })
            for route in list(self.web_app.router.routes()):
                cors.add(route)

        runner = aiohttp.web.AppRunner(self.web_app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, self.config.WEB_SERVER_HOST, self.config.WEB_SERVER_PORT)

        logger.info(f"Starting web server on http://{self.config.WEB_SERVER_HOST}:{self.config.WEB_SERVER_PORT}")
        logger.info(f"WebSocket endpoint on ws://{self.config.WEB_SERVER_HOST}:{self.config.WEB_SERVER_PORT}/ws")

        # Start the site and the broadcast loop concurrently
        # The site itself runs until stop_event is set (or KeyboardInterrupt)
        # We need a way to gracefully shut down the site runner.

        # The broadcast_to_websockets task needs to be started and managed
        broadcast_task = asyncio.create_task(self.broadcast_to_websockets())

        try:
            await site.start()
            # Keep site running while stop_event is not set
            while not self._stop_event.is_set():
                await asyncio.sleep(1) # Check stop_event periodically
        except KeyboardInterrupt: # Should be caught by main start loop
            logger.info("Web server received KeyboardInterrupt.")
        except Exception as e:
            logger.error(f"Web server run error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down web server...")
            if not broadcast_task.done(): broadcast_task.cancel()
            await runner.cleanup() # Gracefully stop the AppRunner
            logger.info("Web server shut down.")


async def main_entry_point(args): # Accept args
    # Setup logging using the centralized logger_setup module
    # This should be the ONLY place logging is configured initially.
    logger_setup.setup_logging(config_obj=app_config) # Pass the config object

    # Profiling (optional)
    if args.profile_mem:
        tracemalloc.start()
        logger.info("Tracemalloc (memory profiling) started.")

    if args.profile_cpu:
        profiler = cProfile.Profile()
        profiler.enable()
        logger.info("cProfile (CPU profiling) started.")

    # Initialize BalanceChecker with the loaded config
    checker = BalanceChecker(cfg=app_config)

    try:
        await checker.start(args) # Pass args to BalanceChecker's start if needed
    except KeyboardInterrupt:
        logger.info("Main entry: KeyboardInterrupt caught, application will shut down via BalanceChecker's finally block.")
    except Exception as e:
        logger.critical(f"Main entry: Unhandled exception: {e}", exc_info=True)
    finally:
        logger.info("Main entry: Application shutdown sequence initiated or completed.")
        if args.profile_cpu and 'profiler' in locals():
            profiler.disable()
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
            ps.print_stats(30) # Print top 30 cumulative time functions
            logger.info("\n--- cProfile CPU Profile ---")
            logger.info(s.getvalue())
            logger.info("--- End cProfile CPU Profile ---")
            profile_file = os.path.join(app_config.LOG_DIR, "app_cpu_profile.prof")
            profiler.dump_stats(profile_file)
            logger.info(f"CPU profile saved to {profile_file}")


        if args.profile_mem:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            logger.info("\n--- Tracemalloc Memory Profile (Top 10) ---")
            for stat in top_stats[:10]:
                logger.info(str(stat))
            logger.info("--- End Tracemalloc Memory Profile ---")

            # Save snapshots periodically if needed (more complex)
            # For now, just one at the end.
            snapshot_dir = os.path.join(app_config.LOG_DIR, "memory_snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_file = os.path.join(snapshot_dir, f"mem_snapshot_{time.strftime('%Y%m%d-%H%M%S')}.snap")
            snapshot.dump(snapshot_file)
            logger.info(f"Memory snapshot saved to {snapshot_file}")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(description="Crypto Wallet Scanner Application")
    parser.add_argument("--profile-cpu", action="store_true", help="Enable CPU profiling with cProfile.")
    parser.add_argument("--profile-mem", action="store_true", help="Enable memory profiling with tracemalloc.")
    # Add other command-line arguments as needed from config.py to allow overrides
    # For example:
    # parser.add_argument("--config-file", type=str, help="Path to a custom YAML/JSON config file.")
    # parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Override log level.")

    cli_args = parser.parse_args()

    # TODO: Handle config overrides from CLI args if implementing that feature
    # For now, app_config is loaded directly from finder.config

    asyncio.run(main_entry_point(cli_args))
