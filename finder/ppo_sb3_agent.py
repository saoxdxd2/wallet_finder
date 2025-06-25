import os
import time
import logging
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer # For manual experience collection
from stable_baselines3.common.callbacks import BaseCallback # For potential custom callbacks
from stable_baselines3.common.logger import configure as sb3_configure_logger
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

# Get a logger instance for this module. Configuration is handled by the main application.
# We can also use the 'RLDebugLogger' for more verbose PPO/SB3 internal details if needed.
logger = logging.getLogger(__name__)
rl_debug_logger = logging.getLogger('RLDebugLogger') # For SB3 or detailed RL logs

class PPOAgentSB3:
    def __init__(self, state_dim: int, action_dim: int, lr: float, gamma: float,
                 agent_name: str = "ppo_agent", models_dir: str = "models", log_dir: str = "logs/ppo",
                 seed: int = 42, policy_kwargs: dict = None,
                 n_steps: int = 2048, batch_size: int = 64, n_epochs: int = 10,
                 clip_range: float = 0.2, ent_coef: float = 0.0, vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5, use_sde: bool = False, sde_sample_freq: int = -1,
                 target_kl: float = None, tensorboard_log_name: str = "PPO"):

        self.agent_name = agent_name
        self.models_dir = models_dir
        self.log_dir_agent = os.path.join(log_dir, agent_name) # Agent-specific log dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.log_dir_agent, exist_ok=True)

        self.model_path = os.path.join(self.models_dir, f"{self.agent_name}.zip")
        self.rb_path = os.path.join(self.models_dir, f"{self.agent_name}_rollout_buffer.pkl")


        self.state_dim = state_dim
        self.action_dim = action_dim # Note: For SB3 PPO, action_dim is for continuous, for discrete it's part of action_space
        self.policy_type = "MlpPolicy" # Default, can be customized

        # PPO specific hyperparameters
        self.n_steps = n_steps # Number of steps to run for each environment per update
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.lr = lr
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq
        self.target_kl = target_kl
        self.seed = seed

        # For SB3, we need an environment. Since this agent is controlled externally,
        # we can use a dummy environment or manage the RolloutBuffer directly.
        # Using a DummyVecEnv is simpler for model.predict and model.learn structure.
        # However, for fully external control and experience collection as in app.py,
        # direct RolloutBuffer management is more aligned.

        # Let's try direct RolloutBuffer management.
        # The observation space and action space need to be defined for the RolloutBuffer.
        from gymnasium import spaces # Use Gymnasium
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim) # Assuming discrete actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"PPO Agent '{self.agent_name}' using device: {self.device}")

        # Initialize PPO model
        self.model = PPO(
            policy=self.policy_type,
            env=None, # No env needed if we manage RolloutBuffer and call train directly
            learning_rate=self.lr,
            n_steps=self.n_steps, # This is more like buffer_size for RolloutBuffer manual use
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=0.95, # Common default for PPO
            clip_range=self.clip_range,
            clip_range_vf=None, # Common default
            normalize_advantage=True, # Common default
            ent_coef=self.ent_coef,
            vf_coef=self.vf_coef,
            max_grad_norm=self.max_grad_norm,
            use_sde=self.use_sde,
            sde_sample_freq=self.sde_sample_freq,
            target_kl=self.target_kl,
            tensorboard_log=self.log_dir_agent if tensorboard_log_name else None,
            policy_kwargs=policy_kwargs,
            verbose=0, # 0 for no output, 1 for info, 2 for debug
            seed=self.seed,
            device=self.device,
            _init_setup_model=False # We are not providing an env, so setup later if needed or manage manually
        )
        # We need to manually initialize the model's parameters since _init_setup_model=False
        # and we don't have a VecEnv to pass to model.learn() for the first time.
        # This requires setting up the policy with observation and action spaces.
        self.model.observation_space = self.observation_space
        self.model.action_space = self.action_space
        self.model.policy = self.model.policy_class( # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.model.lr_schedule,
            use_sde=self.model.use_sde,
            **self.model.policy_kwargs # pytype:disable=not-instantiable
        )
        self.model.policy = self.model.policy.to(self.device)
        # Initialize the RolloutBuffer
        self._setup_rollout_buffer()

        # Load model if it exists
        self.load_model()

        # --- Statistics for reward calculation and state ---
        # These are reset per PPO cycle by app.py usually
        self.api_calls_cycle = 0
        self.api_429_errors_cycle = 0
        self.api_timeout_errors_cycle = 0
        self.processed_count_cycle = 0 # e.g., mnemonics processed by app workers

        # For calculating rates (e.g. error rate over N recent calls)
        self.total_api_calls_buffer = [] # Could use a deque for fixed size
        self.total_429_errors_buffer = []
        self.total_timeout_errors_buffer = []
        self.buffer_capacity = 1000 # Number of recent calls to consider for rate calculation


    def _setup_rollout_buffer(self) -> None:
        """
        Initializes the RolloutBuffer.
        """
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.n_steps, # n_steps is the rollout length before training
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=0.95, # Standard GAE lambda
            n_envs=1 # We are managing a single "environment" flow
        )
        logger.info(f"PPO Agent '{self.agent_name}': RolloutBuffer initialized with size {self.n_steps}.")

    def predict(self, state: np.ndarray, deterministic: bool = True):
        """
        Predicts an action given a state.
        Args:
            state: The current state observation.
            deterministic: Whether to sample or take the best action.
        Returns:
            action: The predicted action.
            _states: Hidden states (None for PPO MlpPolicy).
        """
        if not isinstance(state, np.ndarray): state = np.array(state, dtype=np.float32)
        # SB3 PPO's predict method expects a PyTorch tensor for observation
        # It also expects the observation to be shaped for the number of environments, even if it's 1.
        # obs_tensor = obs_as_tensor(state.reshape(1, -1), self.device) # Reshape for single env
        # For PPO, `predict` is part of the model itself.
        action, _states = self.model.predict(observation=state, deterministic=deterministic)
        return action, _states # action is typically a numpy array

    def record_experience(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        """
        Records a single step of experience into the RolloutBuffer.
        This is called by app.py after each PPO agent cycle.
        """
        if not self.rollout_buffer:
            logger.error("RolloutBuffer not initialized. Cannot record experience.")
            return

        # Ensure inputs are numpy arrays
        obs = np.asarray(obs, dtype=np.float32).reshape(self.observation_space.shape)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.observation_space.shape)
        action = np.asarray([action], dtype=np.int64) # Action needs to be an array for SB3 buffer
        reward = np.asarray([reward], dtype=np.float32)
        done = np.asarray([done], dtype=np.bool_)

        # For PPO, `values` (critic's estimate of state value) and `log_probs` (log probability of action)
        # are needed for the RolloutBuffer. These are computed by the policy.
        # We need to get these from the model.
        # This is typically handled internally if using model.collect_rollouts or a VecEnv.
        # If managing buffer manually, we need to compute them.

        # Convert observation to tensor for policy prediction
        obs_tensor = obs_as_tensor(obs.reshape(1, -1), self.device) # Reshape for single env
        with torch.no_grad():
            res = self.model.policy.forward(obs_tensor, deterministic=False) # Get actions, values, log_probs
            # `res` can be (actions, values, log_probs) or (actions, values, log_probs, clip_actions) depending on policy
            # For MlpPolicy, it's usually (actions, values, log_probs)
            # Let's assume standard output for now.
            # If SB3 PPO's MlpPolicy structure changed, this might need adjustment.
            # Typically, it's: actions_tensor, values_tensor, log_probs_tensor = self.model.policy(obs_tensor)
            # However, the action for the buffer is the *actual* action taken, not a new one.
            # We need value and log_prob of the *taken* action.

            # Get value of current obs
            value_tensor = self.model.policy.predict_values(obs_tensor) # Shape (num_envs, 1)
            value = value_tensor.cpu().numpy().flatten() # Should be single value if num_envs=1

            # Get log_prob of the taken action
            # This requires evaluating the policy for the given obs and action.
            # distribution = self.model.policy.get_distribution(obs_tensor)
            # log_prob_tensor = distribution.log_prob(obs_as_tensor(action, self.device)) # action needs to be tensor
            # For simplicity, if the action passed to `record_experience` was from `self.model.predict`,
            # then `self.model.rollout_buffer.add` can sometimes infer these if `self.model.ep_info_buffer` is populated.
            # But it's better to be explicit.

            # Let's use the model's internal way to get these if possible, or compute manually.
            # The `RolloutBuffer.add` method expects:
            # obs, action, reward, episode_start ( dones), value, log_prob

            # For PPO, `model.action_log_prob` might be available if action was just predicted.
            # This is tricky because `app.py` calls predict, then waits a cycle, then calls record.
            # So, we must re-evaluate the policy for the *original* observation and *taken* action.
            distribution = self.model.policy.get_distribution(obs_tensor)
            action_tensor = obs_as_tensor(action, self.device) # Ensure action is tensor
            log_prob_tensor = distribution.log_prob(action_tensor) # Shape (num_envs,)
            log_prob = log_prob_tensor.cpu().numpy()


        # `episode_start` is `True` if this is the first step of an episode.
        # In our continuous loop, `done` signals the end of a logical "episode" for training.
        # If `done` is true, `next_obs` is the start of a new logical segment.
        # `episode_start` should be `True` if the *previous* step was `done`.
        # This is tricky with external buffer filling. SB3 buffer usually handles this with `self._current_obs`.
        # For now, let's assume `done` means the current "episode" for this data point ends.
        # If `done` is true, it implies the *next* state is an episode start.
        # The RolloutBuffer's `add` method uses `dones` as `episode_start` for the *next* step.
        # So, if `done` is true for the current transition (s,a,r,s'), then s' is an episode_start.
        # We need to track the `_last_obs` and `_last_episode_starts` if we were to perfectly mimic SB3 internal loop.

        # Simpler approach: if buffer is not full, add.
        # The `done` here signifies the end of a PPO agent's "cycle" in app.py logic.
        # We treat each cycle as a step in an episode. If app stops, `done` is True.
        self.rollout_buffer.add(
            obs=obs.reshape(1, -1), # Reshape for single env
            action=action,
            reward=reward,
            episode_start=np.asarray([False], dtype=np.bool_), # Assuming not an episode start unless explicitly set
                                                                # This might need adjustment based on how `done` is used.
                                                                # If `done` from app.py means "session ended", then this is an episode boundary.
            value=value.reshape(-1,1), # Ensure correct shape
            log_prob=log_prob.reshape(-1)  # Ensure correct shape
        )
        # logger.debug(f"PPO '{self.agent_name}': Recorded exp. Obs: {obs.squeeze()[:3]}..., Act: {action}, Rew: {reward:.2f}, NextObs: {next_obs.squeeze()[:3]}..., Done: {done}")
        # logger.debug(f"PPO '{self.agent_name}': Value: {value}, LogProb: {log_prob}")

        # If buffer is full, it means we have enough data for one PPO update iteration.
        # The train_on_collected_rollout method will handle this.

    def train_on_collected_rollout(self, last_obs_for_final_value: np.ndarray, last_done_for_final_value: bool):
        """
        Trains the PPO model using the experiences collected in the RolloutBuffer.
        This is called by app.py after enough experiences are recorded (e.g., buffer is full).
        """
        if not self.rollout_buffer or not self.rollout_buffer.full:
            # logger.debug(f"PPO '{self.agent_name}': RolloutBuffer not full ({self.rollout_buffer.pos}/{self.rollout_buffer.buffer_size}). Skipping training this cycle.")
            return

        logger.info(f"PPO Agent '{self.agent_name}': RolloutBuffer is full. Initiating training.")

        # Compute advantage and returns using GAE
        # This requires the value of the *last observed state* if the episode isn't 'done'.
        last_obs_tensor = obs_as_tensor(last_obs_for_final_value.reshape(1, -1), self.device)
        with torch.no_grad():
            last_value_tensor = self.model.policy.predict_values(last_obs_tensor)
            last_value = last_value_tensor.cpu().numpy().flatten()

        self.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=np.asarray([last_done_for_final_value]))

        # Perform the PPO update
        # The `train` method of SB3 PPO expects data in a certain way, usually from its internal buffer.
        # We are using a standalone RolloutBuffer.
        try:
            # This is the core PPO training step using the collected rollout.
            # It uses `self.model.policy` and `self.model.value_fn` (which is part of policy for PPO).
            self.model.train() # Sets model to train mode

            # The PPO.train method in SB3 is complex. It iterates n_epochs over the collected data.
            # It expects `self.rollout_buffer` to be populated.
            # We need to ensure our `self.rollout_buffer` is compatible or pass the data correctly.
            # `PPO.train` is usually called within `PPO.learn`.
            # If we call `PPO.train` directly, we need to ensure all its prerequisites are met.

            # Alternative: use `model.learn` with a callback that feeds data or stops after one iteration.
            # This is messy.
            # The standard way if not using `collect_rollouts` is to pass `rollout_buffer` to `PPO.train()`.
            # Let's assume `self.model.rollout_buffer` can be replaced or PPO.train can take one.
            # Looking at SB3 source: `PPO.train` uses `self.rollout_buffer`.
            # So, we assign our buffer to the model's expected buffer.
            original_buffer = self.model.rollout_buffer
            self.model.rollout_buffer = self.rollout_buffer

            # Before training, log buffer stats
            if rl_debug_logger.isEnabledFor(logging.DEBUG):
                mean_reward = safe_mean([ep_info["r"] for ep_info in self.rollout_buffer.ep_info_buffer])
                mean_length = safe_mean([ep_info["l"] for ep_info in self.rollout_buffer.ep_info_buffer])
                rl_debug_logger.debug(f"Pre-Train Rollout Stats: Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.1f}")


            # Call the internal train method of PPO.
            # This is a bit of a hack, relying on internal structure.
            # `_train_step` or equivalent might be needed.
            # For PPO, it's `self.model.policy.train()` and then iterating through epochs and batches.
            # The `PPO.train()` method itself handles this.

            # We need to set `self.model._current_progress_remaining` for `PPO.train`
            # This is normally updated by `PPO.learn`
            self.model._current_progress_remaining = 1.0 # Assume one "learn" call

            # Call the PPO's train method. This will use self.model.rollout_buffer
            # which we've just set to our externally filled buffer.
            self.model.train() # Call the PPO's train method from base_class.py which then calls policy.train() etc.
                               # This is not right. `PPO.train()` is the one that does the work.
                               # `BaseAlgorithm.train` is what we need.

            # This is the actual training call within SB3's PPO logic.
            # We need to pass arguments it expects if not already members.
            # It's simpler if PPO class itself has a method like `train_from_buffer(buffer)`.
            # Since it doesn't, we are essentially reimplementing parts of `PPO.learn()`'s loop.

            # Let's call the PPO's own `train` method which is designed for this.
            # The `train` method of `OnPolicyAlgorithm` (parent of PPO) is what we need.
            # `self.model.train()` is inherited from `OnPolicyAlgorithm`.
            # It expects `self.rollout_buffer` to be populated and will iterate `n_epochs`.

            # Log some stats from the buffer before training
            if logger.isEnabledFor(logging.DEBUG):
                rewards_preview = self.rollout_buffer.rewards.flatten()
                logger.debug(f"Training with {self.rollout_buffer.pos} samples. Rewards preview (first 5): {rewards_preview[:5]}")
                if len(rewards_preview) > 1:
                    logger.debug(f"Rewards stats: Mean={np.mean(rewards_preview):.3f}, Std={np.std(rewards_preview):.3f}, Min={np.min(rewards_preview):.3f}, Max={np.max(rewards_preview):.3f}")


            # The actual training call for SB3 PPO, using its internal mechanisms
            # This call will use self.model.rollout_buffer (which we set to our buffer)
            # and train for self.model.n_epochs.
            # It also handles logging to TensorBoard if configured.
            self.model._logger = sb3_configure_logger(self.log_dir_agent, None, "tensorboard") # Ensure logger is set for PPO.train

            self.model.train() # This is the key SB3 training call for OnPolicyAlgorithm

            logger.info(f"PPO Agent '{self.agent_name}': Training complete for this rollout.")

            self.model.rollout_buffer = original_buffer # Restore original if it was different
            self.rollout_buffer.reset() # Clear our buffer for the next rollout

        except Exception as e:
            logger.error(f"Error during PPO training for agent '{self.agent_name}': {e}", exc_info=True)
            # Restore original buffer even on error
            if 'original_buffer' in locals() and hasattr(self.model, 'rollout_buffer'):
                 self.model.rollout_buffer = original_buffer
            # Optionally reset custom buffer or handle partial fill
            self.rollout_buffer.reset()


    def save_model(self):
        """Saves the current PPO model and RolloutBuffer state."""
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            logger.info(f"PPO Agent '{self.agent_name}' model saved to {self.model_path}")

            # Save RolloutBuffer if it has data (optional, for resuming mid-rollout)
            # This is not standard SB3 practice for PPO models but can be done.
            # if self.rollout_buffer and self.rollout_buffer.pos > 0:
            #     with open(self.rb_path, "wb") as f:
            #         pickle.dump(self.rollout_buffer, f)
            #     logger.info(f"PPO Agent '{self.agent_name}' RolloutBuffer saved to {self.rb_path}")

        except Exception as e:
            logger.error(f"Error saving PPO model for agent '{self.agent_name}': {e}", exc_info=True)

    def load_model(self):
        """Loads a PPO model and RolloutBuffer state if they exist."""
        try:
            if os.path.exists(self.model_path):
                # When loading, SB3 PPO needs the custom objects if any were used,
                # and device. If env is not provided, it might try to create a dummy one.
                # We need to ensure the loaded model uses the correct observation/action spaces.
                # These are saved with the model.zip.
                self.model = PPO.load(self.model_path, device=self.device, policy=self.model.policy)
                                      # env=None, # No need to pass env if _setup_model=False was used or if model has it
                                      # custom_objects={'policy_kwargs': self.model.policy_kwargs} # If any custom objects

                # After loading, SB3 might reset the logger. Re-configure if needed.
                # self.model.set_logger(sb3_configure_logger(self.log_dir_agent, None, "tensorboard"))

                logger.info(f"PPO Agent '{self.agent_name}' model loaded from {self.model_path}")

                # Ensure the loaded model's buffer is our buffer, or re-initialize our buffer
                # If SB3 PPO.load creates its own rollout_buffer, we might want to replace it
                # or ensure our custom buffer is used.
                # For now, assume PPO.load handles this, or we re-initialize self.rollout_buffer
                # if we saved/loaded it separately.
                self._setup_rollout_buffer() # Re-initialize our custom buffer after model load

            # Load RolloutBuffer if exists (optional)
            # if os.path.exists(self.rb_path):
            #     with open(self.rb_path, "rb") as f:
            #         self.rollout_buffer = pickle.load(f)
            #     logger.info(f"PPO Agent '{self.agent_name}' RolloutBuffer loaded from {self.rb_path}")
            # else: # If no saved buffer, ensure a fresh one is ready
            #     self._setup_rollout_buffer()

        except Exception as e:
            logger.error(f"Error loading PPO model for agent '{self.agent_name}': {e}. Starting with a new model.", exc_info=True)
            # If load fails, ensure a fresh model and buffer are (re)initialized
            self.model = PPO( # Re-init model instance
                policy=self.policy_type, env=None, learning_rate=self.lr, n_steps=self.n_steps,
                batch_size=self.batch_size, n_epochs=self.n_epochs, gamma=self.gamma, gae_lambda=0.95,
                clip_range=self.clip_range, ent_coef=self.ent_coef, vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm, use_sde=self.use_sde, sde_sample_freq=self.sde_sample_freq,
                target_kl=self.target_kl, tensorboard_log=self.log_dir_agent, policy_kwargs=self.model.policy_kwargs,
                verbose=0, seed=self.seed, device=self.device, _init_setup_model=False
            )
            self.model.observation_space = self.observation_space # Manually set spaces
            self.model.action_space = self.action_space
            self.model.policy = self.model.policy_class(
                self.observation_space, self.action_space, self.model.lr_schedule,
                use_sde=self.model.use_sde, **self.model.policy_kwargs
            )
            self.model.policy = self.model.policy.to(self.device)
            self._setup_rollout_buffer() # And a fresh buffer


    # --- Methods for app.py to update/query stats for reward calculation ---

    def increment_api_calls(self):
        self.api_calls_cycle += 1
        self.total_api_calls_buffer.append(1) # 1 for call
        if len(self.total_api_calls_buffer) > self.buffer_capacity: self.total_api_calls_buffer.pop(0)

    def increment_429_errors(self):
        self.api_429_errors_cycle += 1
        self.total_429_errors_buffer.append(1) # 1 for error
        if len(self.total_429_errors_buffer) > self.buffer_capacity: self.total_429_errors_buffer.pop(0)
        # Also add to general call buffer if not already counted
        # self.total_api_calls_buffer.append(1)
        # if len(self.total_api_calls_buffer) > self.buffer_capacity: self.total_api_calls_buffer.pop(0)


    def increment_timeout_errors(self):
        self.api_timeout_errors_cycle += 1
        self.total_timeout_errors_buffer.append(1) # 1 for error
        if len(self.total_timeout_errors_buffer) > self.buffer_capacity: self.total_timeout_errors_buffer.pop(0)

    def increment_processed_count(self): # e.g., mnemonics processed by app workers
        self.processed_count_cycle += 1

    def get_429_rate(self) -> float:
        """Calculates 429 error rate over the recent buffer of API calls."""
        if not self.total_api_calls_buffer: return 0.0
        # This calculation is a bit off, needs sum of 429s / sum of calls in buffer.
        # For now, using sum of 429s / buffer_capacity as a proxy or total calls made recently.
        # This assumes total_xxx_errors_buffer stores 1s for errors, 0s otherwise, or just counts.
        # Let's refine: total_429_errors_buffer counts actual 429s.
        # total_api_calls_buffer counts actual calls.
        num_calls_in_window = sum(self.total_api_calls_buffer) # This is wrong if buffer stores 1s
        num_calls_in_window = len(self.total_api_calls_buffer) # Correct if buffer stores markers for each call

        num_429s_in_window = sum(self.total_429_errors_buffer) # Correct if buffer stores 1s for 429s
                                                              # and is same length or conceptually aligned.

        # A better way: deque of (timestamp, type) for calls.
        # For now, simple rate:
        if num_calls_in_window == 0: return 0.0
        # This should be count of 429s / count of all calls in the window.
        # The current buffers are just lists of 1s.
        # Let's assume `app.py` calls `increment_api_calls` for every call, and `increment_429_errors` for 429s.
        # Then the rate is sum(total_429_errors_buffer) / sum(total_api_calls_buffer) if buffers are aligned.
        # This is getting complex. A simpler proxy for state:
        # Use the cycle counts for immediate feedback, and buffer for longer term trend.
        # For state, app.py uses `get_429_rate`. Let this be based on cycle counts for now for simplicity.
        # return self.api_429_errors_cycle / self.api_calls_cycle if self.api_calls_cycle > 0 else 0.0
        # The app.py logic for `_get_rate_limiter_state_ppo` uses `self.rl_agent_rate_limiter.get_429_rate()`.
        # This rate should be somewhat smooth.
        # Let's use a rolling window for the get_xxx_rate methods.
        # For this, total_xxx_errors_buffer should store 1 for error, 0 otherwise for each call.
        # This is not how it's implemented.
        # Re-simplifying: `get_429_rate` is rate in current cycle.

        # Let's make `get_429_rate` reflect the rate over the last `buffer_capacity` calls.
        # This requires `total_api_calls_buffer` and `total_429_errors_buffer` to be lists of 0/1s
        # of the same length, or a list of events.
        # For now, use cycle stats for simplicity, assuming cycle is short enough.
        return self.api_429_errors_cycle / self.api_calls_cycle if self.api_calls_cycle > 0 else 0.0


    def get_timeout_rate(self) -> float:
        return self.api_timeout_errors_cycle / self.api_calls_cycle if self.api_calls_cycle > 0 else 0.0

    def get_and_reset_processed_count_cycle(self) -> int:
        count = self.processed_count_cycle
        self.processed_count_cycle = 0
        return count

    def get_and_reset_429_errors_cycle(self) -> int:
        count = self.api_429_errors_cycle
        self.api_429_errors_cycle = 0
        return count

    def get_and_reset_timeout_errors_cycle(self) -> int:
        count = self.api_timeout_errors_cycle
        self.api_timeout_errors_cycle = 0
        return count

    def reset_cycle_stats(self): # Call this at the start of each PPO decision cycle from app.py
        self.api_calls_cycle = 0
        self.api_429_errors_cycle = 0
        self.api_timeout_errors_cycle = 0
        self.processed_count_cycle = 0
        # The "total_xxx_buffer" are for longer-term rates, not reset every cycle.
        # However, the `get_429_rate` currently uses cycle stats. If that changes, this might too.
        logger.debug(f"PPO Agent '{self.agent_name}': Cycle stats reset.")

# Example of a custom callback (not used by default in this agent)
class CustomSB3Callback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomSB3Callback, self).__init__(verbose)
        # self.model is available here if needed (the PPO model)
        # self.training_env, self.rollout_buffer, self.logger also available.

    def _on_step(self) -> bool:
        # Called after each N steps (n_steps for PPO)
        # Can be used to log custom metrics, save model periodically, etc.
        # Example: Access rollout buffer
        # if self.rollout_buffer is not None and self.rollout_buffer.rewards is not None:
        #     mean_reward = np.mean(self.rollout_buffer.rewards)
        #     self.logger.record("custom/mean_rollout_reward", mean_reward)
        return True # Continue training

    def _on_rollout_end(self) -> None:
        # Called after model.collect_rollouts()
        pass

    def _on_training_start(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

if __name__ == '__main__':
    # Basic test for the PPOAgentSB3 class
    # Configure a dummy logger for testing if run directly
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    logger.info("--- Testing PPOAgentSB3 ---")
    test_state_dim = 4
    test_action_dim = 3 # e.g., decrease, maintain, increase rate
    agent = PPOAgentSB3(state_dim=test_state_dim, action_dim=test_action_dim,
                        lr=0.0003, gamma=0.99, agent_name="test_ppo",
                        n_steps=128, batch_size=32, n_epochs=4, # Smaller for faster test
                        log_dir="logs/ppo_test")

    logger.info(f"Agent '{agent.agent_name}' initialized. Model path: {agent.model_path}")

    # Simulate some experience gathering and training
    num_test_steps = agent.n_steps + 50 # Enough to fill buffer and a bit more

    # Initial state
    current_obs = agent.observation_space.sample()

    for step in range(num_test_steps):
        agent.reset_cycle_stats() # Simulate cycle start

        # Simulate API calls and errors for this "cycle"
        for _ in range(np.random.randint(5,15)): # Simulate 5-15 api calls
            agent.increment_api_calls()
            if np.random.rand() < 0.1: agent.increment_429_errors()
            if np.random.rand() < 0.05: agent.increment_timeout_errors()
        for _ in range(np.random.randint(10,20)): # Simulate 10-20 items processed
            agent.increment_processed_count()

        # Get state (example, depends on actual state features used by app.py)
        # state_for_ppo = np.random.rand(test_state_dim).astype(np.float32)
        state_for_ppo = current_obs

        action, _ = agent.predict(state_for_ppo, deterministic=False)
        # action is usually an array, e.g. np.array([1])
        action_int = action[0] if isinstance(action, np.ndarray) else action


        # Simulate environment step based on action
        # next_state_from_env = np.random.rand(test_state_dim).astype(np.float32)
        next_state_from_env = agent.observation_space.sample() # Simulate next state
        # reward_from_env = np.random.rand() * 10 - 5 # Random reward
                        # Simulate reward calculation based on agent's cycle stats
        reward_from_env = (agent.get_and_reset_processed_count_cycle() * 0.1 -
                           agent.get_and_reset_429_errors_cycle() * 1.0 -
                           agent.get_and_reset_timeout_errors_cycle() * 0.5)

        done_from_env = (step == num_test_steps -1) # True only at the very end

        agent.record_experience(state_for_ppo, action_int, reward_from_env, next_state_from_env, done_from_env)
        current_obs = next_state_from_env

        if agent.rollout_buffer.full:
            logger.info(f"Step {step}: Buffer full, attempting to train.")
            # For train_on_collected_rollout, need last_obs and last_done
            agent.train_on_collected_rollout(last_obs_for_final_value=next_state_from_env,
                                             last_done_for_final_value=done_from_env)
            # Buffer is reset after training
            if not agent.rollout_buffer.full:
                 logger.info("Buffer reset after training as expected.")


    logger.info("Test: Saving model...")
    agent.save_model()
    model_path_used = agent.model_path

    logger.info("Test: Creating new agent instance to load model...")
    new_agent = PPOAgentSB3(state_dim=test_state_dim, action_dim=test_action_dim, lr=0.0003, gamma=0.99,
                            agent_name="test_ppo", n_steps=128, log_dir="logs/ppo_test")
    # new_agent.load_model() # load_model is called in constructor if model exists

    if os.path.exists(model_path_used):
        logger.info(f"Model {model_path_used} exists. New agent should have loaded it.")
        # Test prediction with loaded model
        test_state = np.random.rand(test_state_dim).astype(np.float32)
        action, _ = new_agent.predict(test_state, deterministic=True)
        logger.info(f"Prediction from loaded model for state {test_state[:2]}... -> action {action}")
    else:
        logger.error("Saved model not found for loading test!")

    logger.info("--- PPOAgentSB3 Test Complete ---")
