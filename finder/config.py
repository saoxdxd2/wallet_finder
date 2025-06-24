import os
from bip_utils import Bip44Coins

# --- Application Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 'finder' directory
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = BASE_DIR

GENERATOR_STATE_FILE = os.path.join(LOG_DIR, "generator_state.txt")
CHECKED_WALLETS_FILE_PATH = os.path.join(LOG_DIR, "checked.txt")
FOUND_WALLETS_FILE_PATH = os.path.join(LOG_DIR, "found.txt")

# --- API Configuration ---
API_ENDPOINTS = {
    Bip44Coins.BITCOIN: [
        "https://blockchain.info/balance?active={address}",
        "https://chain.api.btc.com/v3/address/{address}",
    ],
    Bip44Coins.ETHEREUM: [
        "https://api.etherscan.io/api?module=account&action=balance&address={address}",
    ],
    "USDT": [
        "https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress=0xdac17f958d2ee523a2206206994597c13d831ec7&address={address}&tag=latest",
    ]
}
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
API_CALL_TIMEOUT = 10
INITIAL_OVERALL_API_RATE = 10.0
TCP_CONNECTOR_LIMIT_PER_HOST = 5

# --- DQN Agent Hyperparameters ---
DQN_BUFFER_SIZE = int(1e4)
DQN_BATCH_SIZE = 64
DQN_GAMMA = 0.99
DQN_TAU = 1e-3
DQN_LR = 5e-4
DQN_UPDATE_EVERY = 4
DQN_DEFAULT_EPSILON_START = 1.0
DQN_DEFAULT_EPSILON_DECAY = 0.995
DQN_DEFAULT_EPSILON_MIN = 0.01

# --- Rate Limiter (RL) DQN Specifics ---
RL_DQN_CYCLE_INTERVAL_SECONDS = 30
RL_AGENT_STATE_SIZE = 4
RL_AGENT_ACTION_SIZE = 3
RL_MIN_RATE_LIMIT = 1.0
RL_MAX_RATE_LIMIT = 50.0
RL_RATE_ADJUSTMENT_STEP = 1.0
RL_AGENT_NAME = "rate_limiter_agent"

# --- Worker Count (WC) DQN Specifics ---
WC_DQN_CYCLE_INTERVAL_SECONDS = 45
WC_AGENT_STATE_SIZE = 3
WC_AGENT_ACTION_SIZE = 3
WC_MIN_PROCESSING_WORKERS = 1
WC_MAX_PROCESSING_WORKERS = (os.cpu_count() or 1) * 4
WC_WORKER_ADJUSTMENT_STEP = 1
WC_AGENT_NAME = "worker_count_agent"

# --- API Endpoint Stats Tracking ---
API_STATS_LOG_INTERVAL_SECONDS = 600
API_ENDPOINT_INITIAL_SCORE = 100.0

# --- Logging Configuration ---
APP_LOG_LEVEL = "INFO"
CHECKED_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
CHECKED_LOG_BACKUP_COUNT = 5

# --- Child Address Derivation ---
NUM_CHILD_ADDRESSES_TO_CHECK = 1 # Will be changed to 10 in the next step of the plan

# --- GUI Configuration ---
GUI_WINDOW_TITLE = "Crypto Wallet Scanner"
DEFAULT_NUM_PROCESSING_WORKERS = (os.cpu_count() or 1) * 2

# --- Seeds ---
SEED_RL_AGENT = 42
SEED_WC_AGENT = 123

```
