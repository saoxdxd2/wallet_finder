import os
from bip_utils import Bip44Coins, Bip44Changes

# --- Application Core Settings ---
APP_NAME = "CryptoWalletScanner"
APP_VERSION = "1.0.0" # TODO: Update as features are added/changed

# --- Application Paths ---
# BASE_DIR is the directory containing this config.py file (i.e., 'finder')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR) # Parent of 'finder', useful for accessing root-level files like README

# Log directory (can be same as BASE_DIR or a subfolder like 'logs')
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True) # Ensure log directory exists

# Models directory for PPO agents and other ML models
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists

# State files
GENERATOR_STATE_FILE_RAW = os.path.join(LOG_DIR, "mnemonic_generator_raw_state.txt") # For MnemonicGeneratorManager's own index
GENERATOR_STATE_FILE_APP = os.path.join(LOG_DIR, "application_processed_mnemonic_state.txt") # For BalanceChecker's processed index

# --- Logging Configuration ---
APP_LOG_LEVEL = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# Main application log file (for general app messages, errors, PPO decisions etc.)
APP_MAIN_LOG_FILE_PATH = os.path.join(LOG_DIR, "app_main.log")
APP_MAIN_LOG_MAX_BYTES = 20 * 1024 * 1024  # 20 MB
APP_MAIN_LOG_BACKUP_COUNT = 5

# Checked wallets log (every wallet checked, typically JSON)
CHECKED_WALLETS_FILE_PATH = os.path.join(LOG_DIR, "checked_wallets.jsonl") # Using .jsonl for line-delimited JSON
CHECKED_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
CHECKED_LOG_BACKUP_COUNT = 5

# Found wallets log (only wallets with balance/history, typically JSON)
FOUND_WALLETS_FILE_PATH = os.path.join(LOG_DIR, "found_wallets.jsonl")
FOUND_LOG_MAX_BYTES = 10 * 1024 * 1024    # 10 MB
FOUND_LOG_BACKUP_COUNT = 3

# RL Agent Debug Log (for verbose PPO/SB3 internal messages if needed)
DEBUG_RL_AGENT_LOG_FILE_PATH = os.path.join(LOG_DIR, "debug_rl_agent.log")
DEBUG_RL_LOG_MAX_BYTES = 10 * 1024 * 1024 # 10 MB
DEBUG_RL_LOG_BACKUP_COUNT = 3

# --- API Configuration ---
# API endpoints are now structured with more details (source, type)
# Bip44Coins.TRON, Bip44Coins.SOLANA etc. can be added if supported by bip_utils and APIs exist
COINS_TO_CHECK = [
    Bip44Coins.BITCOIN,
    Bip44Coins.ETHEREUM,
    # "USDT_ERC20", # Representing USDT on Ethereum, handled via ETH address with specific contract
    Bip44Coins.LITECOIN,
    Bip44Coins.DOGECOIN,
    # Bip44Coins.BINANCE_SMART_CHAIN, # Example: BNB
    # "USDT_TRC20" # Example: USDT on Tron
]

# Special handling for token checks might be needed, mapping them to their base coin's address format
# For now, USDT is handled via Ethereum's structure.
# If checking USDT on Tron, it would use Tron addresses.

# Structure: { Bip44CoinEnum_or_CustomString: [ { "url": "...", "source": "...", "parser_type": "..." }, ... ] }
API_ENDPOINTS = {
    Bip44Coins.BITCOIN: [
        {"url": "https://blockchain.info/balance?active={address}", "source": "blockchain.info", "parser_type": "blockchain_info_balance"},
        {"url": "https://api.blockcypher.com/v1/btc/main/addrs/{address}/balance", "source": "blockcypher", "parser_type": "blockcypher_balance"},
        {"url": "https://api.blockchair.com/bitcoin/dashboards/address/{address}", "source": "blockchair_bitcoin", "parser_type": "blockchair_address_dashboard"},
    ],
    Bip44Coins.ETHEREUM: [ # Also used for ERC20 tokens like USDT by checking balance against contract
        {"url": f"https://api.etherscan.io/api?module=account&action=balance&address={{address}}&tag=latest&apikey={{apikey}}", "source": "etherscan", "parser_type": "etherscan_balance"},
        {"url": f"https://api.ethplorer.io/getAddressInfo/{{address}}?apiKey=freekey", "source": "ethplorer", "parser_type": "ethplorer_address_info"}, # Provides ETH balance and token list
    ],
    "USDT_ERC20": [ # This is a conceptual key; app.py will use ETH address and this specific call for USDT
        {"url": f"https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress=0xdac17f958d2ee523a2206206994597c13d831ec7&address={{address}}&tag=latest&apikey={{apikey}}", "source": "etherscan_usdt", "parser_type": "etherscan_token_balance"},
    ],
    Bip44Coins.LITECOIN: [
        {"url": "https://api.blockchair.com/litecoin/dashboards/address/{address}", "source": "blockchair_litecoin", "parser_type": "blockchair_address_dashboard"},
        {"url": "https://ltc.tokenview.io/api/address/balancetrend/ltc/{address}", "source": "tokenview_litecoin", "parser_type": "tokenview_balance_trend"}, # Example, might need specific parsing
    ],
    Bip44Coins.DOGECOIN: [
        {"url": "https://api.blockchair.com/dogecoin/dashboards/address/{address}", "source": "blockchair_dogecoin", "parser_type": "blockchair_address_dashboard"},
        {"url": "https://dogechain.info/api/v1/address/balance/{address}", "source": "dogechain_info", "parser_type": "dogechain_balance"},
    ]
    # Add other coins and their endpoints here
}

EXISTENCE_CHECK_API_ENDPOINTS = {
    Bip44Coins.BITCOIN: [
        {"url": "https://blockchain.info/rawaddr/{address}?limit=1", "source": "blockchain.info_tx", "parser_type": "blockchain_info_tx_count"},
        {"url": "https://api.blockchair.com/bitcoin/dashboards/address/{address}", "source": "blockchair_bitcoin_tx", "parser_type": "blockchair_tx_count"}, # Already in balance, but good for tx count
    ],
    Bip44Coins.ETHEREUM: [
        {"url": f"https://api.etherscan.io/api?module=account&action=txlist&address={{address}}&startblock=0&endblock=99999999&page=1&offset=1&sort=asc&apikey={{apikey}}", "source": "etherscan_tx", "parser_type": "etherscan_tx_list"},
    ],
    # USDT existence is tied to ETH address history
    Bip44Coins.LITECOIN: [
        {"url": "https://api.blockchair.com/litecoin/dashboards/address/{address}", "source": "blockchair_litecoin_tx", "parser_type": "blockchair_tx_count"},
    ],
    Bip44Coins.DOGECOIN: [
        {"url": "https://api.blockchair.com/dogecoin/dashboards/address/{address}", "source": "blockchair_dogecoin_tx", "parser_type": "blockchair_tx_count"},
        {"url": "https://dogechain.info/api/v1/address/txs/{address}", "source": "dogechain_info_txs", "parser_type": "dogechain_txs_count"}, # Check if this lists transactions
    ]
}

# API Keys (replace with your actual keys if needed, or use placeholders that APIs might accept for basic rates)
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "YourApiKeyToken") # Get from env var or use default
BLOCKCYPHER_API_KEY = os.environ.get("BLOCKCYPHER_API_KEY", "YourBlockcypherApiKeyToken")
# BLOCKCHAIR_API_KEY = os.environ.get("BLOCKCHAIR_API_KEY", "YourBlockchairApiKey") # Blockchair often works without key for basic use

DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
API_CALL_TIMEOUT_SECONDS = 10  # Timeout for individual API calls
TCP_CONNECTOR_LIMIT_PER_HOST = 10 # Max simultaneous connections to a single host by aiohttp session

# API Endpoint Scoring/Selection
API_ENDPOINT_INITIAL_SCORE = 100.0
API_SCORE_SUCCESS_INCREMENT = 1.0
API_SCORE_FAILURE_DECREMENT = -5.0
API_SCORE_TIMEOUT_DECREMENT = -10.0
API_SCORE_429_DECREMENT = -20.0 # Penalize rate limit errors more heavily
API_STATS_LOG_INTERVAL_SECONDS = 300 # How often to log detailed API endpoint stats

# --- Mnemonic and Address Generation ---
MNEMONIC_STRENGTH = 256  # Options: 128 (12 words), 160 (15 words), 192 (18 words), 224 (21 words), 256 (24 words)
MNEMONIC_LANGUAGE = "english" # bip_utils supports others if wordlists are available
MNEMONIC_GENERATOR_WORKERS = max(1, (os.cpu_count() or 2) // 2) # Number of processes for generating mnemonics
MNEMONIC_GENERATOR_WORKER_SLEEP = 0.001 # Tiny sleep in generator loop to yield CPU if needed, 0 for max speed
MNEMONIC_WORKER_JOIN_TIMEOUT_SECONDS = 5 # How long to wait for generator workers to shut down

# BIP44 Derivation Path Settings
# Standard path: m / purpose' / coin_type' / account' / change / address_index
# Example: m/44'/0'/0'/0/0  (BTC, Mainnet, Account 0, External Chain, Address 0)
BIP44_PURPOSE = 44
BIP44_ACCOUNT = 0  # Default account index
BIP44_CHANGE_TYPE = Bip44Changes.CHAIN_EXT # 0 for external, 1 for internal (change addresses)
NUM_CHILD_ADDRESSES_TO_CHECK = 20 # Check first N addresses for each derivation path type (external/internal)

# --- Application Performance & Queues ---
APP_MAX_EXECUTOR_WORKERS = (os.cpu_count() or 1) * 2 # For ThreadPoolExecutor in app.py
ASYNC_MNEMONIC_QUEUE_SIZE = 20000 # Max mnemonics in the asyncio queue for processing workers
MP_MNEMONIC_QUEUE_SIZE = 50000    # Max mnemonics in the multiprocessing queue from generator workers

# --- PPO Agent General Settings ---
PPO_TRAIN_MODE = True # If False, agents will only predict, not learn. Set to False for "production" use after training.
SEED_GENERAL_PPO = 42 # General seed for PPO agents if not specified per agent

# --- PPO Rate Limiter Agent (RLA) ---
ENABLE_RL_AGENT_RATE_LIMITER = True
RL_AGENT_NAME = "rate_limiter_ppo_agent"
RL_AGENT_STATE_SIZE = 4  # Example: [current_rate_norm, 429_error_rate, timeout_rate, output_queue_fill_ratio]
RL_AGENT_ACTION_SIZE = 3 # Example: 0: decrease rate, 1: maintain rate, 2: increase rate
RL_CYCLE_INTERVAL_SECONDS = 15 # How often PPO agent makes a decision for rate limiting
# Rate Limiter bounds and step
RL_MIN_RATE_LIMIT = 0.5  # Minimum API calls per second
RL_MAX_RATE_LIMIT = 20.0 # Maximum API calls per second
RL_RATE_ADJUSTMENT_STEP = 0.5 # How much to change rate per action
INITIAL_OVERALL_API_RATE = 5.0 # Starting API rate before PPO takes over
# PPO Hyperparameters for Rate Limiter Agent
PPO_LR_RATE_LIMITER = 3e-4
PPO_GAMMA_RATE_LIMITER = 0.99
PPO_N_STEPS_RATE_LIMITER = 256 # Steps per PPO update (rollout buffer size)
PPO_BATCH_SIZE_RATE_LIMITER = 64
PPO_N_EPOCHS_RATE_LIMITER = 10
PPO_CLIP_RANGE_RATE_LIMITER = 0.2
PPO_ENT_COEF_RATE_LIMITER = 0.01 # Encourages exploration
PPO_VF_COEF_RATE_LIMITER = 0.5
SEED_RL_AGENT_RATE_LIMITER = 101
# Reward components for Rate Limiter Agent
RL_REWARD_THROUGHPUT_SCALAR = 1.0  # Reward for mnemonics processed by API callers
RL_PENALTY_429 = -10.0             # Penalty for each 429 error
RL_PENALTY_TIMEOUT = -5.0          # Penalty for each API timeout
RL_PENALTY_QUEUE_LOW_ON_INCREASE = -2.0 # Penalty for increasing rate when output queue is low
RL_PENALTY_QUEUE_HIGH = -1.0          # Penalty if output queue is too full (input to API callers)
RL_QUEUE_LOW_THRESHOLD_PENALTY = 0.1  # Queue fill ratio below which penalty applies if rate increased
RL_QUEUE_HIGH_THRESHOLD_PENALTY = 0.9 # Queue fill ratio above which penalty applies

# --- PPO Worker Count Agent (WCA) ---
ENABLE_RL_AGENT_WORKER_COUNT = True
WC_AGENT_NAME = "worker_count_ppo_agent"
WC_AGENT_STATE_SIZE = 3 # Example: [current_workers_norm, input_queue_fill_ratio, cpu_load_norm]
WC_AGENT_ACTION_SIZE = 3 # Example: 0: decrease workers, 1: maintain, 2: increase workers
WC_CYCLE_INTERVAL_SECONDS = 30 # How often PPO agent makes a decision for worker count
# Worker Count bounds and step
WC_MIN_PROCESSING_WORKERS = 1
WC_MAX_PROCESSING_WORKERS = max(2, (os.cpu_count() or 1) * 3) # Max async processing tasks in app.py
WC_WORKER_ADJUSTMENT_STEP = 1 # How many workers to add/remove per action
INITIAL_PROCESSING_WORKERS = max(1, (os.cpu_count() or 1)) # Initial number of wallet processing workers
WC_INCLUDE_CPU_LOAD_STATE = True # Whether to use CPU load as a state feature (requires psutil)
# PPO Hyperparameters for Worker Count Agent
PPO_LR_WORKER_COUNT = 5e-4
PPO_GAMMA_WORKER_COUNT = 0.99
PPO_N_STEPS_WORKER_COUNT = 128
PPO_BATCH_SIZE_WORKER_COUNT = 32
PPO_N_EPOCHS_WORKER_COUNT = 5
PPO_CLIP_RANGE_WORKER_COUNT = 0.2
PPO_ENT_COEF_WORKER_COUNT = 0.02
PPO_VF_COEF_WORKER_COUNT = 0.5
SEED_RL_AGENT_WORKER_COUNT = 202
# Reward components for Worker Count Agent
WC_REWARD_THROUGHPUT_SCALAR = 1.0 # Reward for mnemonics processed by these workers
WC_PENALTY_QUEUE_HIGH_ON_DECREASE = -5.0 # Penalize decreasing workers if input queue is full
WC_PENALTY_QUEUE_LOW_ON_INCREASE = -3.0  # Penalize increasing workers if input queue is empty
WC_REWARD_QUEUE_OPTIMAL_RANGE_BONUS = 2.0 # Bonus if input queue fill is in optimal range
WC_PENALTY_HIGH_CPU_ON_INCREASE = -2.0    # Penalty if CPU is high and workers were increased
# Thresholds for Worker Count Agent rewards
WC_QUEUE_HIGH_THRESHOLD_PENALTY = 0.9
WC_QUEUE_LOW_THRESHOLD_PENALTY = 0.1
WC_QUEUE_OPTIMAL_RANGE_LOW = 0.25
WC_QUEUE_OPTIMAL_RANGE_HIGH = 0.75
WC_CPU_HIGH_THRESHOLD_PENALTY = 85.0 # CPU percentage

# --- ML Model Integration (Mnemonic Classifier) ---
ENABLE_CLASSIFIER_SCORING = False # Default to False, enable for experimentation
# Path to the trained classifier model (e.g., joblib or ONNX file)
# Assumed to be in MODELS_DIR for simplicity
CLASSIFIER_MODEL_HISTORY_PATH = os.path.join(MODELS_DIR, "wallet_history_classifier.joblib")
CLASSIFIER_SKIP_THRESHOLD = 0.05 # If history score < this, skip detailed balance checks (if enabled)

# --- Web Server (for potential external UI or API control) ---
ENABLE_WEB_SERVER = True # Whether to run the aiohttp web server
WEB_SERVER_HOST = "127.0.0.1"
WEB_SERVER_PORT = 8080
# CORS_ALLOWED_ORIGINS = ["http://localhost:3000"] # Example for React dev server on port 3000
CORS_ALLOWED_ORIGINS = ["*"] # Allow all for simplicity in dev, tighten for production
APP_STATS_UI_UPDATE_INTERVAL_SECONDS = 1.0 # How often to push app_stats to WebSocket

# --- Proxy Configuration ---
ENABLE_PROXY = False  # Set to True to use a proxy for all API requests
# Example for TOR SOCKS proxy: "socks5h://127.0.0.1:9050" (socks5h for DNS resolution via proxy)
# Example for HTTP/HTTPS proxy: "http://user:pass@host:port" or "https://user:pass@host:port"
PROXY_URL = "" # E.g., "socks5h://localhost:9050" or "http://yourproxy:port"

# --- Miscellaneous ---
# Obsolete DQN parameters are removed.
# GUI related parameters like GUI_WINDOW_TITLE, GUI_MAX_TREE_ITEMS are removed as GUI is separate.
# DEFAULT_NUM_PROCESSING_WORKERS is replaced by INITIAL_PROCESSING_WORKERS for PPO.
# Ensure all paths and settings are appropriate for your environment.
# Placeholders like "YourApiKeyToken" should be replaced if using APIs that require them.
