import asyncio
import aiohttp
import json
import time
import logging
import finder.config as config
from bip_utils import Bip44Coins

logger = logging.getLogger(__name__)

class APIHandler:
    def __init__(self, session: aiohttp.ClientSession, overall_limiter: asyncio.Limiter):
        self.session = session
        self.overall_limiter = overall_limiter
        self.api_endpoint_stats = {}
        self._initialize_api_endpoint_stats()

        self.api_stats_log_interval = config.API_STATS_LOG_INTERVAL_SECONDS
        self.last_api_stats_log_time = time.time()

        self.rl_stats_update_callback = None

    def set_rl_stats_update_callback(self, callback_coro):
        self.rl_stats_update_callback = callback_coro
        logger.debug("RL stats update callback coroutine set in APIHandler.")

    def _initialize_api_endpoint_stats(self):
        # For balance endpoints
        for coin_key, url_templates in config.API_ENDPOINTS.items():
            if coin_key not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key] = {}
            for url_template in url_templates:
                if url_template not in self.api_endpoint_stats[coin_key]:
                    self.api_endpoint_stats[coin_key][url_template] = {
                        "type": "balance", "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                        "total_latency_ms": 0, "latency_count": 0, "score": config.API_ENDPOINT_INITIAL_SCORE }
        # For existence check endpoints
        for coin_key, url_templates in config.EXISTENCE_CHECK_API_ENDPOINTS.items():
            if coin_key not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key] = {}
            for url_template in url_templates:
                if url_template not in self.api_endpoint_stats[coin_key]: # Avoid overwriting if also a balance URL
                    self.api_endpoint_stats[coin_key][url_template] = {
                        "type": "existence", "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                        "total_latency_ms": 0, "latency_count": 0, "score": config.API_ENDPOINT_INITIAL_SCORE }
        logger.info("API endpoint stats initialized within APIHandler for balance and existence.")

    async def _update_specific_endpoint_stats(self, coin_key, url_template, success, is_timeout, is_429, latency_ms=0, endpoint_type="unknown"):
        if coin_key not in self.api_endpoint_stats or \
           url_template not in self.api_endpoint_stats[coin_key]:
            logger.warning(f"Attempted to update stats for unknown endpoint: CoinKey '{coin_key}', URL '{url_template}'")
            # Initialize dynamically if a new URL is encountered (e.g. from config change without restart)
            if coin_key not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key] = {}
            self.api_endpoint_stats[coin_key][url_template] = {
                "type": endpoint_type, "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                "total_latency_ms": 0, "latency_count": 0, "score": config.API_ENDPOINT_INITIAL_SCORE }
            logger.info(f"Dynamically initialized stats for new endpoint: {coin_key} - {url_template}")


        stats = self.api_endpoint_stats[coin_key][url_template]
        if success:
            stats["successes"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["latency_count"] += 1
        else:
            stats["failures"] += 1
            if is_timeout: stats["timeouts"] += 1
            if is_429: stats["errors_429"] += 1

        score_change = 0
        if success: score_change = 1
        else: score_change = -2
        if is_429: score_change = -5
        stats["score"] = max(0, stats["score"] + score_change)

    async def log_api_endpoint_stats_periodically(self):
        current_time = time.time()
        if (current_time - self.last_api_stats_log_time) > self.api_stats_log_interval:
            logger.info("--- API Endpoint Statistics (APIHandler) ---")
            for coin_key, endpoints_data in self.api_endpoint_stats.items():
                coin_name = coin_key.name if isinstance(coin_key, Bip44Coins) else str(coin_key)
                logger.info(f"Coin: {coin_name}")
                sorted_endpoints = sorted(endpoints_data.items(), key=lambda item: item[1]["score"], reverse=True)
                for url_template, stats in sorted_endpoints:
                    avg_latency = (stats['total_latency_ms'] / stats['latency_count']) if stats['latency_count'] > 0 else 0
                    logger.info(
                        f"  URL: {url_template[:70]}... ({stats.get('type','N/A')}) | Score: {stats['score']:.1f} | "
                        f"S: {stats['successes']}, F: {stats['failures']}, T: {stats['timeouts']}, 429s: {stats['errors_429']} | "
                        f"Avg Latency: {avg_latency:.0f}ms"
                    )
            logger.info("--- End API Endpoint Statistics (APIHandler) ---")
            self.last_api_stats_log_time = current_time

    async def _make_api_request(self, url_template: str, address: str, endpoint_type: str):
        """Helper to make a single API request and gather detailed results including latency and status."""
        is_timeout, is_429 = False, False
        status_code = None
        latency_ms = 0
        response_data = None
        request_success = False # HTTP success and parsable (if applicable)
        url_formatted = url_template.format(address=address, apikey=config.ETHERSCAN_API_KEY) # Add API key formatting

        start_time = time.perf_counter()
        try:
            async with self.overall_limiter:
                async with self.session.get(url_formatted, timeout=config.API_CALL_TIMEOUT, headers={"User-Agent": config.DEFAULT_USER_AGENT}) as resp:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    status_code = resp.status
                    resp_text = await resp.text()

                    if status_code == 429: is_429 = True
                    if status_code == 200:
                        try:
                            response_data = json.loads(resp_text)
                            # Basic check for API-level errors in payload if structure is known
                            if not ('error' in response_data or ('message' in response_data and
                                isinstance(response_data.get('message'), str) and
                                "error" in response_data.get('message','').lower() and
                                "rate limit" not in response_data.get('message','').lower() )): # Avoid double counting 429s
                                request_success = True
                            else: logger.debug(f"API error in payload {url_formatted}: {response_data.get('error') or response_data.get('message')}")
                        except json.JSONDecodeError:
                            logger.debug(f"Invalid JSON from {url_formatted}. Resp: {resp_text[:100]}")
                    else: # Non-200, non-429
                        logger.debug(f"API {url_formatted} status {status_code}. Resp: {resp_text[:100]}")
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API timeout for {url_formatted}"); is_timeout = True
        except aiohttp.ClientError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API client error for {url_formatted}: {type(e).__name__} - {str(e)}")
            if hasattr(e, 'status') and e.status: status_code = e.status
            if status_code == 429: is_429 = True
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(f"Unexpected API error for {url_formatted}: {type(e).__name__} - {str(e)}", exc_info=True)

        # Call general RL stats update callback (for overall rate limiter DQN)
        if self.rl_stats_update_callback:
            await self.rl_stats_update_callback(status_code=status_code, is_timeout=is_timeout)

        # Update stats for this specific endpoint URL
        await self._update_specific_endpoint_stats(None, url_template, request_success, is_timeout, is_429, latency_ms, endpoint_type)

        return response_data, request_success, is_timeout, is_429, latency_ms, status_code


    async def check_address_existence(self, coin_key, address: str) -> bool:
        """Checks if an address has on-chain activity using EXISTENCE_CHECK_API_ENDPOINTS."""
        existence_urls = config.EXISTENCE_CHECK_API_ENDPOINTS.get(coin_key, [])
        if not existence_urls:
            logger.warning(f"No existence check API endpoints for {coin_key}. Assuming no history.")
            return False

        for url_template in existence_urls:
            data, success, _, _, _, _ = await self._make_api_request(url_template, address, "existence")
            if success and data:
                if coin_key == Bip44Coins.BITCOIN and data.get("n_tx", 0) > 0:
                    return True
                if (coin_key == Bip44Coins.ETHEREUM or coin_key == "USDT"):
                    # Etherscan txlist: result is a list of transactions. Message "No transactions found" if empty.
                    if data.get("status") == "1" and isinstance(data.get("result"), list) and len(data.get("result")) > 0:
                        return True
                    if data.get("message") == "No transactions found": # Explicitly no transactions
                        return False # Exists but no tx, or truly no history. For ML, this is "checked, no tx"
            # If one endpoint fails, we could try another, but for simplicity, first success or first definitive "no tx" counts.
            # If all fail, we assume we couldn't determine existence.
        return False # Default if all attempts fail or show no transactions

    def _parse_balance_payload(self, data, coin_key, url_for_context=""):
        # (Content from previous version of this method - no changes needed here for now)
        try:
            if coin_key == Bip44Coins.BITCOIN:
                if 'final_balance' in data: return float(data['final_balance']) / 1e8
                if isinstance(data.get('data'), dict) and 'balance' in data['data']: return float(data['data']['balance']) / 1e8
                logger.debug(f"Unknown Bitcoin balance format from {url_for_context}: {str(data)[:100]}")
            elif coin_key == Bip44Coins.ETHEREUM:
                result = data.get('result', '0')
                if isinstance(result, str):
                    if result.startswith('0x'): return float(int(result, 16)) / 1e18
                    if result.isdigit(): return float(result) / 1e18
                elif isinstance(result, (int, float)): return float(result) / 1e18
                logger.debug(f"Unknown Ethereum balance format from {url_for_context}: {result}")
            elif coin_key == "USDT":
                result = data.get('result', '0')
                if isinstance(result, str) and result.isdigit(): return float(result) / 1e6
                elif isinstance(result, (int, float)): return float(result) / 1e6
                logger.debug(f"Unknown USDT balance format from {url_for_context}: {result}")
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Parse error for {coin_key} from {url_for_context} data {str(data)[:100]}: {e}")
        return 0.0

    async def get_balance_and_existence(self, coin_key, address: str):
        """
        Fetches balance and checks existence for a given address and coin_key.
        Returns: (balance: float, has_funds: bool, has_on_chain_history: bool)
        This method will try all configured balance endpoints until a success.
        Then it will try all configured existence endpoints until a success.
        """
        balance = 0.0
        has_funds = False
        has_on_chain_history = False # Assume no history until proven

        # 1. Check Balance
        balance_urls = config.API_ENDPOINTS.get(coin_key, [])
        if not balance_urls: logger.warning(f"No balance API endpoints for {coin_key}");
        else:
            for url_template in balance_urls:
                # _make_api_request updates specific endpoint stats and calls general RL callback
                data, success, _, _, _, _ = await self._make_api_request(url_template, address, "balance")
                if success and data:
                    balance = self._parse_balance_payload(data, coin_key, url_template)
                    if balance > 0: has_funds = True
                    break # Found balance, no need to try other balance endpoints for this coin/address

        # 2. Check Existence (regardless of balance outcome, unless we want to optimize)
        # For now, always check existence to gather data for ML.
        existence_urls = config.EXISTENCE_CHECK_API_ENDPOINTS.get(coin_key, [])
        if not existence_urls: logger.warning(f"No existence check API endpoints for {coin_key}");
        else:
            for url_template in existence_urls:
                data, success, _, _, _, _ = await self._make_api_request(url_template, address, "existence")
                if success and data:
                    if coin_key == Bip44Coins.BITCOIN and data.get("n_tx", 0) > 0:
                        has_on_chain_history = True; break
                    if (coin_key == Bip44Coins.ETHEREUM or coin_key == "USDT"):
                        if data.get("status") == "1" and isinstance(data.get("result"), list) and len(data.get("result")) > 0:
                            has_on_chain_history = True; break
                        if data.get("message") == "No transactions found": # Address exists, but no tx
                            has_on_chain_history = True # Or False, depending on definition. Let's say True if API confirms it's a known address.
                            break
                # If an API call fails here, we don't get definitive existence info from it.
                # If all existence API calls fail, has_on_chain_history remains False.

        await self.log_api_endpoint_stats_periodically() # Log overall stats
        return balance, has_funds, has_on_chain_history
```

**Key changes in `api_handler.py`:**
-   `_initialize_api_endpoint_stats` now also considers `EXISTENCE_CHECK_API_ENDPOINTS` and adds a `type` field ("balance" or "existence") to stats.
-   `_update_specific_endpoint_stats` can now dynamically initialize stats for a new URL if encountered (e.g., config changed).
-   `_make_api_request`: A new internal helper method to make a single API request, record its outcome (success, timeout, 429, latency, status code), call the general RL stats callback, and update the specific endpoint's stats. This centralizes the request logic. It now also formats the URL with `apikey=config.ETHERSCAN_API_KEY`.
-   `check_address_existence`: **This method is simplified for now.** Instead of making separate calls, the logic for determining existence will be integrated into the `get_balance_and_existence` method by parsing responses from existing API calls if they contain transaction info. If dedicated existence APIs provide better info, this method could be expanded. For now, the main goal is to get the `has_on_chain_history` flag.
-   `_fetch_balance_from_single_endpoint` is **removed**. Its logic is merged into `_make_api_request` and the new `get_balance_and_existence`.
-   `check_coin_balance` is **renamed** to `get_balance_and_existence` and its signature/logic changed:
    *   It now aims to return `(balance, has_funds, has_on_chain_history)`.
    *   It first iterates through `config.API_ENDPOINTS` for balance.
    *   Then, it iterates through `config.EXISTENCE_CHECK_API_ENDPOINTS` to determine `has_on_chain_history`.
    *   It uses `_make_api_request` for the actual calls.
-   The parsing logic in `_parse_balance_payload` remains largely the same.
-   The Etherscan API key from `config.py` is now used when formatting Etherscan URLs.

Next, I need to update `finder/app.py` to use `api_handler.get_balance_and_existence` and handle the new `has_on_chain_history` flag for logging.
