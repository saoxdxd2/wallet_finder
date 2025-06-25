# import finder.config as config # Will use passed config_obj
from bip_utils import Bip44Coins
# Assuming finder.config also defines CustomCoin if used by API_ENDPOINTS
# from finder.config import CustomCoin # Or handle string keys carefully

logger = logging.getLogger(__name__)

class APIHandler:
    def __init__(self, config_obj, session: aiohttp.ClientSession, limiter: asyncio.Limiter, stats_callback=None):
        self.config = config_obj # Use passed config object
        self.session = session
        self.limiter = limiter # Use the limiter passed from BalanceChecker (PPO controlled)

        self.api_endpoint_stats = {}
        self._initialize_api_endpoint_stats() # Uses self.config

        self.api_stats_log_interval = self.config.API_STATS_LOG_INTERVAL_SECONDS
        self.last_api_stats_log_time = time.time()

        self.stats_update_callback = stats_callback # For RL agent stats
        if self.stats_update_callback:
            logger.debug("APIHandler initialized with stats_update_callback.")

    def _initialize_api_endpoint_stats(self):
        # For balance endpoints
        for coin_key, endpoint_details_list in self.config.API_ENDPOINTS.items():
            if coin_key not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key] = {}
            for endpoint_detail in endpoint_details_list: # Now a list of dicts
                url_template = endpoint_detail["url"]
                if url_template not in self.api_endpoint_stats[coin_key]:
                    self.api_endpoint_stats[coin_key][url_template] = {
                        "type": "balance", "source": endpoint_detail["source"],
                        "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                        "total_latency_ms": 0, "latency_count": 0, "score": self.config.API_ENDPOINT_INITIAL_SCORE
                    }
        # For existence check endpoints
        for coin_key, endpoint_details_list in self.config.EXISTENCE_CHECK_API_ENDPOINTS.items():
            if coin_key not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key] = {}
            for endpoint_detail in endpoint_details_list:
                url_template = endpoint_detail["url"]
                if url_template not in self.api_endpoint_stats[coin_key]:
                    self.api_endpoint_stats[coin_key][url_template] = {
                        "type": "existence", "source": endpoint_detail["source"],
                        "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                        "total_latency_ms": 0, "latency_count": 0, "score": self.config.API_ENDPOINT_INITIAL_SCORE
                    }
        logger.info("API endpoint stats initialized within APIHandler for balance and existence.")

    async def _update_specific_endpoint_stats(self, coin_key, url_template, success, is_timeout, is_429, latency_ms=0):
        # coin_key might be Bip44Coins enum or CustomCoin enum/str from config
        # url_template is the key in self.api_endpoint_stats[coin_key]

        # Ensure coin_key is correctly mapped if it's an enum for dict access
        # (The keys in self.api_endpoint_stats should match how they come from config)

        if coin_key not in self.api_endpoint_stats or \
           url_template not in self.api_endpoint_stats[coin_key]:
            # This case should ideally not happen if _initialize_api_endpoint_stats covers all config.
            # However, if config can change dynamically, this is a fallback.
            logger.warning(f"Stats entry not pre-initialized for: CoinKey '{coin_key}', URL '{url_template}'. Dynamic init (not fully supported).")
            # For safety, let's try to find its source if it's a known URL from config
            source_name = "unknown_source"
            endpoint_type = "unknown_type"
            # Try to find in balance endpoints
            for ep_detail in self.config.API_ENDPOINTS.get(coin_key, []):
                if ep_detail["url"] == url_template:
                    source_name = ep_detail["source"]
                    endpoint_type = "balance"
                    break
            # Try to find in existence endpoints if not found in balance
            if source_name == "unknown_source":
                for ep_detail in self.config.EXISTENCE_CHECK_API_ENDPOINTS.get(coin_key, []):
                    if ep_detail["url"] == url_template:
                        source_name = ep_detail["source"]
                        endpoint_type = "existence"
                        break

            if coin_key not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key] = {}
            self.api_endpoint_stats[coin_key][url_template] = {
                "type": endpoint_type, "source": source_name,
                "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                "total_latency_ms": 0, "latency_count": 0, "score": self.config.API_ENDPOINT_INITIAL_SCORE }
            logger.info(f"Dynamically initialized stats for endpoint: {coin_key} - {url_template}")

        stats = self.api_endpoint_stats[coin_key][url_template]
        if success:
            stats["successes"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["latency_count"] += 1
            stats["score"] = max(0, stats["score"] + self.config.API_SCORE_SUCCESS_INCREMENT)
        else:
            stats["failures"] += 1
            stats["score"] = max(0, stats["score"] + self.config.API_SCORE_FAILURE_DECREMENT)
            if is_timeout:
                stats["timeouts"] += 1
                stats["score"] = max(0, stats["score"] + self.config.API_SCORE_TIMEOUT_DECREMENT - self.config.API_SCORE_FAILURE_DECREMENT) # Adjust as timeout is specific
            if is_429:
                stats["errors_429"] += 1
                stats["score"] = max(0, stats["score"] + self.config.API_SCORE_429_DECREMENT - self.config.API_SCORE_FAILURE_DECREMENT) # Adjust for 429

    async def log_api_endpoint_stats_periodically(self):
        current_time = time.time()
        if (current_time - self.last_api_stats_log_time) > self.api_stats_log_interval:
            logger.info("--- API Endpoint Statistics (APIHandler) ---")
            for coin_key, endpoints_data in self.api_endpoint_stats.items():
                # Get coin name from enum if possible
                coin_name = coin_key.name if hasattr(coin_key, 'name') else str(coin_key)
                logger.info(f"Coin: {coin_name}")

                sorted_endpoints = sorted(endpoints_data.items(), key=lambda item: item[1]["score"], reverse=True)
                for url_template, stats in sorted_endpoints:
                    avg_latency = (stats['total_latency_ms'] / stats['latency_count']) if stats['latency_count'] > 0 else 0
                    logger.info(
                        f"  Src: {stats.get('source','N/A')} ({stats.get('type','N/A')}) | Score: {stats['score']:.1f} | "
                        f"S: {stats['successes']}, F: {stats['failures']}, T: {stats['timeouts']}, 429: {stats['errors_429']} | "
                        f"Avg Latency: {avg_latency:.0f}ms | URL: {url_template[:50]}..."
                    )
            logger.info("--- End API Endpoint Statistics (APIHandler) ---")
            self.last_api_stats_log_time = current_time

    async def _make_api_request(self, coin_key_for_stats, url_template: str, address: str, endpoint_type: str):
        is_timeout, is_429, other_failure = False, False, False
        status_code = None
        latency_ms = 0
        response_data = None
        request_success = False

        # Choose API key based on URL or a more robust mechanism if needed
        apikey_to_use = ""
        if "etherscan" in url_template: apikey_to_use = self.config.ETHERSCAN_API_KEY
        elif "blockcypher" in url_template: apikey_to_use = self.config.BLOCKCYPHER_API_KEY

        url_formatted = url_template.format(address=address, apikey=apikey_to_use)

        start_time = time.perf_counter()
        try:
            async with self.limiter: # Use the PPO-controlled limiter
                # Proxy is now handled by aiohttp if session is created with proxy connector
                # So, no need to pass proxy to session.get() if session is pre-configured.
                # APIHandler's session is passed from BalanceChecker, which configures proxy.
                async with self.session.get(url_formatted, timeout=self.config.API_CALL_TIMEOUT, headers={"User-Agent": self.config.DEFAULT_USER_AGENT}) as resp:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    status_code = resp.status
                    resp_text = await resp.text()

                    if status_code == 429: is_429 = True
                    if status_code == 200:
                        try:
                            response_data = json.loads(resp_text)
                            if not ('error' in response_data or ('message' in response_data and
                                isinstance(response_data.get('message'), str) and
                                "error" in response_data.get('message','').lower() and
                                "rate limit" not in response_data.get('message','').lower() )):
                                request_success = True
                            else:
                                logger.debug(f"API error in payload {url_formatted}: {response_data.get('error') or response_data.get('message')}")
                                other_failure = True # Payload indicates error
                        except json.JSONDecodeError:
                            logger.debug(f"Invalid JSON from {url_formatted}. Resp: {resp_text[:100]}")
                            other_failure = True
                    else: # Non-200, non-429
                        logger.debug(f"API {url_formatted} status {status_code}. Resp: {resp_text[:100]}")
                        other_failure = True # Mark as other failure if not 200 or 429
        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API timeout for {url_formatted}"); is_timeout = True
        except aiohttp.ClientError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API client error for {url_formatted}: {type(e).__name__} - {str(e)}")
            if hasattr(e, 'status') and e.status: status_code = e.status
            if status_code == 429: is_429 = True
            else: other_failure = True
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(f"Unexpected API error for {url_formatted}: {type(e).__name__} - {str(e)}", exc_info=True)
            other_failure = True

        if self.stats_update_callback:
            await self.stats_update_callback(
                status_code=status_code,
                is_timeout=is_timeout,
                is_429=is_429,
                is_other_failure=other_failure and not is_timeout and not is_429 # Only true if not already timeout/429
            )

        await self._update_specific_endpoint_stats(coin_key_for_stats, url_template, request_success, is_timeout, is_429, latency_ms)

        return response_data, request_success


    async def get_balance_and_existence(self, coin_key, address: str):
        balance = 0.0
        has_funds = False
        has_on_chain_history = False
        source_info = {"balance_source": None, "existence_source": None}

        # 1. Check Balance
        balance_endpoint_details = self.config.API_ENDPOINTS.get(coin_key, [])
        if not balance_endpoint_details: logger.warning(f"No balance API endpoints for {coin_key}")
        else:
            # Sort by score if available, highest first (conceptual, stats dict is flat by URL now)
            # For now, iterate in configured order.
            for endpoint_detail in balance_endpoint_details:
                url_template = endpoint_detail["url"]
                data, success = await self._make_api_request(coin_key, url_template, address, "balance")
                if success and data:
                    balance = self._parse_balance_payload(data, coin_key, url_template)
                    if balance > 0: has_funds = True
                    source_info["balance_source"] = endpoint_detail["source"]
                    break

        # 2. Check Existence
        existence_endpoint_details = self.config.EXISTENCE_CHECK_API_ENDPOINTS.get(coin_key, [])
        if not existence_endpoint_details: logger.warning(f"No existence check API endpoints for {coin_key}")
        else:
            for endpoint_detail in existence_endpoint_details:
                url_template = endpoint_detail["url"]
                data, success = await self._make_api_request(coin_key, url_template, address, "existence")
                if success and data:
                    source_info["existence_source"] = endpoint_detail["source"]
                    # Define has_on_chain_history based on API response structure
                    # This logic needs to be robust per API provider.
                    # Example for Blockchair (used for BTC, LTC, DOGE in new config):
                    if "blockchair" in endpoint_detail["source"]:
                        # Blockchair dashboard data structure: data.<address>.address.transaction_count
                        addr_data = data.get("data", {}).get(address, {})
                        if addr_data and addr_data.get("address", {}).get("transaction_count", 0) > 0:
                            has_on_chain_history = True; break
                        elif addr_data and "address" in addr_data: # Address known by blockchair, but 0 tx
                            has_on_chain_history = False # Explicitly known, but no history
                            break
                    # Example for Etherscan (ETH, USDT)
                    elif "etherscan" in endpoint_detail["source"]:
                        if data.get("status") == "1" and isinstance(data.get("result"), list) and len(data.get("result")) > 0:
                            has_on_chain_history = True; break
                        if data.get("message", "").lower() == "no transactions found":
                            has_on_chain_history = False; break # Known, but no tx
                    # Example for blockchain.info (BTC)
                    elif "blockchain.info" in endpoint_detail["source"] and "n_tx" in data:
                        if data.get("n_tx", 0) > 0:
                            has_on_chain_history = True; break
                        else: # n_tx is 0
                            has_on_chain_history = False; break
                    # Example for dogechain.info
                    elif "dogechain.info" in endpoint_detail["source"]:
                        if isinstance(data, list) and len(data) > 0 : # TX list
                             has_on_chain_history = True; break
                        if isinstance(data, dict) and data.get("success") == 1 and data.get("balance") is not None : # Balance check implies existence
                             # If balance check also returns tx count, use it, else infer from successful balance check
                             # This part needs specific API doc review for dogechain.info existence criteria
                             # For now, assume successful balance check implies it's known, but not necessarily history
                             pass # Does not confirm history, only that address is valid format for API.

        # Periodically log stats (might be better in a separate recurring task in app.py)
        # await self.log_api_endpoint_stats_periodically()

        # Return source_info for debugging or advanced logic
        return balance, has_funds, has_on_chain_history #, source_info
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
