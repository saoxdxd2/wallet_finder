import asyncio
import aiohttp
import json
import time
import logging
from decimal import Decimal, InvalidOperation

from bip_utils import Bip44Coins
# No direct import of config, it's passed in __init__

logger = logging.getLogger(__name__)

class APIHandler:
    def __init__(self, config_obj, session: aiohttp.ClientSession, limiter: asyncio.Limiter, stats_callback=None):
        self.config = config_obj # Use passed config object
        self.session = session    # Use passed aiohttp session
        self.limiter = limiter    # Use passed asyncio.Limiter (PPO controlled)

        self.api_endpoint_stats = {} # Stores stats for each URL template
        self._initialize_api_endpoint_stats() # Uses self.config

        self.api_stats_log_interval_sec = self.config.API_STATS_LOG_INTERVAL_SECONDS
        self.last_api_stats_log_time = time.time()

        self.stats_update_callback = stats_callback # For RL agent stats (e.g., from app.py)
        if self.stats_update_callback:
            logger.info("APIHandler initialized with stats_update_callback.")
        else:
            logger.info("APIHandler initialized without stats_update_callback.")

    def _initialize_api_endpoint_stats(self):
        """Initializes stats dictionary for all configured API endpoints."""
        all_endpoints_map = {**self.config.API_ENDPOINTS, **self.config.EXISTENCE_CHECK_API_ENDPOINTS}

        for coin_key, endpoint_details_list in all_endpoints_map.items():
            # coin_key can be Bip44Coins enum or a string like "USDT_ERC20"
            # Ensure coin_key_str is consistently used for dict keys
            coin_key_str = coin_key.name if isinstance(coin_key, Bip44Coins) else str(coin_key)

            if coin_key_str not in self.api_endpoint_stats:
                self.api_endpoint_stats[coin_key_str] = {}

            for endpoint_detail in endpoint_details_list: # endpoint_detail is a dict
                url_template = endpoint_detail["url"]
                # Determine type ('balance' or 'existence') based on which dict it came from
                # This is a bit implicit; could add 'type' field to endpoint_detail in config.
                endpoint_type = "balance"
                if coin_key in self.config.EXISTENCE_CHECK_API_ENDPOINTS and \
                   any(ed["url"] == url_template for ed in self.config.EXISTENCE_CHECK_API_ENDPOINTS[coin_key]):
                    endpoint_type = "existence"

                if url_template not in self.api_endpoint_stats[coin_key_str]:
                    self.api_endpoint_stats[coin_key_str][url_template] = {
                        "type": endpoint_type,
                        "source": endpoint_detail.get("source", "unknown_source"),
                        "parser_type": endpoint_detail.get("parser_type", "default_parser"),
                        "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                        "total_latency_ms": 0, "latency_count": 0,
                        "score": self.config.API_ENDPOINT_INITIAL_SCORE
                    }
        logger.info("API endpoint stats initialized within APIHandler.")

    async def _update_specific_endpoint_stats(self, coin_key_str: str, url_template: str,
                                              success: bool, is_timeout: bool, is_429: bool, latency_ms: int = 0):
        """Updates the statistics for a specific API endpoint URL."""
        if coin_key_str not in self.api_endpoint_stats or \
           url_template not in self.api_endpoint_stats[coin_key_str]:
            logger.warning(f"Stats entry not pre-initialized for: CoinKey '{coin_key_str}', URL '{url_template}'. Attempting dynamic init (may lack parser_type/source).")
            # Fallback: dynamically create a basic entry if missing (should ideally not happen)
            if coin_key_str not in self.api_endpoint_stats: self.api_endpoint_stats[coin_key_str] = {}
            self.api_endpoint_stats[coin_key_str][url_template] = {
                "type": "unknown", "source": "unknown_dynamic", "parser_type": "default_parser",
                "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                "total_latency_ms": 0, "latency_count": 0, "score": self.config.API_ENDPOINT_INITIAL_SCORE
            }

        stats = self.api_endpoint_stats[coin_key_str][url_template]
        if success:
            stats["successes"] += 1
            stats["total_latency_ms"] += latency_ms
            stats["latency_count"] += 1
            stats["score"] = max(0, stats["score"] + self.config.API_SCORE_SUCCESS_INCREMENT)
        else:
            stats["failures"] += 1
            # General failure decrement
            stats["score"] = max(0, stats["score"] + self.config.API_SCORE_FAILURE_DECREMENT)
            # Specific penalties (additive to general failure, so adjust if config values are absolute penalties)
            if is_timeout:
                stats["timeouts"] += 1
                # If API_SCORE_TIMEOUT_DECREMENT is the total penalty for timeout:
                stats["score"] += (self.config.API_SCORE_TIMEOUT_DECREMENT - self.config.API_SCORE_FAILURE_DECREMENT)
            if is_429:
                stats["errors_429"] += 1
                stats["score"] += (self.config.API_SCORE_429_DECREMENT - self.config.API_SCORE_FAILURE_DECREMENT)
        stats["score"] = round(stats["score"], 2) # Keep score tidy

    async def log_api_endpoint_stats_periodically(self):
        """Logs current API endpoint statistics if interval has passed."""
        current_time = time.time()
        if (current_time - self.last_api_stats_log_time) > self.api_stats_log_interval_sec:
            logger.info("--- API Endpoint Statistics (APIHandler) ---")
            for coin_key_str, endpoints_data in self.api_endpoint_stats.items():
                logger.info(f"Coin: {coin_key_str}")
                # Sort by score for display
                sorted_endpoints = sorted(endpoints_data.items(), key=lambda item: item[1]["score"], reverse=True)
                for url_template, stats in sorted_endpoints:
                    avg_latency = (stats['total_latency_ms'] / stats['latency_count']) if stats['latency_count'] > 0 else 0
                    logger.info(
                        f"  Src: {stats.get('source','N/A')} ({stats.get('type','N/A')}, {stats.get('parser_type','N/A')}) | "
                        f"Score: {stats['score']:.1f} | S: {stats['successes']}, F: {stats['failures']}, "
                        f"T: {stats['timeouts']}, 429: {stats['errors_429']} | "
                        f"Avg Latency: {avg_latency:.0f}ms | URL: {url_template[:70]}..."
                    )
            logger.info("--- End API Endpoint Statistics (APIHandler) ---")
            self.last_api_stats_log_time = current_time

    async def _make_api_request(self, coin_key_str: str, address: str, endpoint_detail: dict):
        """
        Makes a single API request using the provided session and limiter.
        Updates stats and calls the PPO callback.
        Args:
            coin_key_str: String representation of the coin (for stats key).
            address: The crypto address to check.
            endpoint_detail: Dict from config (e.g. {"url": "...", "source": "...", "parser_type": "..."}).
        Returns:
            Tuple (response_data: dict/None, request_success: bool)
        """
        is_timeout, is_429, other_failure = False, False, False
        status_code = None
        latency_ms = 0
        response_data = None
        request_success = False # Assume failure unless explicitly successful

        url_template = endpoint_detail["url"]
        # API key substitution (format URL with address and any required API key)
        apikey_to_use = ""
        if "{apikey}" in url_template: # Check if placeholder exists
            if "etherscan" in endpoint_detail.get("source","").lower():
                apikey_to_use = self.config.ETHERSCAN_API_KEY
            elif "blockcypher" in endpoint_detail.get("source","").lower():
                apikey_to_use = self.config.BLOCKCYPHER_API_KEY
            # Add other API key lookups if needed

        try:
            url_formatted = url_template.format(address=address, apikey=apikey_to_use)
        except KeyError as e: # If a placeholder like 'apikey' is missing but expected
            logger.error(f"URL formatting error for {url_template} with address {address}. Missing key: {e}")
            other_failure = True # Mark as failure
            # No actual request made, so call callbacks and update stats, then return
            if self.stats_update_callback:
                await self.stats_update_callback(status_code=None, is_timeout=False, is_429=False, is_other_failure=True)
            await self._update_specific_endpoint_stats(coin_key_str, url_template, False, False, False, 0)
            return None, False


        start_time = time.perf_counter()
        try:
            async with self.limiter: # Use the externally provided PPO-controlled limiter
                # Proxy is handled by aiohttp if session is created with proxy connector in app.py
                async with self.session.get(
                    url_formatted,
                    timeout=self.config.API_CALL_TIMEOUT_SECONDS,
                    headers={"User-Agent": self.config.DEFAULT_USER_AGENT}
                ) as resp:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    status_code = resp.status
                    resp_text = await resp.text() # Read response text for parsing or debugging

                    if status_code == 429:
                        is_429 = True
                        logger.debug(f"API 429 (Rate Limit) for {url_formatted}. Resp: {resp_text[:100]}")
                    elif status_code == 200:
                        try:
                            response_data = json.loads(resp_text)
                            # Basic check for error messages in payload (can be provider-specific)
                            if not ('error' in response_data or
                                    isinstance(response_data.get('message'), str) and "error" in response_data['message'].lower() and "rate limit" not in response_data['message'].lower() or
                                    isinstance(response_data.get('status'), str) and response_data['status'] == "0" and "etherscan" in endpoint_detail.get("source","")): # Etherscan specific error status
                                request_success = True # Assume success if 200 and no obvious error in payload
                            else:
                                logger.debug(f"API error in 200 payload for {url_formatted}: {response_data.get('error') or response_data.get('message') or response_data.get('result')}")
                                other_failure = True # Payload indicates an error despite 200 OK
                        except json.JSONDecodeError:
                            logger.debug(f"Invalid JSON response from {url_formatted}. Status: 200. Response text (first 100 chars): '{resp_text[:100]}'")
                            other_failure = True
                    else: # Other non-200, non-429 status codes
                        logger.debug(f"API request to {url_formatted} failed with status {status_code}. Response: {resp_text[:100]}")
                        other_failure = True

        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000) # Capture latency even on timeout
            logger.debug(f"API timeout for {url_formatted} after {self.config.API_CALL_TIMEOUT_SECONDS}s.")
            is_timeout = True
        except aiohttp.ClientError as e: # Covers connection errors, proxy errors, etc.
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API client error for {url_formatted}: {type(e).__name__} - {str(e)}")
            if hasattr(e, 'status') and e.status == 429: is_429 = True # Some client errors might have status
            else: other_failure = True
            if hasattr(e, 'status') and e.status: status_code = e.status # Capture status if available
        except Exception as e: # Catch-all for unexpected errors during request
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(f"Unexpected API error for {url_formatted}: {type(e).__name__} - {str(e)}", exc_info=True)
            other_failure = True

        # Call the PPO stats callback if provided
        if self.stats_update_callback:
            await self.stats_update_callback(
                status_code=status_code,
                is_timeout=is_timeout,
                is_429=is_429,
                is_other_failure=other_failure and not is_timeout and not is_429 # Only true if not already counted as timeout/429
            )

        # Update this specific endpoint's persistent stats
        await self._update_specific_endpoint_stats(coin_key_str, url_template, request_success, is_timeout, is_429, latency_ms)

        return response_data, request_success

    def _parse_balance_payload(self, data: dict, coin_key, parser_type: str) -> Decimal:
        """
        Parses the balance from API response data based on coin and parser_type.
        Args:
            data: The JSON response data from the API.
            coin_key: Bip44Coins enum or string key (e.g., "USDT_ERC20").
            parser_type: String hint from config for how to parse this payload.
        Returns:
            Balance as a Decimal, or Decimal(0) if parsing fails or balance is zero.
        """
        balance = Decimal(0)
        try:
            # Placeholder for actual parsing logic based on parser_type
            # This needs to be implemented for each API provider's response structure.
            # Example for a generic 'balance' field, often in satoshis or smallest unit
            if parser_type == "blockchain_info_balance": # BTC from blockchain.info (final_balance in satoshis)
                balance_satoshi = Decimal(data.get(next(iter(data)), {}).get('final_balance', 0)) # Address is the key
                balance = balance_satoshi / Decimal("100000000") # Convert satoshis to BTC
            elif parser_type == "blockcypher_balance": # BTC from blockcypher (final_balance in satoshis)
                balance_satoshi = Decimal(data.get('final_balance', 0))
                balance = balance_satoshi / Decimal("100000000")
            elif parser_type == "blockchair_address_dashboard": # BTC, LTC, DOGE from Blockchair
                # Example: data.<address>.address.balance (smallest unit)
                # Need to get the address string that was queried to find the key in data.
                # This is complex if address not passed. Assume data is directly the <address_data> part for now.
                # This parser_type might be too generic.
                # For now, assume data is the dict for the specific address.
                balance_smallest_unit = Decimal(data.get("context",{}).get("market_price_usd", -1)) # This is wrong, placeholder
                # Correct parsing for Blockchair:
                # The actual address is a dynamic key in the `data` field.
                # e.g. data = {"<address_str>": {"address": {"balance": 100000 ...}}}
                # This requires knowing the address, which this isolated parser doesn't.
                # The calling function `get_balance_and_existence` needs to handle this.
                # For now, this parser is a placeholder.
                # Let's assume the caller extracts `data[address_str]['address']` and passes that.
                # If `data` is `data[address_str]['address']` from Blockchair:
                raw_balance = Decimal(data.get("balance", 0))
                if coin_key == Bip44Coins.BITCOIN: balance = raw_balance / Decimal("100000000")
                elif coin_key == Bip44Coins.LITECOIN: balance = raw_balance / Decimal("100000000")
                elif coin_key == Bip44Coins.DOGECOIN: balance = raw_balance / Decimal("100000000")
                # Add other coin divisions if Blockchair is used for them
            elif parser_type == "etherscan_balance": # ETH from Etherscan (result in Wei)
                balance_wei = Decimal(data.get('result', 0))
                balance = balance_wei / Decimal("1000000000000000000") # Convert Wei to ETH
            elif parser_type == "etherscan_token_balance": # ERC20 Token (USDT) from Etherscan (result in token's smallest unit)
                # USDT has 6 decimal places.
                balance_smallest_unit = Decimal(data.get('result', 0))
                if coin_key == "USDT_ERC20": # Check specific key if needed
                    balance = balance_smallest_unit / Decimal("1000000") # USDT has 6 decimals
                # Add other token divisions here if parser_type is reused
            elif parser_type == "ethplorer_address_info": # ETH from Ethplorer (ETH.balance in ETH)
                balance = Decimal(data.get("ETH", {}).get("balance", 0))
            elif parser_type == "dogechain_balance": # DOGE from dogechain.info (balance as string anount in DOGE)
                balance = Decimal(data.get("balance",0))

            # Add more parsers as needed...
            else:
                logger.warning(f"Unsupported parser_type '{parser_type}' for coin {coin_key}. Cannot parse balance.")

            if balance < 0: balance = Decimal(0) # Ensure balance is not negative

        except (InvalidOperation, TypeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing balance for coin {coin_key} with parser '{parser_type}': {e}. Data: {str(data)[:100]}", exc_info=True)
            balance = Decimal(0)
        return balance

    def _parse_existence_payload(self, data: dict, coin_key, parser_type: str, address_queried: str) -> bool:
        """
        Parses transaction history/existence from API response data.
        Args:
            data: The JSON response data from the API.
            coin_key: Bip44Coins enum or string key.
            parser_type: String hint from config for how to parse.
            address_queried: The specific address that was part of the API request.
        Returns:
            True if address has on-chain history, False otherwise.
        """
        has_history = False
        try:
            if parser_type == "blockchain_info_tx_count": # BTC from blockchain.info (n_tx)
                has_history = int(data.get("n_tx", 0)) > 0
            elif parser_type == "blockchair_tx_count": # BTC, LTC, DOGE from Blockchair dashboard
                # data is {"<address_queried>": {"address": {"transaction_count": N ...}}}
                # We need address_queried to get the actual data part.
                address_specific_data = data.get(address_queried, {}).get("address", {})
                has_history = int(address_specific_data.get("transaction_count", 0)) > 0
            elif parser_type == "etherscan_tx_list": # ETH, ERC20 tokens from Etherscan txlist
                # status="1" and result is a list of transactions (non-empty if history)
                # status="0" and message="No transactions found" if no history but valid address
                # status="0" and message="Error! Invalid address format" if address is bad (should not happen here)
                if data.get("status") == "1" and isinstance(data.get("result"), list) and len(data.get("result")) > 0:
                    has_history = True
                elif data.get("status") == "0" and data.get("message", "").lower() == "no transactions found":
                    has_history = False # Known address, but no transactions
                # Else, if status is 0 with other message, or other conditions, assume no confirmed history.
            elif parser_type == "dogechain_txs_count": # DOGE from dogechain.info/api/v1/address/txs/
                # Response is a list of TX hashes if any, or empty list/error if none.
                # Example: {"success":1,"txs":["hash1","hash2"]} or {"success":0,"error":"No transactions found"}
                if data.get("success") == 1 and isinstance(data.get("txs"), list) and len(data.get("txs")) > 0:
                    has_history = True
                elif data.get("success") == 0 and "no transactions found" in data.get("error","").lower():
                    has_history = False
            # Add more parsers...
            else:
                logger.warning(f"Unsupported parser_type '{parser_type}' for existence check on coin {coin_key}.")

        except (TypeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing existence for coin {coin_key} with parser '{parser_type}': {e}. Data: {str(data)[:100]}", exc_info=True)
            has_history = False # Default to no history on parsing error
        return has_history

    async def get_balance_and_existence(self, coin_key_enum_or_str, address: str):
        """
        Checks balance and transaction history for a given address and coin.
        Iterates through configured API endpoints, using the first successful one.
        Args:
            coin_key_enum_or_str: Bip44Coins enum member or a custom string key (e.g., "USDT_ERC20").
            address: The cryptocurrency address string.
        Returns:
            Tuple (balance: Decimal, has_funds: bool, has_on_chain_history: bool)
        """
        balance = Decimal(0)
        has_funds = False
        has_on_chain_history = False # Default to False

        # Use string representation for dictionary keys and logging
        coin_key_str = coin_key_enum_or_str.name if isinstance(coin_key_enum_or_str, Bip44Coins) else str(coin_key_enum_or_str)

        # 1. Check Balance
        # Get relevant balance endpoints for this coin_key_str from config
        # Need to handle if coin_key_enum_or_str is enum or string to match keys in API_ENDPOINTS
        balance_endpoints_for_coin = self.config.API_ENDPOINTS.get(coin_key_enum_or_str, [])

        # Sort endpoints by score (descending) before trying
        # This requires stats to be initialized and accessible by original coin_key and URL.
        # The self.api_endpoint_stats uses coin_key_str.
        if balance_endpoints_for_coin:
            sorted_balance_endpoints = sorted(
                balance_endpoints_for_coin,
                key=lambda ep_detail: self.api_endpoint_stats.get(coin_key_str, {}).get(ep_detail["url"], {}).get("score", self.config.API_ENDPOINT_INITIAL_SCORE),
                reverse=True
            )
            for endpoint_detail in sorted_balance_endpoints:
                data, success = await self._make_api_request(coin_key_str, address, endpoint_detail)
                if success and data is not None:
                    parser_type = endpoint_detail.get("parser_type", "default_parser")

                    # Special handling for Blockchair which nests data under address key
                    if parser_type == "blockchair_address_dashboard":
                        data_for_parsing = data.get("data", {}).get(address, {}).get("address", {})
                        if not data_for_parsing and address in data.get("data",{}): # Check if only address level exists
                             data_for_parsing = data.get("data", {}).get(address, {}) # For cases like address info directly
                    else:
                        data_for_parsing = data

                    if data_for_parsing: # Ensure we have something to parse
                        balance = self._parse_balance_payload(data_for_parsing, coin_key_enum_or_str, parser_type)
                        if balance > Decimal(0):
                            has_funds = True
                        # If balance is found, we might also get existence info from same payload
                        # This avoids a second call if parser_type can provide both.
                        if not has_on_chain_history: # Only if not already determined
                             if parser_type == "blockchair_address_dashboard": # Blockchair gives tx count
                                 has_on_chain_history = self._parse_existence_payload(data.get("data",{}), coin_key_enum_or_str, "blockchair_tx_count", address)
                        break # Stop on first successful balance check
                    else: # Data was None or not in expected structure for parser
                        logger.debug(f"Balance check for {coin_key_str} on {address} via {endpoint_detail['source']} succeeded but data_for_parsing was empty/invalid.")


        # 2. Check Existence (if not already determined by balance check)
        if not has_on_chain_history:
            existence_endpoints_for_coin = self.config.EXISTENCE_CHECK_API_ENDPOINTS.get(coin_key_enum_or_str, [])
            if existence_endpoints_for_coin:
                sorted_existence_endpoints = sorted(
                    existence_endpoints_for_coin,
                    key=lambda ep_detail: self.api_endpoint_stats.get(coin_key_str, {}).get(ep_detail["url"], {}).get("score", self.config.API_ENDPOINT_INITIAL_SCORE),
                    reverse=True
                )
                for endpoint_detail in sorted_existence_endpoints:
                    data, success = await self._make_api_request(coin_key_str, address, endpoint_detail)
                    if success and data is not None:
                        parser_type = endpoint_detail.get("parser_type", "default_parser")
                        # Blockchair existence parsing might need the full `data` if address is a key
                        if parser_type == "blockchair_tx_count": # Specific parser for blockchair existence
                            has_on_chain_history = self._parse_existence_payload(data.get("data",{}), coin_key_enum_or_str, parser_type, address)
                        else:
                            has_on_chain_history = self._parse_existence_payload(data, coin_key_enum_or_str, parser_type, address)

                        if has_on_chain_history: # Found history, no need to check other existence endpoints
                            break
            elif not balance_endpoints_for_coin: # No balance endpoints either
                 logger.warning(f"No balance or existence API endpoints configured for coin/key: {coin_key_str}")


        # Log overall API stats periodically (moved to app.py's periodic task)
        # await self.log_api_endpoint_stats_periodically()

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
