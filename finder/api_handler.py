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
        for coin_key, url_templates in config.API_ENDPOINTS.items():
            self.api_endpoint_stats[coin_key] = {}
            for endpoint_url_template in url_templates:
                self.api_endpoint_stats[coin_key][endpoint_url_template] = {
                    "successes": 0, "failures": 0, "timeouts": 0, "errors_429": 0,
                    "total_latency_ms": 0, "latency_count": 0,
                    "score": config.API_ENDPOINT_INITIAL_SCORE
                }
        logger.info("API endpoint stats initialized within APIHandler.")

    async def _update_specific_endpoint_stats(self, coin_key, url_template, success, is_timeout, is_429, latency_ms=0):
        if coin_key not in self.api_endpoint_stats or \
           url_template not in self.api_endpoint_stats[coin_key]:
            logger.warning(f"Attempted to update stats for unknown endpoint: CoinKey '{coin_key}', URL '{url_template}'")
            return

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
                        f"  URL: {url_template[:70]}... | Score: {stats['score']:.1f} | "
                        f"S: {stats['successes']}, F: {stats['failures']}, T: {stats['timeouts']}, 429s: {stats['errors_429']} | "
                        f"Avg Latency: {avg_latency:.0f}ms"
                    )
            logger.info("--- End API Endpoint Statistics (APIHandler) ---")
            self.last_api_stats_log_time = current_time

    async def _fetch_balance_from_single_endpoint(self, url_template: str, coin_key, address: str):
        is_timeout_local, is_429_local = False, False
        status_code_local = None
        latency_ms = 0
        url_formatted = url_template.format(address=address)
        start_time = time.perf_counter()
        balance_val = 0.0
        success_flag = False

        try:
            async with self.overall_limiter:
                async with self.session.get(url_formatted, timeout=config.API_CALL_TIMEOUT, headers={"User-Agent": config.DEFAULT_USER_AGENT}) as resp:
                    latency_ms = int((time.perf_counter() - start_time) * 1000)
                    status_code_local = resp.status
                    resp_text = await resp.text()

                    if status_code_local == 429: is_429_local = True

                    if not is_429_local and status_code_local == 200:
                        try:
                            data = json.loads(resp_text)
                            if not ('error' in data or ('message' in data and isinstance(data.get('message'), str) and
                                           "error" in data.get('message','').lower() and
                                           not "rate limit" in data.get('message','').lower())):
                                balance_val = self._parse_balance_payload(data, coin_key, url_formatted)
                                success_flag = True
                            else:
                                logger.debug(f"API error in payload for {address} at {url_formatted}: {data.get('error', data.get('message'))}")
                        except json.JSONDecodeError:
                            logger.debug(f"Invalid JSON from {url_formatted}. Resp: {resp_text[:100]}")
                    elif not is_429_local:
                         logger.debug(f"API {url_formatted} status {status_code_local}. Resp: {resp_text[:100]}")

        except asyncio.TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API timeout for {address} at {url_formatted}")
            is_timeout_local = True
        except aiohttp.ClientError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.debug(f"API client error for {address} at {url_formatted}: {type(e).__name__} - {str(e)}")
            if hasattr(e, 'status') and e.status: status_code_local = e.status
            if status_code_local == 429: is_429_local = True
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.warning(f"Unexpected API error for {address} at {url_formatted}: {type(e).__name__} - {str(e)}", exc_info=True)

        if self.rl_stats_update_callback:
            await self.rl_stats_update_callback(status_code=status_code_local, is_timeout=is_timeout_local)

        return balance_val, success_flag, is_timeout_local, is_429_local, latency_ms

    def _parse_balance_payload(self, data, coin_key, url_for_context=""):
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

    async def check_coin_balance(self, coin_key, address: str):
        endpoint_url_templates = config.API_ENDPOINTS.get(coin_key, [])
        if not endpoint_url_templates:
            logger.warning(f"No API endpoints for coin key: {coin_key}")
            return 0.0

        tasks_info = [{"url_template": url, "task": asyncio.create_task(
            self._fetch_balance_from_single_endpoint(url, coin_key, address),
            name=f"fetch_{str(coin_key)}_{url[:30]}"
        )} for url in endpoint_url_templates]

        final_balance = 0.0
        first_success_found = False
        all_task_results_for_stats = []

        active_tasks = [ti["task"] for ti in tasks_info]

        for completed_task in asyncio.as_completed(active_tasks):
            # Find corresponding url_template for the completed task
            url_template_for_task = "unknown_url" # Default
            for ti in tasks_info:
                if ti["task"] == completed_task:
                    url_template_for_task = ti["url_template"]
                    break
            try:
                balance, success, is_timeout, is_429, latency_ms = await completed_task

                all_task_results_for_stats.append({
                    "url_template": url_template_for_task, "coin_key": coin_key,
                    "success": success, "is_timeout": is_timeout, "is_429": is_429, "latency_ms": latency_ms
                })

                if success and balance > 0 and not first_success_found:
                    final_balance = balance
                    first_success_found = True
                    # Optionally, cancel remaining tasks if a positive balance is found and we want to optimize
                    # for ti_to_cancel in tasks_info:
                    #    if not ti_to_cancel["task"].done() and ti_to_cancel["task"] != completed_task:
                    #        ti_to_cancel["task"].cancel()
                    # break # Exit as_completed loop if we cancel others
            except asyncio.CancelledError:
                logger.debug(f"Task for {url_template_for_task} was cancelled.")
                all_task_results_for_stats.append({
                    "url_template": url_template_for_task, "coin_key": coin_key,
                    "success": False, "is_timeout": False, "is_429": False, "latency_ms": 0
                })
            except Exception as e:
                logger.error(f"Error processing completed task result for {url_template_for_task}: {e}", exc_info=True)
                all_task_results_for_stats.append({
                    "url_template": url_template_for_task, "coin_key": coin_key,
                    "success": False, "is_timeout": True, "is_429": False, "latency_ms": 0
                })

        for res in all_task_results_for_stats:
            await self._update_specific_endpoint_stats(
                res["coin_key"], res["url_template"],
                res["success"], res["is_timeout"], res["is_429"], res["latency_ms"]
            )

        # Ensure any tasks not processed by as_completed (e.g., if loop broke early or due to external cancellation) are handled.
        # This is more robust if cancellation logic is added to the as_completed loop.
        remaining_tasks = [ti["task"] for ti in tasks_info if not ti["task"].done()]
        if remaining_tasks:
            await asyncio.gather(*remaining_tasks, return_exceptions=True)

        await self.log_api_endpoint_stats_periodically()
        return final_balance

```
