import os
import sys
import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from bip_utils import Bip39SeedGenerator, Bip44, Bip44Coins
from aiolimiter import AsyncLimiter
import aiofiles
from concurrent.futures import ThreadPoolExecutor

if getattr(sys, 'frozen', False):
    # PyInstaller bundle path
    base_path = sys._MEIPASS
    os.environ["BIP39_WORDLISTS_PATH"] = os.path.join(base_path, "bip_utils", "bip", "bip39", "wordlist")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

API_ENDPOINTS = {
    Bip44Coins.BITCOIN: [
        "https://blockchain.info/balance?active={address}",
        "https://chain.api.btc.com/v3/address/{address}",
    ],
    Bip44Coins.ETHEREUM: [
        "https://api.etherscan.io/api?module=account&action=balance&address={address}",
        "https://eth-mainnet.alchemyapi.io/v2/demo/balance?address={address}",
    ],
    "USDT": [
        "https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress=0xdac17f958d2ee523a2206206994597c13d831ec7&address={address}&tag=latest",
    ]
}

class BalanceChecker:
    def __init__(self):
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=100)
        self.checked_file = "checked.txt"
        self.found_file = "found.txt"
        self.input_file = "mnemonics.txt"
        self.limiter = AsyncLimiter(1000, 1)  # Adjust based on API capacity
        self.queue = asyncio.Queue(maxsize=10000)
        self.coins = [Bip44Coins.BITCOIN, Bip44Coins.ETHEREUM, "USDT"]
        self.checked_counter = 0
        self.rotation_lock = asyncio.Lock()
        self.max_checked_lines = 2000 

    async def start(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            headers={"User-Agent": "Mozilla/5.0"}
        )
        producer = asyncio.create_task(self.load_mnemonics())
        workers = [asyncio.create_task(self.process_queue()) for _ in range(1000)]
        await asyncio.gather(producer, *workers)
        await self.session.close()
        self._cleanup_system_cache()  

    async def load_mnemonics(self):
        async with aiofiles.open(self.input_file, "r") as f:
            async for line in f:
                await self.queue.put(line.strip())
        for _ in range(1000):
            await self.queue.put(None)  # Sentinel to stop workers

    async def process_queue(self):
        while True:
            mnemonic = await self.queue.get()
            if mnemonic is None:
                break
            try:
                addresses = await self.generate_addresses(mnemonic)
                balances = {}
                for coin in self.coins:
                    if coin in addresses:
                        balance = await self.check_balance(addresses[coin], coin)
                        balances[str(coin)] = balance
                await self.log_result(mnemonic, balances)
            except Exception as e:
                logging.error(f"Error processing {mnemonic}: {str(e)}")
            finally:
                self.queue.task_done()

    async def generate_addresses(self, mnemonic):
        loop = asyncio.get_running_loop()
        try:
            seed = await loop.run_in_executor(
                self.executor, 
                Bip39SeedGenerator(mnemonic).Generate
            )
            return {
                Bip44Coins.BITCOIN: await loop.run_in_executor(
                    self.executor,
                    lambda: Bip44.FromSeed(seed, Bip44Coins.BITCOIN).PublicKey().ToAddress()
                ),
                Bip44Coins.ETHEREUM: await loop.run_in_executor(
                    self.executor,
                    lambda: Bip44.FromSeed(seed, Bip44Coins.ETHEREUM).PublicKey().ToAddress()
                )
            }
        except Exception as e:
            logging.error(f"Address gen failed: {str(e)}")
            return {}

    async def check_balance(self, address, coin):
        tasks = []
        for url in API_ENDPOINTS.get(coin, []):
            # Create task explicitly
            task = asyncio.create_task(
                self.fetch_balance(url.format(address=address), coin)
            )
            tasks.append(task)
        
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        balance = 0.0
        for task in done:
            balance = await task
            if balance > 0:
                break
        for task in pending:
            task.cancel()
        return balance

    async def fetch_balance(self, url, coin):
        try:
            async with self.limiter:
                async with self.session.get(url, timeout=5) as resp:
                    # First check status code
                    if resp.status != 200:
                        logging.debug(f"API {url} returned status {resp.status}")
                        return 0.0
                    
                    # Then try to parse JSON
                    try:
                        data = await resp.json()
                    except json.JSONDecodeError:
                        logging.debug(f"Invalid JSON from {url}")
                        return 0.0
                    
                    # Check for API error messages
                    if 'error' in data or 'message' in data:
                        logging.debug(f"API error response: {data.get('error', data.get('message'))}")
                        return 0.0
                    
                    return self.parse_balance(data, coin)
                    
        except Exception as e:
            logging.debug(f"API error {url}: {str(e)}")
            return 0.0


    def parse_balance(self, data, coin):
        try:
            if coin == Bip44Coins.BITCOIN:
                # Handle Bitcoin API variations
                if 'final_balance' in data:
                    return float(data['final_balance']) / 1e8
                if 'balance' in data:
                    return float(data['balance']) / 1e8
                if 'data' in data:
                    addr_data = next(iter(data['data'].values()))['address']
                    return float(addr_data['balance']) / 1e8
                return 0.0
                
            elif coin == Bip44Coins.ETHEREUM:
                result = data.get('result', '0')
                # Handle error messages in result field
                if isinstance(result, str) and not result.isdigit():
                    return 0.0
                return float(result) / 1e18
                
            elif coin == "USDT":
                result = data.get('result', '0')
                if isinstance(result, str) and not result.isdigit():
                    return 0.0
                return float(result) / 1e6
                
        except Exception as e:
            logging.error(f"Parse error: {str(e)}")
        return 0.0

    async def log_result(self, mnemonic, balances):
        log_entry = f"{datetime.now().isoformat()}|{mnemonic}|{json.dumps(balances)}\n"
        
        # Use a lock to prevent race conditions
        async with self.rotation_lock:
            # Append to current checked.txt
            async with aiofiles.open(self.checked_file, "a") as f:
                await f.write(log_entry)
            
            self.checked_counter += 1
            
            # Rotate file if limit reached
            if self.checked_counter >= self.max_checked_lines:
                await self._rotate_checked_file()
                self.checked_counter = 0

    async def _rotate_checked_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"checked_{timestamp}.txt"
        
        # Use executor for filesystem operations
        loop = asyncio.get_running_loop()
        if os.path.exists(self.checked_file):
            await loop.run_in_executor(
                self.executor,
                os.rename,
                self.checked_file,
                backup_name
            )
            logging.info(f"Rotated checked.txt to {backup_name}")            

async def main():
    checker = BalanceChecker()
    await checker.start()

if __name__ == "__main__":
    asyncio.run(main())