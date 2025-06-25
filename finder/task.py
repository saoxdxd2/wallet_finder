from datetime import datetime
from typing import List, Dict, Any, Optional
from bip_utils import Bip44Coins

class Task:
    """
    Represents a unit of work for processing a mnemonic phrase,
    including its derived addresses and their states.
    """
    def __init__(self, mnemonic_phrase: str, index: int):
        self.mnemonic_phrase: str = mnemonic_phrase
        self.index: int = index
        self.derived_addresses: Dict[str, List[Dict[str, Any]]] = {} # Key: coin_name (str), Value: list of address_details
        self.status: str = "pending" # e.g., pending, processing, completed, failed
        self.creation_time: datetime = datetime.now()
        self.completion_time: Optional[datetime] = None
        self.error_message: Optional[str] = None

    def set_derived_addresses(self, derived_addresses_by_coin: Dict[Bip44Coins, List[Dict[str, Any]]]):
        """
        Sets the derived addresses for the task.
        Converts Bip44Coins enum keys to string keys for JSON serialization friendliness.
        Initializes addresses with default 'checked', 'balance', 'has_funds', 'has_on_chain_history'.
        """
        self.derived_addresses = {}
        for coin_enum, addr_list in derived_addresses_by_coin.items():
            coin_name = coin_enum.name # Use coin name as string key
            self.derived_addresses[coin_name] = []
            for addr_info in addr_list:
                self.derived_addresses[coin_name].append({
                    "address": addr_info["address"],
                    "path": addr_info["path"],
                    "balance": 0.0,
                    "has_funds": False,
                    "has_on_chain_history": False,
                    "checked": False # Mark as not checked yet
                })
        self.status = "processing" # Update status as addresses are now derived

    def update_address_details(self, coin_name: str, address_str: str, balance: float, has_funds: bool, has_history: bool):
        """
        Updates the details for a specific derived address.
        """
        if coin_name in self.derived_addresses:
            for addr_details in self.derived_addresses[coin_name]:
                if addr_details["address"] == address_str:
                    addr_details["balance"] = balance
                    addr_details["has_funds"] = has_funds
                    addr_details["has_on_chain_history"] = has_history
                    addr_details["checked"] = True
                    return
            # print(f"Warning: Address {address_str} not found under coin {coin_name} for task {self.index}") # Optional logging
        # else:
            # print(f"Warning: Coin {coin_name} not found in derived addresses for task {self.index}") # Optional logging

    def mark_as_completed(self):
        """Marks the task as completed and sets completion time."""
        self.status = "completed"
        self.completion_time = datetime.now()

    def mark_as_failed(self, error_message: str):
        """Marks the task as failed and records an error message."""
        self.status = "failed"
        self.error_message = error_message
        self.completion_time = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Task object to a dictionary, suitable for JSON serialization.
        """
        return {
            "index": self.index,
            "mnemonic_phrase": self.mnemonic_phrase,
            "status": self.status,
            "creation_time": self.creation_time.isoformat(),
            "completion_time": self.completion_time.isoformat() if self.completion_time else None,
            "error_message": self.error_message,
            "derived_addresses": self.derived_addresses # Already in JSON-friendly format
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """
        Creates a Task object from a dictionary (e.g., loaded from JSON).
        Note: This is a basic version. For full fidelity, time strings would need parsing.
        """
        task = cls(mnemonic_phrase=data["mnemonic_phrase"], index=data["index"])
        task.status = data.get("status", "pending")
        task.creation_time = datetime.fromisoformat(data["creation_time"]) if data.get("creation_time") else datetime.now()
        if data.get("completion_time"):
            task.completion_time = datetime.fromisoformat(data["completion_time"])
        task.error_message = data.get("error_message")
        task.derived_addresses = data.get("derived_addresses", {}) # Assumes derived_addresses is already in correct format
        return task

if __name__ == '__main__':
    # Example Usage
    sample_mnemonic = "test mnemonic phrase for task example usage"
    task_idx = 12345

    # Create a task
    task = Task(mnemonic_phrase=sample_mnemonic, index=task_idx)
    print(f"Initial Task: {task.to_dict()}")

    # Simulate deriving addresses
    # (Using Bip44Coins directly here for example, in app it would be Bip44Coins enum)
    simulated_derived_data = {
        Bip44Coins.BITCOIN: [
            {"address": "btc_address_1", "path": "m/44'/0'/0'/0/0"},
            {"address": "btc_address_2", "path": "m/44'/0'/0'/0/1"}
        ],
        Bip44Coins.ETHEREUM: [
            {"address": "eth_address_1", "path": "m/44'/60'/0'/0/0"}
        ]
    }
    task.set_derived_addresses(simulated_derived_data)
    print(f"\nTask after setting derived addresses: {task.to_dict()}")

    # Simulate updating address details
    task.update_address_details(Bip44Coins.BITCOIN.name, "btc_address_1", 0.005, True, True)
    task.update_address_details(Bip44Coins.ETHEREUM.name, "eth_address_1", 1.2, True, False)
    # Simulate checking an address that wasn't found (or a typo)
    task.update_address_details(Bip44Coins.BITCOIN.name, "btc_address_non_existent", 0.1, True, True)


    # Check if all addresses for a coin are checked (example logic)
    all_btc_checked = all(addr.get("checked", False) for addr in task.derived_addresses.get(Bip44Coins.BITCOIN.name, []))
    print(f"\nAll BTC addresses checked: {all_btc_checked}")


    # Mark task as completed
    task.mark_as_completed()
    print(f"\nTask after completion: {task.to_dict()}")

    # Example of creating from dict (simplified)
    task_data_for_reload = task.to_dict()
    reloaded_task = Task.from_dict(task_data_for_reload)
    print(f"\nReloaded Task: {reloaded_task.to_dict()}")

    assert reloaded_task.index == task.index
    assert reloaded_task.mnemonic_phrase == task.mnemonic_phrase
    assert reloaded_task.derived_addresses[Bip44Coins.BITCOIN.name][0]["balance"] == 0.005

    # Test failure case
    failed_task = Task("failed mnemonic", 54321)
    failed_task.set_derived_addresses({
        Bip44Coins.LITECOIN: [{"address": "ltc_addr_1", "path": "m/44'/2'/0'/0/0"}]
    })
    failed_task.mark_as_failed("API connection error during LTC check")
    print(f"\nFailed Task: {failed_task.to_dict()}")
    assert failed_task.status == "failed"
    assert failed_task.error_message is not None

    print("\nTask class example usage complete.")
