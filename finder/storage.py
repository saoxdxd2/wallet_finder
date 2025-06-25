import json
import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class JsonStorage:
    """
    A utility class for saving and loading data in JSON format.
    """

    @staticmethod
    def save(data: Any, filepath: str, indent: Optional[int] = 4) -> bool:
        """
        Serializes data to JSON and saves it to the specified filepath.

        Args:
            data: The Python object to serialize (e.g., dict, list).
            filepath: The path to the file where JSON data will be saved.
            indent: The indentation level for pretty-printing JSON. None for compact.

        Returns:
            True if saving was successful, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
            logger.debug(f"Data successfully saved to JSON file: {filepath}")
            return True
        except IOError as e:
            logger.error(f"IOError saving data to {filepath}: {e}", exc_info=True)
        except TypeError as e:
            logger.error(f"TypeError serializing data to JSON for {filepath}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error saving data to {filepath}: {e}", exc_info=True)
        return False

    @staticmethod
    def load(filepath: str) -> Optional[Any]:
        """
        Loads data from a JSON file at the specified filepath.

        Args:
            filepath: The path to the JSON file to load.

        Returns:
            The deserialized Python object, or None if loading fails
            (e.g., file not found, invalid JSON).
        """
        if not os.path.exists(filepath):
            logger.debug(f"JSON file not found at {filepath}, cannot load.")
            return None
        if os.path.getsize(filepath) == 0:
            logger.warning(f"JSON file at {filepath} is empty, cannot load.")
            return None

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Data successfully loaded from JSON file: {filepath}")
            return data
        except IOError as e:
            logger.error(f"IOError loading data from {filepath}: {e}", exc_info=True)
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError: Invalid JSON in file {filepath}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error loading data from {filepath}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Configure basic logging for example usage
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Example Usage
    test_data_dir = "test_storage_data"
    os.makedirs(test_data_dir, exist_ok=True)
    test_filepath = os.path.join(test_data_dir, "test_data.json")
    test_filepath_compact = os.path.join(test_data_dir, "test_data_compact.json")

    sample_data = {
        "name": "Test User",
        "age": 30,
        "isStudent": False,
        "courses": [
            {"title": "History", "credits": 3},
            {"title": "Art", "credits": 2}
        ],
        "unicode_test": "こんにちは世界" # Hello World in Japanese
    }

    # Test saving
    print(f"\n--- Testing JsonStorage.save ---")
    if JsonStorage.save(sample_data, test_filepath):
        print(f"Sample data saved to {test_filepath}")
    else:
        print(f"Failed to save sample data to {test_filepath}")

    # Test saving compact
    if JsonStorage.save(sample_data, test_filepath_compact, indent=None):
        print(f"Sample data saved compactly to {test_filepath_compact}")
    else:
        print(f"Failed to save sample data to {test_filepath_compact}")


    # Test loading
    print(f"\n--- Testing JsonStorage.load ---")
    loaded_data = JsonStorage.load(test_filepath)
    if loaded_data:
        print(f"Data loaded from {test_filepath}:")
        # print(json.dumps(loaded_data, indent=2, ensure_ascii=False)) # Pretty print loaded data
        assert loaded_data["name"] == sample_data["name"]
        assert loaded_data["unicode_test"] == sample_data["unicode_test"]
        print("Loaded data matches sample data.")
    else:
        print(f"Failed to load data from {test_filepath}")

    loaded_data_compact = JsonStorage.load(test_filepath_compact)
    if loaded_data_compact:
        print(f"Data loaded from {test_filepath_compact}:")
        assert loaded_data_compact["name"] == sample_data["name"]
        print("Loaded compact data matches sample data.")
    else:
        print(f"Failed to load data from {test_filepath_compact}")


    # Test loading non-existent file
    print(f"\n--- Testing loading non-existent file ---")
    non_existent_filepath = os.path.join(test_data_dir, "non_existent.json")
    data_from_non_existent = JsonStorage.load(non_existent_filepath)
    if data_from_non_existent is None:
        print(f"Correctly returned None for non-existent file: {non_existent_filepath}")
    else:
        print(f"Incorrectly loaded data from non-existent file: {data_from_non_existent}")


    # Test loading invalid JSON file
    print(f"\n--- Testing loading invalid JSON file ---")
    invalid_json_filepath = os.path.join(test_data_dir, "invalid_data.json")
    with open(invalid_json_filepath, 'w') as f:
        f.write("{'name': 'Test', 'age': 30,}") # Invalid JSON (single quotes, trailing comma)

    invalid_data = JsonStorage.load(invalid_json_filepath)
    if invalid_data is None:
        print(f"Correctly returned None for invalid JSON file: {invalid_json_filepath}")
    else:
        print(f"Incorrectly loaded data from invalid JSON file: {invalid_data}")

    # Test loading empty JSON file
    print(f"\n--- Testing loading empty JSON file ---")
    empty_json_filepath = os.path.join(test_data_dir, "empty_data.json")
    with open(empty_json_filepath, 'w') as f:
        pass # Create an empty file

    empty_data = JsonStorage.load(empty_json_filepath)
    if empty_data is None:
        print(f"Correctly returned None for empty JSON file: {empty_json_filepath}")
    else:
        print(f"Incorrectly loaded data from empty JSON file: {empty_data}")


    # Clean up test files (optional)
    # import shutil
    # shutil.rmtree(test_data_dir)
    # print(f"\nCleaned up test directory: {test_data_dir}")

    print("\nJsonStorage class example usage complete.")
