"""Main entry point for JSON Manager GUI."""
from json_gui.json_manager import json_manager_gui, json_manager_starter


if __name__ == "__main__":
    json_manager_starter.apply_custom_paths()
    json_manager_gui.main()
