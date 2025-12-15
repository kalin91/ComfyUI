"""Main entry point for JSON Manager GUI."""
from json_gui import json_manager_starter, json_manager_gui


if __name__ == "__main__":
    json_manager_starter.apply_custom_paths()
    json_manager_gui.main()
