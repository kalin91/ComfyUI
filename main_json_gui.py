"""Main entry point for JSON Manager GUI."""
from json_gui import json_manager_starter
from json_gui.json_manager_gui import main


if __name__ == "__main__":
    json_manager_starter.apply_custom_paths()
    main()
