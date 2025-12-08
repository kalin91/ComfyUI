"""Utility functions for JSON GUI management."""
import os
import logging
import folder_paths


def get_main_images_path() -> str:
    """Returns the path to the main images directory."""

    ret_path: str = os.path.join(folder_paths.get_user_directory(), "images")
    if not os.path.exists(ret_path):
        os.makedirs(ret_path)
    return ret_path


def get_main_script_path(script_name: str) -> str:
    """Returns the path to an specific script."""
    script_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scripts", f"{script_name}.py")
    # Verify that the script exists and is a file
    assert os.path.isfile(script_dir), f"Script {script_name} does not exist."
    return script_dir


def get_input_files_recursive() -> list[str]:
    """Returns a list of input files filtered by content types."""
    input_folder = folder_paths.get_input_directory()
    output_list = set()
    files, _ = folder_paths.recursive_search(input_folder, excluded_dir_names=[".git"])
    output_list.update(folder_paths.filter_files_content_types(files, ["image"]))
    return sorted(output_list)


def get_folder_files_recursive(folder: str) -> list[str]:
    """Retrieves the list of filenames and the directory they are located in."""
    input_dir = folder_paths.get_filename_list_(folder)
    result: tuple[list[str], str] = input_dir[0], next(iter(input_dir[1].keys()))
    logging.debug("Input directory for %s; folder %s; files: %s", folder, result[0], result[1])
    return result[0]
