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


def get_scripts_folder_path() -> str:
    """Returns the path to the scripts folder."""
    scripts_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "scripts")
    if not os.path.exists(scripts_path):
        os.makedirs(scripts_path)
    return scripts_path


def get_flow_and_body_path(script_name: str) -> tuple[str, str]:
    """Returns the path to an specific script."""
    script_dir = os.path.join(get_scripts_folder_path(), script_name)
    # Verify that the script dir exists and is a directory
    assert os.path.isdir(script_dir), f"Script directory {script_name} does not exist."
    flow = os.path.join(script_dir, "flow.py")
    body = os.path.join(script_dir, "body.yml")

    # Verify that flow and body exists and are files
    assert os.path.isfile(flow), f"Flow script {flow} does not exist."
    assert os.path.isfile(body), f"Body file {body} does not exist."
    return flow, body


def get_input_files_recursive() -> tuple[list[str], str]:
    """Returns a list of input files filtered by content types."""
    input_folder = folder_paths.get_input_directory()
    output_list = set()
    files, _ = folder_paths.recursive_search(input_folder, excluded_dir_names=[".git"])
    output_list.update(folder_paths.filter_files_content_types(files, ["image"]))
    return sorted(output_list), input_folder


def get_folder_files_recursive(folder: str) -> tuple[list[str], str]:
    """Retrieves the list of filenames and the directory they are located in."""
    input_dir = folder_paths.get_filename_list_(folder)
    result: tuple[list[str], str] = input_dir[0], next(iter(input_dir[1].keys()))
    logging.debug("Input directory for %s; folder %s; files: %s", folder, result[1], result[0])
    return sorted(result[0]), result[1]
