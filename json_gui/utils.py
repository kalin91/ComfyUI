"""Utility functions for JSON GUI management."""

import os
import logging
import folder_paths
import torch
from PIL import Image
import numpy as np


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


def get_flow_and_body_paths(script_name: str) -> tuple[str, str]:
    """
    Returns a tuple containing the paths to the 'flow.py' and 'body.yml' files
    within the specified script directory.

    Args:
        script_name (str): The name of the script directory.

    Returns:
        tuple[str, str]: Paths to 'flow.py' and 'body.yml' within the script directory.

    Raises:
        AssertionError: If the script directory or either file does not exist.
    """
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


def save_image(
    created_images: list[str], filename: str, images: torch.Tensor, identifier: str, steps: int, is_temp: bool = True
) -> None:
    """Saves generated images to the temporary directory and appends their paths to created_images."""
    j: int = 0
    output_dir = folder_paths.get_temp_directory() if is_temp else folder_paths.get_output_directory()
    file_saved: bool = False
    while not file_saved and j < 40:
        for i, image in enumerate(images):
            sampler_file_name = os.path.join(output_dir, f"{filename}_{identifier}_{i}_{j}.png")
            if os.path.exists(sampler_file_name):
                j += 1
                continue  # Skip if already exists
            img_np = 255.0 * image.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            img_pil.save(sampler_file_name)
            logging.info("Saved refiner output to %s", sampler_file_name)
            file_saved = True
            created_images.append(sampler_file_name)
            break
    if not file_saved:
        raise RuntimeError("Failed to save refiner output after multiple attempts. clean up temp files.")
    if len(created_images) >= steps:
        raise EndOfFlowException(created_images)


class EndOfFlowException(Exception):
    """Custom exception to indicate the end of a flow process."""

    def __init__(self, created_images: list[str]) -> None:
        self.created_images = created_images
        super().__init__("End of flow reached with created images.")
