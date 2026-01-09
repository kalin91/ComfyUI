"""Utility functions for JSON GUI management."""

import os
import re
import logging
import shutil
from functools import partial
from abc import ABC, abstractmethod
from typing import Callable, Optional
import json_gui.server as __  # noqa: F401, E402 pylint: disable=C0413
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


def _get_output_files_recursive() -> tuple[list[str], str]:
    """Returns a list of output files filtered by content types."""
    output_folder = folder_paths.get_output_directory()
    files, _ = folder_paths.recursive_search(output_folder, excluded_dir_names=[".git"])
    return sorted(files), output_folder


def get_folder_files_recursive(folder: str) -> tuple[list[str], str]:
    """Retrieves the list of filenames and the directory they are located in."""
    input_dir = folder_paths.get_filename_list_(folder)
    result: tuple[list[str], str] = input_dir[0], next(iter(input_dir[1].keys()))
    logging.debug("Input directory for %s; folder %s; files: %s", folder, result[1], result[0])
    return sorted(result[0]), result[1]


def save_image(
    data: dict[str, bool | list[str]],
    images: torch.Tensor,
    identifier: str,
    steps: int,  # from flow instance
    file_identifier: str,  # from flow instance
    is_temp: bool = True,
) -> tuple[bool | dict]:
    """Saves generated images to the temporary directory and appends their paths to created_images."""
    assert isinstance(data, dict) and len(data) == 2
    # assert keys present
    assert (
        "last_saved_to_temp" in data and "created_images" in data
    ), "Data dict must contain 'last_saved_to_temp' and 'created_images' keys."
    created_images = data["created_images"]
    j: int = len(created_images)
    output_dir = folder_paths.get_temp_directory() if is_temp else folder_paths.get_output_directory()
    file_saved: bool = False
    while not file_saved and j < 40:
        for image in images:
            sampler_file_name = os.path.join(output_dir, f"{file_identifier}_{identifier}_{j}.png")
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
    data["last_saved_to_temp"] = is_temp
    if len(created_images) >= steps:
        raise EndOfFlowException(steps)


class AbsFlow(ABC):
    """Abstract base class for flow implementations."""

    @property
    def json_path(self) -> str:
        """Returns the JSON path associated with the flow."""
        return self._json_path

    @property
    def file_path(self) -> str:
        """Returns the file identifier associated with the flow."""
        return self._file_path

    @property
    def save_image(self) -> Optional[Callable[[dict[bool | list[str]], torch.Tensor, str, bool], tuple[bool | dict]]]:
        """Returns the save image callback."""
        return self._save_image

    @property
    def saved_data(self) -> dict[str, Optional[bool | list[str]]]:
        """Returns the saved data dictionary."""
        return self._saved_data

    def __init__(self, file_path: str, filename: str) -> None:
        """Initializes the AbsFlow instance."""
        self._save_image: Optional[Callable[[dict[bool | list[str]], torch.Tensor, str, bool], tuple[bool | dict]]] = (
            None
        )
        self._saved_data: dict = {"last_saved_to_temp": None, "created_images": []}
        self._file_path = file_path
        self._json_path = os.path.join(get_main_images_path(), file_path, f"{filename}.json")
        files, folder = _get_output_files_recursive()
        idx = 0
        if files:
            pattern: str = filename + r"_r(\d+)\.json$"
            # Find all matching files and extract the max index
            indexes = [
                int(re.search(pattern, os.path.basename(f)).group(1))
                for f in files
                if re.search(pattern, os.path.basename(f))
            ]
            if indexes:
                idx = max(indexes) + 1
        self._file_identifier = f"{filename}_r{idx}"

        # delete any output files with this identifier
        for f in files:
            if self._file_identifier in f:
                os.remove(f)
                logging.info("Deleted existing output file: %s", f)

        # delete any temp files with this identifier
        files, _ = folder_paths.recursive_search(folder_paths.get_temp_directory(), excluded_dir_names=[".git"])
        for f in files:
            if self._file_identifier in f:
                os.remove(os.path.join(folder, f))
                logging.info("Deleted existing temp file: %s", f)

    def run(self, steps: int) -> list[str]:
        """Runs the flow and returns a list of created image file paths."""
        # Saving a copy of json file to output directory
        self._saved_data["created_images"].clear()
        self._save_image = partial(save_image, steps=steps, file_identifier=self._file_identifier)
        output_json_path = os.path.join(folder_paths.get_output_directory(), f"{self._file_identifier}.json")
        shutil.copy2(self._json_path, output_json_path)
        logging.info("Saved flow JSON to output directory: %s", output_json_path)
        with torch.inference_mode():
            try:
                self._run_impl(steps)
            except EndOfFlowException as eofe:
                logging.info("Flow ended early after %d steps.", eofe.steps)
        if self._saved_data["last_saved_to_temp"] is True:
            # Copy last saved image to output directory
            last_image_path = self._saved_data["created_images"][-1]
            img_filename = os.path.basename(last_image_path)
            output_image_path = os.path.join(folder_paths.get_output_directory(), img_filename)
            shutil.copy2(last_image_path, output_image_path)
            logging.info("Copied final image to output directory: %s", output_image_path)
            self._saved_data["created_images"][-1] = output_image_path
        return self._saved_data["created_images"]

    @abstractmethod
    def _run_impl(self, steps: int) -> None:
        """Runs the flow and returns a list of created image file paths."""


class EndOfFlowException(Exception):
    """Custom exception to indicate the end of a flow process."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        super().__init__(f"End of flow after reaching {steps} steps limit.")
