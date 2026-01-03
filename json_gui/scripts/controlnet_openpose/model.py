"""Module defining the Flow class for loading and managing flow data from JSON files."""

import os
import json
import logging
from typing import Any, Callable, Optional

import torch
from json_gui.mimic_classes import (
    OpenPosePose,
    CannyEdge,
    ApplyControlNet,
    EmptyLatent,
    SimpleKSampler,
    FaceDetailerNode,
    Rotator,
    SkipLayers,
)


class Model:
    """Class representing a flow loaded from a JSON file.""" ""

    file_path: Optional[str] = None
    save_call: Optional[Callable[[torch.Tensor, str], None]] = None

    @property
    def positive(self) -> str:
        """Returns the positive prompt."""
        return self._positive

    @property
    def negative(self) -> str:
        """Returns the negative prompt."""
        return self._negative

    @property
    def apply_control_net(self) -> list[ApplyControlNet]:
        """Returns the ApplyControlNet instance."""
        return self._apply_control_net

    @property
    def empty_latent(self) -> EmptyLatent:
        """Returns the EmptyLatent instance."""
        return self._empty_latent

    @property
    def simple_k_sampler(self) -> list[SimpleKSampler]:
        """Returns the SimpleKSampler instance."""
        return self._simple_k_sampler

    @property
    def face_detailer(self) -> FaceDetailerNode:
        """Returns the FaceDetailer instance."""
        return self._face_detailer

    @property
    def rotator(self) -> Rotator:
        """Returns the Rotator instance."""
        return self._rotator

    @property
    def skip_layers_model(self) -> SkipLayers:
        """Returns the SkipLayers instance."""
        return self._skip_layers_model

    def __init__(
        self, filepath: Optional[str] = None, save_call: Optional[Callable[[torch.Tensor, str], None]] = None
    ) -> None:
        """Initializes the Flow instance by loading data from a JSON file."""
        if filepath is None and self.__class__.file_path is None:
            raise ValueError("File path must be provided at least once.")
        if save_call is None and self.__class__.save_call is None:
            raise ValueError("Save call must be provided at least once.")
        if filepath is not None:
            self.__class__.file_path = filepath
            assert os.path.exists(filepath), f"Flow file {filepath} does not exist."
            logging.info("Loading flow from %s", filepath)
        if save_call is not None:
            self.__class__.save_call = save_call
        self._file_path: str = self.__class__.file_path
        self._save_call: Callable[[torch.Tensor, str], None] = self.__class__.save_call
        self.load_json()

    def load_json(self) -> None:
        """Loads the flow data from the JSON file."""
        with open(self._file_path, "r", encoding="utf-8") as file:
            json_props: dict[str, Any] = json.load(file)
        cnet_list = json_props["apply_control_net"]
        assert isinstance(cnet_list, list), "Expected apply_control_net to be a list."
        self._apply_control_net = []
        cnet_dicts = {}
        for cnet in cnet_list:
            target_name: str = cnet["target"]
            assert target_name not in cnet_dicts, f"Duplicate target {target_name} in apply_control_net."
            assert target_name in json_props, f"Target {target_name} not found in JSON properties."
            target_dict = json_props.pop(target_name)
            if target_name == "canny_edge":
                target_inst = CannyEdge(**target_dict)
            elif target_name == "openpose_pose":
                target_inst = OpenPosePose(**target_dict)
            else:
                raise ValueError(f"Unknown ControlNet target: {target_name}")
            if not target_inst:
                cnet_dicts[target_name] = None
                continue
            target_inst.save_tensor = lambda img, name=target_name: self._save_call(img, name)  # pylint: disable=E1102
            cnet["target"] = target_inst
            cnet_dicts[target_name] = ApplyControlNet(**cnet)
        self._apply_control_net.extend([v for v in cnet_dicts.values() if v is not None])
        self._positive = json_props["positive"]
        self._negative = json_props["negative"]
        self._empty_latent = EmptyLatent(**json_props["empty_latent"])
        self._simple_k_sampler = [SimpleKSampler(**s) for s in json_props["simple_k_sampler"]]
        self._face_detailer = FaceDetailerNode(**json_props["face_detailer"])
        self._face_detailer.save_tensor = self._save_call
        self._rotator = Rotator(**json_props["rotator"])
        self._skip_layers_model = SkipLayers(**json_props["skip_layers_model"])
