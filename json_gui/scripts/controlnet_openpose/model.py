"""Module defining the Flow class for loading and managing flow data from JSON files."""

import os
import json
import logging
from typing import Any, Callable, Optional, cast

import torch
from json_gui.scripts.mimic_classes import (
    ControlNetImgPreprocessor,
    OpenPosePose,
    CannyEdge,
    ApplyControlNet,
    EmptyLatent,
    SimpleKSampler,
    FaceDetailerNode,
    Rotator,
    SkipLayers,
    MimicNode,
    Prompts
)


class Model:
    """Class representing a flow loaded from a JSON file.""" ""

    file_path: Optional[str] = None

    def set_save_call(self, value: Callable[[torch.Tensor, str], None]) -> None:
        """Sets the save image callback."""
        self._save_call = value

    @property
    def prompts(self) -> Prompts:
        """Returns the negative prompt."""
        return self._prompts

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
        if save_call is None:
            raise ValueError("Save call must be provided.")
        if filepath is not None:
            self.__class__.file_path = filepath
            assert os.path.exists(filepath), f"Flow file {filepath} does not exist."
            logging.info("Loading flow from %s", filepath)
        self._file_path: str = self.__class__.file_path
        self._save_call: Callable[[torch.Tensor, str], None] = save_call
        self.load_json()

    def load_json(self) -> None:
        """Loads the flow data from the JSON file."""
        with open(self._file_path, "r", encoding="utf-8") as file:
            json_props: dict[str, Any] = json.load(file)
        cnet_list = json_props[ApplyControlNet.key()]
        assert isinstance(cnet_list, list), "Expected apply_control_net to be a list."
        self._apply_control_net = []
        cnet_dicts = {}
        for cnet in cnet_list:
            target_name: str = cnet["target"]
            preprocesors_dict = {t.key(): t for t in [CannyEdge, OpenPosePose]}
            assert target_name in preprocesors_dict, f"Unknown ControlNet target: {target_name}"
            target_type: type = preprocesors_dict[target_name]
            assert target_name not in cnet_dicts, f"Duplicate target {target_name} in apply_control_net."
            assert target_name in json_props, f"Target {target_name} not found in JSON properties."
            target_dict = json_props.pop(target_name)
            target_inst = target_type(**target_dict)
            if not target_inst:
                cnet_dicts[target_name] = None
                continue
            target_inst.save_tensor = lambda img, name=target_name: self._save_call(img, name)  # pylint: disable=E1102
            cnet["target"] = target_inst
            cnet_dicts[target_name] = ApplyControlNet(**cnet)
        self._apply_control_net.extend([v for v in cnet_dicts.values() if v is not None])
        self._prompts = Prompts(**json_props[Prompts.key()])
        self._empty_latent = EmptyLatent(**json_props[EmptyLatent.key()])
        self._simple_k_sampler = [SimpleKSampler(**s) for s in json_props[SimpleKSampler.key()]]
        self._face_detailer = FaceDetailerNode(**json_props[FaceDetailerNode.key()])
        self._face_detailer.save_tensor = self._save_call
        self._rotator = Rotator(**json_props[Rotator.key()])
        self._skip_layers_model = SkipLayers(**json_props[SkipLayers.key()])

    def update_json(self) -> None:
        """Loads the flow data from the JSON file."""
        with open(self._file_path, "r", encoding="utf-8") as file:
            json_props: dict[str, Any] = json.load(file)
        for name, value in vars(self).items():
            assert value is not None, f"Property {name} is None."
            is_list = False
            prop_type = type(value)
            if isinstance(value, list):
                is_list = True
                prop_type = type(value[0]) if value else None
            if issubclass(prop_type, MimicNode):
                key = prop_type.key()
                data = json_props[key]
                if issubclass(prop_type, ApplyControlNet):
                    for i, item in enumerate(data):
                        item = cast(dict[str, Any], item)
                        node = cast(ApplyControlNet, value[i])
                        target_inst: ControlNetImgPreprocessor = node.target
                        target_name: str = item["target"]
                        target_dict = json_props.pop(target_name)
                        target_inst.update(**target_dict)
                        target_inst.save_tensor = lambda img, name=target_name: self._save_call(
                            img, name
                        )  # pylint: disable=E1102
                        item["target"] = target_inst
                        node.update(**item)
                else:
                    if is_list:
                        for i, item in enumerate(data):
                            item = cast(dict[str, Any], item)
                            node = cast(MimicNode, value[i])
                            node.update(**item)
                    else:
                        node = cast(MimicNode, value)
                        node.update(**data)
                        node.save_tensor = self._save_call
