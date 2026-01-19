"""ControlNet image preprocessors and application node definitions."""

from abc import ABC, abstractmethod
import pickle
import copy
import logging
from collections import Counter
from typing import Any, Generic, Optional, Tuple, Type, TypeVar, Union
import torch
from comfy_extras.nodes_images import ResizeAndPadImage
from nodes import ControlNetApplyAdvanced
from comfy.model_management import soft_empty_cache, get_torch_device
from comfy.sd import VAE
from comfy.controlnet import load_controlnet, ControlNet
from custom_nodes.comfyui_controlnet_aux import utils as aux_utils
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.canny import CannyDetector
from json_gui.scripts.mimic import MimicNode, DataWrapper
from json_gui.scripts.mimic_classes import SkipLayers, Conditional
import folder_paths

T = TypeVar("T")
M = TypeVar("M", bound=MimicNode)


class ControlNetImgPreprocessor(Generic[T, M], MimicNode[T, M], ABC):
    """Abstract base class for ControlNet image preprocessors."""

    @property
    @abstractmethod
    def controlnet_path(self) -> str:
        """Returns the ControlNet path."""

    @property
    def skip(self) -> bool:
        """Returns whether to skip this preprocessor."""
        return self._skip

    def tensor(self) -> T:
        """Returns the processed tensor."""
        return self.process()

    # pylint: disable=W0221
    def _process_impl(self) -> T:
        """Processes the image and returns a tensor."""
        res = self._tensor_impl(self._controlnet_img)
        if self._save_tensor:
            self._save_tensor(res)
        return res

    @abstractmethod
    def _tensor_impl(self, cnet_img: torch.Tensor) -> T:
        """Implementation-specific tensor processing."""

    def __init__(self, image_name: str, skip: bool) -> None:
        """Initializes the ControlNetImgPreprocessor with the given image name."""
        super().__init__()
        if type(self) is ControlNetImgPreprocessor:  # pylint: disable=C0123
            self.update(image_name=image_name, skip=skip)

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, image_name: str, skip: bool) -> None:
        """Updates the ControlNet image preprocessor."""
        self._skip = skip
        self._controlnet_img = self._upload_image(image_name)


class ApplyControlNet(MimicNode[tuple[DataWrapper[Conditional], DataWrapper[Conditional]], "ApplyControlNet"]):
    """Returns the ControlNet application parameters."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "ApplyControlNet"]]:
        res: list[MimicNode.ClassParam[Any, "ApplyControlNet"]] = []
        res.append(cls.build_class_param(SkipLayers, lambda inst: cls._set_current_model(inst) or {"model": inst}))
        return res

    @classmethod
    def key(cls) -> str:
        """Returns the key for the ApplyControlNet."""
        return "apply_control_net"

    @property
    def target(self) -> ControlNetImgPreprocessor:
        """Returns the target ControlNet image preprocessor."""
        if self._target is None:
            raise ValueError("Target ControlNet image preprocessor is not set.")
        return self._target

    CNET_CACHE: Optional[ControlNet] = None

    @staticmethod
    def _data_wrapper_call(
        controlnet_paths: list[str], cond: Conditional, cnet_attrs: list[dict[str, Any]]
    ) -> DataWrapper[Conditional]:
        """
        Applies the ControlNet model to the given conditional.

        Args:
            controlnet_paths (list[str]): The paths to the ControlNet models.
            cond (Conditional): The conditional to apply the ControlNet to.
            cnet_attrs (list[dict[str, Any]]): The list of ControlNet attributes to set.

        Returns:
            DataWrapper: A DataWrapper containing the updated conditional with the ControlNet applied.
        """
        if len(controlnet_paths) != len(cnet_attrs):
            raise ValueError("Number of controlnet paths must match number of condition hints.")

        vae: VAE = ApplyControlNet._get_current_model(SkipLayers).vae
        controlnet_paths = copy.deepcopy(controlnet_paths)
        cnet_attrs = copy.deepcopy(cnet_attrs)
        controlnet: ControlNet
        controlnet_full_path = folder_paths.get_full_path_or_raise("controlnet", controlnet_paths.pop(0))
        if ApplyControlNet.CNET_CACHE is not None:
            controlnet = copy.copy(ApplyControlNet.CNET_CACHE)
            for k, v in ((k, v) for k, v in vars(controlnet).items() if isinstance(v, torch.Tensor)):
                setattr(controlnet, k, v.clone())
        else:
            controlnet = load_controlnet(controlnet_full_path)
            # ApplyControlNet.CNET_CACHE = controlnet
            controlnet.vae = vae
            for k, v in cnet_attrs.pop(0).items():
                setattr(controlnet, k, v)
        first_controlnet = controlnet
        for path in controlnet_paths:
            controlnet_full_path = folder_paths.get_full_path_or_raise("controlnet", path)
            next_controlnet: ControlNet
            if ApplyControlNet.CNET_CACHE is not None:
                next_controlnet = copy.copy(controlnet.previous_controlnet)
                for k, v in ((k, v) for k, v in vars(next_controlnet).items() if isinstance(v, torch.Tensor)):
                    setattr(next_controlnet, k, v.clone())
            else:
                next_controlnet = load_controlnet(controlnet_full_path)
                next_controlnet.vae = vae
            for k, v in cnet_attrs.pop(0).items():
                setattr(next_controlnet, k, v)
            controlnet.previous_controlnet = next_controlnet
            controlnet = next_controlnet
        cond[0][1]["control"] = first_controlnet
        # ApplyControlNet.CNET_CACHE = copy.copy(first_controlnet)
        return cond

    # pylint: disable=W0221
    def _process_impl(
        self,
        cond_pos: Union[DataWrapper[Conditional], Conditional],
        cond_neg: Union[DataWrapper[Conditional], Conditional],
        model: SkipLayers,
    ) -> tuple[DataWrapper[Conditional], DataWrapper[Conditional]]:
        """
        Applies the ControlNet model to the given positive and negative conditionals.

        Args:
            cond_pos (Union[DataWrapper[Any], Any]): The positive conditional.
            cond_neg (Union[DataWrapper[Any], Any]): The negative conditional.
            model (SkipLayers): The model to use for processing.

        Returns:
            tuple[DataWrapper[Any], DataWrapper[Any]]: The updated conditionals wrapped in DataWrapper instances.
        """
        vae: VAE = model.vae
        assert type(cond_pos) is type(cond_neg), "cond_pos and cond_neg must be of the same type"
        control_nets: list[str] = [self.target.controlnet_path]
        cnet_attrs_pos: list[torch.Tensor] = []
        cnet_attrs_neg: list[torch.Tensor] = []
        if isinstance(cond_pos, DataWrapper):
            assert Counter(cond_pos.args.keys()) == Counter(cond_neg.args.keys()), "Mismatched conditionals keys"
            if "controlnet_paths" in cond_pos.args:
                control_nets.extend(cond_pos.args["controlnet_paths"])
                cnet_attrs_pos.extend(cond_pos.args["cnet_attrs"])
                cnet_attrs_neg.extend(cond_neg.args["cnet_attrs"])
            cond_pos.skip_unwrap = False
            cond_pos = cond_pos.get()
            cond_neg.skip_unwrap = False
            cond_neg = cond_neg.get()

        image_tensor: torch.Tensor = self.target.tensor()

        self._save_tensor(image_tensor, self.target.key())

        logging.info("Loading ControlNet...")
        controlnet_full_path = folder_paths.get_full_path_or_raise("controlnet", self.target.controlnet_path)
        controlnet: ControlNet = load_controlnet(controlnet_full_path)

        conds = ControlNetApplyAdvanced().apply_controlnet(
            cond_pos,
            cond_neg,
            controlnet,
            image_tensor,
            self._strength,
            self._start_percentage,
            self._end_percentage,
            vae,
        )

        main_compare = compare_instance_attributes(conds[0][0][1]["control"], conds[1][0][1]["control"])
        similar_main = {k: v for k, v in main_compare.items() if v[2]}
        diffs_main = {k: v for k, v in main_compare.items() if not v[2]}
        if conds[0][0][1]["control"].previous_controlnet and conds[1][0][1]["control"].previous_controlnet:
            main_prev = compare_instance_attributes(
                conds[0][0][1]["control"].previous_controlnet, conds[1][0][1]["control"].previous_controlnet
            )
            similar_prev = {k: v for k, v in main_prev.items() if v[2]}
            diffs_prev = {k: v for k, v in main_prev.items() if not v[2]}
            logging.info(
                "ControlNet previous_controlnet similar attributes (%s): %s",
                len(similar_prev),
                similar_prev.keys(),
            )
        logging.info(
            "ControlNet main similar attributes (%s): %s",
            len(similar_main),
            similar_main.keys(),
        )

        conds = ((conds[0], cnet_attrs_pos), (conds[1], cnet_attrs_neg))

        # Note: Don't delete controlnet here - it's copied into conds and
        # will be managed by ComfyUI's memory system via load_models_gpu()
        del image_tensor

        # delete controlnet from conds

        def update_cond(cond: Any, cnet_attrs: list[dict[str, Any]]) -> DataWrapper[Any]:
            """
            Updates the conditional by removing the controlnet reference and wrapping it.

            Args:
                cond (Any): The conditional to update.
                cnet_attrs (list[dict[str, Any]]): The list of ControlNet attributes to set.

            Returns:
                DataWrapper[Conditional]: A DataWrapper containing the updated conditional.
            """
            cnet: ControlNet = cond[0][1]["control"]
            # Remove VAE and previous_controlnet references to avoid pickling issues
            cnet.vae = None
            controlnet.vae = None
            cnet.previous_controlnet = None
            controlnet.previous_controlnet = None

            compare = compare_instance_attributes(controlnet, cnet)
            diffs = {k: v for k, v in compare.items() if not v[2]}
            attrs: dict[str, Any] = {}
            for k in diffs:
                attrs[k] = getattr(cnet, k)
            try:
                data: bytes = pickle.dumps(attrs)
                size_kb = len(data) / 1024
                logging.info(
                    "ControlNet attributes differences (%s): %s (size: %s KB)",
                    len(diffs),
                    diffs.keys(),
                    f"{size_kb:.2f}",
                )
            except Exception as e:
                logging.exception("Failed to pickle ControlNet attributes differences.")
                raise e
            cnet_attrs.insert(0, attrs)
            del cond[0][1]["control"]
            return DataWrapper(
                self._data_wrapper_call,
                args={
                    "controlnet_paths": control_nets,
                    "cond": cond,
                    "cnet_attrs": cnet_attrs,
                },
                skip_unwrap=True,
            )

        res = tuple(update_cond(*c) for c in conds)

        return res

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(
        self,
        strength: float,
        start_percentage: float,
        end_percentage: float,
        target: Tuple[Type[ControlNetImgPreprocessor], dict[str, Any]],
    ) -> None:
        target_cls, target_args = target
        assert "skip" in target_args, "target_args must include 'skip' key"
        self._strength = strength
        self._start_percentage = start_percentage
        self._end_percentage = end_percentage
        self._target = target_cls(**target_args) if not target_args.get("skip", False) else None

    def __init__(
        self,
        strength: float,
        start_percentage: float,
        end_percentage: float,
        target: Tuple[Type[ControlNetImgPreprocessor], dict[str, Any]],
    ):
        super().__init__()
        self.update(
            strength=strength,
            start_percentage=start_percentage,
            end_percentage=end_percentage,
            target=target,
        )


def compare_instance_attributes(instance_a: Any, instance_b: Any) -> dict[str, tuple[type, Any, bool]]:
    """
    Compare attributes (non-recursive) of two instances of the same class.

    Returns:
        dict[str, tuple[type, Any, bool]]: Mapping of attribute name to a tuple of:
            (attribute type, value from instance_b, equality flag).
    """
    if type(instance_a) is not type(instance_b):
        raise ValueError("Both instances must be of the same class.")

    attrs_a = vars(instance_a)
    attrs_b = vars(instance_b)

    result: dict[str, tuple[type, Any, bool]] = {}
    for attr in sorted(set(attrs_a.keys()) | set(attrs_b.keys())):
        val_a = attrs_a.get(attr, None)
        val_b = attrs_b.get(attr, None)
        attr_type_a = type(val_a) if attr in attrs_a else type(val_b)
        attr_type_b = type(val_b) if attr in attrs_b else type(val_a)
        bol: bool
        if attr_type_a is attr_type_b and attr_type_a is torch.Tensor:
            bol = (val_a == val_b).all()
        else:
            bol = val_a == val_b
        result[attr] = (attr_type_a, attr_type_b, bol)

    return result


class OpenPosePose(ControlNetImgPreprocessor[torch.Tensor, "OpenPosePose"]):
    """A class representing OpenPose pose settings."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "OpenPosePose"]]:
        return []  # No class params needed for Prompts

    @classmethod
    def key(cls) -> str:
        """Returns the key for the OpenPosePose."""
        return "openpose_pose"

    @property
    def controlnet_path(self) -> str:
        """Returns the ControlNet path."""
        return self._controlnet_path

    def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
        """Processes the image tensor using OpenPose preprocessor."""
        # Free memory before loading OpenPose detector
        soft_empty_cache()

        # Initialize OpenPose Detector
        openpose_model: OpenposeDetector = OpenposeDetector.from_pretrained().to(get_torch_device())

        # Run preprocessor
        result = aux_utils.common_annotator_call(
            lambda image, **kwargs: openpose_model(image, **kwargs)[0],  # noqa: F821
            cnet_img,
            include_hand=self._detect_hands,
            include_face=self._detect_face,
            include_body=self._detect_body,
            image_and_json=True,
            xinsr_stick_scaling=self._scale_stick_for_xinsr_cn,
            resolution=self._resolution,
        )

        # Clean up OpenPose model after use
        del openpose_model
        soft_empty_cache()

        return result

    # pylint: disable=W0221, W0201
    def _update_impl(  # type: ignore[override]
        self,
        image_name: str,
        detect_body: bool,
        detect_hands: bool,
        detect_face: bool,
        scale_stick_for_xinsr_cn: bool,
        resolution: int,
        controlnet_path: str,
        skip: bool,
    ) -> None:
        super()._update_impl(image_name, skip)
        self._detect_body = detect_body
        self._detect_hands = detect_hands
        self._detect_face = detect_face
        self._scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn
        self._resolution = resolution
        self._controlnet_path = controlnet_path

    def __init__(
        self,
        image_name: str,
        detect_body: bool,
        detect_hands: bool,
        detect_face: bool,
        scale_stick_for_xinsr_cn: bool,
        resolution: int,
        controlnet_path: str,
        skip: bool,
    ):
        super().__init__(image_name, skip)
        self.update(
            image_name=image_name,
            detect_body=detect_body,
            detect_hands=detect_hands,
            detect_face=detect_face,
            scale_stick_for_xinsr_cn=scale_stick_for_xinsr_cn,
            resolution=resolution,
            controlnet_path=controlnet_path,
            skip=skip,
        )


class CannyEdge(ControlNetImgPreprocessor[torch.Tensor, "CannyEdge"]):
    """A class representing Canny edge detector settings."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "CannyEdge"]]:
        return []  # No class params needed for Prompts

    @classmethod
    def key(cls) -> str:
        """Returns the key for the CannyEdge."""
        return "canny_edge"

    @property
    def controlnet_path(self) -> str:
        """Returns the ControlNet path."""
        return self._controlnet_path

    def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
        """Processes the image tensor using Canny edge detector."""
        cnet_height, cnet_width = cnet_img.shape[1], cnet_img.shape[2]

        res = aux_utils.common_annotator_call(
            CannyDetector(),
            cnet_img,
            low_threshold=self._low_threshold,
            high_threshold=self._high_threshold,
            resolution=self._resolution,
        )
        # Resize to match ControlNet input size if needed
        if (res.shape[2], res.shape[3]) != (cnet_height, cnet_width):
            res = ResizeAndPadImage().resize_and_pad(
                res,
                cnet_width,
                cnet_height,
                "white",
                "lanczos",
            )[0]
        return res

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(  # type: ignore[override]
        self,
        image_name: str,
        low_threshold: int,
        high_threshold: int,
        resolution: int,
        controlnet_path: str,
        skip: bool,
    ) -> None:
        super()._update_impl(image_name, skip)
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._resolution = resolution
        self._controlnet_path = controlnet_path

    def __init__(
        self,
        image_name: str,
        low_threshold: int,
        high_threshold: int,
        resolution: int,
        controlnet_path: str,
        skip: bool,
    ) -> None:
        super().__init__(image_name, skip)
        self.update(
            image_name=image_name,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            resolution=resolution,
            controlnet_path=controlnet_path,
            skip=skip,
        )
