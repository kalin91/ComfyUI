"""Mimic classes for various components."""

import torch
import comfy.model_management
from custom_nodes.comfyui_controlnet_aux import utils as aux_utils
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector


class SimpleKSampler:
    """A simple KSampler class for demonstration purposes."""

    @property
    def seed(self) -> int:
        """Returns the seed value."""
        return self._seed

    @property
    def steps(self) -> int:
        """Returns the number of steps."""
        return self._steps

    @property
    def cfg_scale(self) -> float:
        """Returns the CFG scale."""
        return self._cfg_scale

    @property
    def sampler_name(self) -> str:
        """Returns the name of the sampler."""
        return self._sampler_name

    @property
    def scheduler_name(self) -> str:
        """Returns the name of the scheduler."""
        return self._scheduler_name

    @property
    def denoise(self) -> float:
        """Returns the denoise value."""
        return self._denoise

    def __init__(
        self, seed: int, steps: int, cfg_scale: float, sampler_name: str, scheduler_name: str, denoise: float
    ):
        self._seed = seed
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._sampler_name = sampler_name
        self._scheduler_name = scheduler_name
        self._denoise = denoise

    def to_dict(self) -> dict:
        """Converts the SimpleKSampler instance to a dictionary."""
        return {
            "seed": self._seed,
            "steps": self._steps,
            "cfg": self._cfg_scale,
            "sampler_name": self._sampler_name,
            "scheduler": self._scheduler_name,
            "denoise": self._denoise,
        }


class EmptyLatent:
    """An empty latent class for placeholder purposes."""

    @property
    def width(self) -> int:
        """Returns the width of the latent."""
        return self._width

    @property
    def height(self) -> int:
        """Returns the height of the latent."""
        return self._height

    @property
    def batch_size(self) -> int:
        """Returns the batch size of the latent."""
        return self._batch_size

    @property
    def latent(self) -> torch.Tensor:
        """Generates and returns an empty latent tensor."""
        return torch.zeros(
            [self._batch_size, 16, self._height // 8, self._width // 8],
            device=comfy.model_management.intermediate_device(),
        )

    def __init__(self, width: int, height: int, batch_size: int):
        self._width = width
        self._height = height
        self._batch_size = batch_size


class ApplyControlNet:
    """Returns the ControlNet application parameters."""

    @property
    def strength(self) -> float:
        """Returns the strength value."""
        return self._strength

    @property
    def start_percentage(self) -> float:
        """Returns the start percentage value."""
        return self._start_percentage

    @property
    def end_percentage(self) -> float:
        """Returns the end percentage value."""
        return self._end_percentage

    def __init__(self, strength: float, start_percentage: float, end_percentage: float):
        self._strength = strength
        self._start_percentage = start_percentage
        self._end_percentage = end_percentage


class OpenPosePose:
    """A class representing OpenPose pose settings."""

    @property
    def openpose_image(self) -> str:
        """Returns the OpenPose image."""
        return self._openpose_image

    @property
    def detect_body(self) -> bool:
        """Returns whether body detection is enabled."""
        return self._detect_body

    @property
    def detect_hands(self) -> bool:
        """Returns whether hand detection is enabled."""
        return self._detect_hands

    @property
    def detect_face(self) -> bool:
        """Returns whether face detection is enabled."""
        return self._detect_face

    @property
    def openpose_scale_stick_for_xinsr_cn(self) -> bool:
        """Returns whether OpenPose scale stick for xinsr_cn is enabled."""
        return self._openpose_scale_stick_for_xinsr_cn

    @property
    def openpose_resolution(self) -> int:
        """Returns the OpenPose resolution."""
        return self._openpose_resolution

    def tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Processes the image tensor using OpenPose preprocessor."""
        # Initialize OpenPose Detector
        openpose_model = OpenposeDetector.from_pretrained().to(comfy.model_management.get_torch_device())

        def openpose_func(image, **kwargs):
            """Processes the image using OpenPose model."""
            pose_img, _ = openpose_model(image, **kwargs)
            return pose_img

        # Run preprocessor
        return aux_utils.common_annotator_call(
            openpose_func,
            img_tensor,
            include_hand=self.detect_hands,
            include_face=self.detect_face,
            include_body=self.detect_body,
            image_and_json=True,
            xinsr_stick_scaling=self.openpose_scale_stick_for_xinsr_cn,
            resolution=self.openpose_resolution,
        )

    def __init__(
        self,
        openpose_image: str,
        detect_body: bool,
        detect_hands: bool,
        detect_face: bool,
        openpose_scale_stick_for_xinsr_cn: bool,
        openpose_resolution: int,
    ):
        self._openpose_image = openpose_image
        self._detect_body = detect_body
        self._detect_hands = detect_hands
        self._detect_face = detect_face
        self._openpose_scale_stick_for_xinsr_cn = openpose_scale_stick_for_xinsr_cn
        self._openpose_resolution = openpose_resolution


class FaceDetailer(SimpleKSampler):
    """A class representing face detailer settings."""

    @property
    def guide_size(self) -> int:
        """Returns the guide size."""
        return self._guide_size

    @property
    def guide_size_for(self) -> bool:
        """Returns whether to use guide size for."""
        return self._guide_size_for

    @property
    def max_size(self) -> int:
        """Returns the maximum size."""
        return self._max_size

    @property
    def feather(self) -> int:
        """Returns the feather value."""
        return self._feather

    @property
    def noise_mask(self) -> bool:
        """Returns whether noise mask is enabled."""
        return self._noise_mask

    @property
    def force_inpaint(self) -> bool:
        """Returns whether force inpaint is enabled."""
        return self._force_inpaint

    @property
    def drop_size(self) -> int:
        """Returns the drop size."""
        return self._drop_size

    @property
    def refiner_ratio(self) -> float:
        """Returns the refiner ratio."""
        return self._refiner_ratio

    @property
    def batch_size(self) -> int:
        """Returns the batch size."""
        return self._batch_size

    @property
    def cycle(self) -> int:
        """Returns the cycle count."""
        return self._cycle

    @property
    def bbox_threshold(self) -> float:
        """Returns the bounding box threshold."""
        return self._bbox_threshold

    @property
    def bbox_dilation(self) -> int:
        """Returns the bounding box dilation."""
        return self._bbox_dilation

    @property
    def bbox_crop_factor(self) -> float:
        """Returns the bounding box crop factor."""
        return self._bbox_crop_factor

    @property
    def sam_detection_hint(self) -> str:
        """Returns the SAM detection hint."""
        return self._sam_detection_hint

    @property
    def sam_dilation(self) -> int:
        """Returns the SAM dilation."""
        return self._sam_dilation

    @property
    def sam_threshold(self) -> float:
        """Returns the SAM threshold."""
        return self._sam_threshold

    @property
    def sam_bbox_expansion(self) -> int:
        """Returns the SAM bounding box expansion."""
        return self._sam_bbox_expansion

    @property
    def sam_mask_hint_threshold(self) -> float:
        """Returns the SAM mask hint threshold."""
        return self._sam_mask_hint_threshold

    @property
    def sam_mask_hint_use_negative(self) -> str:
        """Returns whether to use negative SAM mask hint."""
        return self._sam_mask_hint_use_negative

    @property
    def wildcard(self) -> str:
        """Returns the wildcard."""
        return self._wildcard
    
    def to_dict(self) -> dict:
        """Converts the FaceDetailer instance to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update({
            "guide_size": self._guide_size,
            "guide_size_for": self._guide_size_for,
            "max_size": self._max_size,
            "feather": self._feather,
            "noise_mask": self._noise_mask,
            "force_inpaint": self._force_inpaint,
            "drop_size": self._drop_size,
            "refiner_ratio": self._refiner_ratio,
            "batch_size": self._batch_size,
            "cycle": self._cycle,
            "bbox_threshold": self._bbox_threshold,
            "bbox_dilation": self._bbox_dilation,
            "bbox_crop_factor": self._bbox_crop_factor,
            "sam_detection_hint": self._sam_detection_hint,
            "sam_dilation": self._sam_dilation,
            "sam_threshold": self._sam_threshold,
            "sam_bbox_expansion": self._sam_bbox_expansion,
            "sam_mask_hint_threshold": self._sam_mask_hint_threshold,
            "sam_mask_hint_use_negative": self._sam_mask_hint_use_negative,
            "wildcard": self._wildcard,
        })
        return base_dict

    def __init__(
        self,
        seed: int,
        steps: int,
        cfg_scale: float,
        sampler_name: str,
        scheduler_name: str,
        denoise: float,
        guide_size: int,
        guide_size_for: bool,
        max_size: int,
        feather: int,
        noise_mask: bool,
        force_inpaint: bool,
        drop_size: int,
        refiner_ratio: float,
        batch_size: int,
        cycle: int,
        bbox_threshold: float,
        bbox_dilation: int,
        bbox_crop_factor: float,
        sam_detection_hint: str,
        sam_dilation: int,
        sam_threshold: float,
        sam_bbox_expansion: int,
        sam_mask_hint_threshold: float,
        sam_mask_hint_use_negative: str,
        wildcard: str,
    ):
        super().__init__(seed, steps, cfg_scale, sampler_name, scheduler_name, denoise)
        self._guide_size = guide_size
        self._guide_size_for = guide_size_for
        self._max_size = max_size
        self._feather = feather
        self._noise_mask = noise_mask
        self._force_inpaint = force_inpaint
        self._drop_size = drop_size
        self._refiner_ratio = refiner_ratio
        self._batch_size = batch_size
        self._cycle = cycle
        self._bbox_threshold = bbox_threshold
        self._bbox_dilation = bbox_dilation
        self._bbox_crop_factor = bbox_crop_factor
        self._sam_detection_hint = sam_detection_hint
        self._sam_dilation = sam_dilation
        self._sam_threshold = sam_threshold
        self._sam_bbox_expansion = sam_bbox_expansion
        self._sam_mask_hint_threshold = sam_mask_hint_threshold
        self._sam_mask_hint_use_negative = sam_mask_hint_use_negative
        self._wildcard = wildcard
