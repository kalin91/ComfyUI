"""Mimic classes for various components."""

import os
from typing import Any
import torch
import numpy as np
import comfy.model_management
from custom_nodes.comfyui_controlnet_aux import utils as aux_utils
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector
import folder_paths
import node_helpers
from PIL import Image, ImageOps, ImageSequence


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
    def cfg(self) -> float:
        """Returns the CFG scale."""
        return self._cfg

    @property
    def sampler_name(self) -> str:
        """Returns the name of the sampler."""
        return self._sampler_name

    @property
    def scheduler(self) -> str:
        """Returns the name of the scheduler."""
        return self._scheduler

    @property
    def denoise(self) -> float:
        """Returns the denoise value."""
        return self._denoise

    def __init__(self, seed: int, steps: int, cfg: float, sampler_name: str, scheduler: str, denoise: float):
        self._seed = seed
        self._steps = steps
        self._cfg = cfg
        self._sampler_name = sampler_name
        self._scheduler = scheduler
        self._denoise = denoise

    def to_dict(self) -> dict:
        """Converts the SimpleKSampler instance to a dictionary."""
        print(
            f"Sampling 1 with seed={self._seed}, steps={self._steps}, cfg={self._cfg}, sampler={self._sampler_name}, scheduler={self._scheduler}..."
        )

        return {
            "seed": self._seed,
            "steps": self._steps,
            "cfg": self._cfg,
            "sampler_name": self._sampler_name,
            "scheduler": self._scheduler,
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
        print(f"Creating empty latent {self._width}x{self._height}...")

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
    def controlnet_path(self) -> str:
        """Returns the ControlNet path."""
        return self._controlnet_path

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

    def conditionals(self, cond_pos: Any, cond_neg: Any, pose_image_tensor: torch.Tensor, vae: Any) -> tuple[Any, Any]:
        """Returns placeholder conditionals."""

        print("Loading ControlNet...")
        controlnet_full_path = folder_paths.get_full_path_or_raise("controlnet", self.controlnet_path)
        controlnet = comfy.controlnet.load_controlnet(controlnet_full_path)

        if self._strength == 0:
            cond_pos_cnet = cond_pos
            cond_neg_cnet = cond_neg
        else:
            control_hint = pose_image_tensor.movedim(-1, 1)
            cnets = {}

            out = []
            for conditioning in [cond_pos, cond_neg]:
                c = []
                for t in conditioning:
                    d = t[1].copy()

                    prev_cnet = d.get("control", None)
                    if prev_cnet in cnets:
                        c_net = cnets[prev_cnet]
                    else:
                        c_net = controlnet.copy().set_cond_hint(
                            control_hint, self.strength, (self.start_percentage, self.end_percentage), vae=vae
                        )
                        c_net.set_previous_controlnet(prev_cnet)
                        cnets[prev_cnet] = c_net

                    d["control"] = c_net
                    d["control_apply_to_uncond"] = False
                    n = [t[0], d]
                    c.append(n)
                out.append(c)
            cond_pos_cnet, cond_neg_cnet = out[0], out[1]
        return cond_pos_cnet, cond_neg_cnet

    def __init__(self, strength: float, start_percentage: float, end_percentage: float, controlnet_path: str):
        self._strength = strength
        self._start_percentage = start_percentage
        self._end_percentage = end_percentage
        self._controlnet_path = controlnet_path


class OpenPosePose:
    """A class representing OpenPose pose settings."""

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
    def scale_stick_for_xinsr_cn(self) -> bool:
        """Returns whether OpenPose scale stick for xinsr_cn is enabled."""
        return self._scale_stick_for_xinsr_cn

    @property
    def resolution(self) -> int:
        """Returns the OpenPose resolution."""
        return self._resolution

    def tensor(self) -> torch.Tensor:
        """Processes the image tensor using OpenPose preprocessor."""
        # Initialize OpenPose Detector
        openpose_model = OpenposeDetector.from_pretrained().to(comfy.model_management.get_torch_device())

        # Run preprocessor
        return aux_utils.common_annotator_call(
            lambda image, **kwargs: openpose_model(image, **kwargs)[0],
            self._openpose_image,
            include_hand=self.detect_hands,
            include_face=self.detect_face,
            include_body=self.detect_body,
            image_and_json=True,
            xinsr_stick_scaling=self.scale_stick_for_xinsr_cn,
            resolution=self.resolution,
        )

    def __init__(
        self,
        image_name: str,
        detect_body: bool,
        detect_hands: bool,
        detect_face: bool,
        scale_stick_for_xinsr_cn: bool,
        resolution: int,
    ):
        self._detect_body = detect_body
        self._detect_hands = detect_hands
        self._detect_face = detect_face
        self._scale_stick_for_xinsr_cn = scale_stick_for_xinsr_cn
        self._resolution = resolution

        print("Loading OpenPose Image...")
        input_folder = folder_paths.get_input_directory()
        image_path = os.path.join(input_folder, image_name)
        img = node_helpers.pillow(Image.open, image_path)

        # Process image to tensor (similar to LoadImage node)
        output_images = []
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)

        if len(output_images) > 1:
            # If multiple frames, stack them? For now assume single image as per workflow
            img_tensor = torch.cat(output_images, dim=0)
        else:
            img_tensor = output_images[0]

        self._openpose_image = img_tensor


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

#    @property
#    def refiner_ratio(self) -> float:
#        """Returns the refiner ratio."""
#        return self._refiner_ratio

#    @property
#    def batch_size(self) -> int:
#        """Returns the batch size."""
#        return self._batch_size

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
        base_dict.update(
            {
                "guide_size": self._guide_size,
                "guide_size_for": self._guide_size_for,
                "max_size": self._max_size,
                "feather": self._feather,
                "noise_mask": self._noise_mask,
                "force_inpaint": self._force_inpaint,
                "drop_size": self._drop_size,
               # "refiner_ratio": self._refiner_ratio,
               # "batch_size": self._batch_size,
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
            }
        )
        return base_dict

    def __init__(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        guide_size: int,
        guide_size_for: bool,
        max_size: int,
        feather: int,
        noise_mask: bool,
        force_inpaint: bool,
        drop_size: int,
       # refiner_ratio: float,
       # batch_size: int,
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
        super().__init__(seed, steps, cfg, sampler_name, scheduler, denoise)
        self._guide_size = guide_size
        self._guide_size_for = guide_size_for
        self._max_size = max_size
        self._feather = feather
        self._noise_mask = noise_mask
        self._force_inpaint = force_inpaint
        self._drop_size = drop_size
      #  self._refiner_ratio = refiner_ratio
      #  self._batch_size = batch_size
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
