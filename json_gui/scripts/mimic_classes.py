"""Mimic child classes for various components."""

import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, cast
import torch
import numpy as np
from segment_anything.build_sam import Sam
import comfy.model_management
from comfy_extras.nodes_sd3 import SkipLayerGuidanceSD3
from comfy_extras.nodes_images import ResizeAndPadImage
from comfy_extras.nodes_mask import MaskToImage
from comfy.sample import fix_empty_latent_channels, prepare_noise, sample
from comfy.sd import load_checkpoint_guess_config, VAE, CLIP
from comfy.model_patcher import ModelPatcher
from comfy.controlnet import load_controlnet
from custom_nodes.comfyui_controlnet_aux import utils as aux_utils
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.canny import CannyDetector
from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import UltralyticsDetectorProvider
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import SAMLoader, FaceDetailer
from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import subcore
from nodes import ControlNetApplyAdvanced
import folder_paths
from PIL import Image
from json_gui.scripts.mimic import MimicNode


class SkipLayers(MimicNode):
    """A class representing skip layer guidance settings."""

    @classmethod
    def key(cls) -> str:
        """Returns the key for the SkipLayers."""
        return "skip_layers_model"

    CHECKPOINT_PATH = "sd3.5_medium.safetensors"

    def get_model(self, use_tuned: bool) -> ModelPatcher:
        """Returns the model based on whether to use the tuned version."""
        return self._process_impl(use_tuned)

    # pylint: disable=W0221
    def _process_impl(self, use_tuned: bool) -> ModelPatcher:
        """Returns the tuned model."""
        return self._tunned_model if use_tuned else self._base_model

    @property
    def vae(self) -> VAE:
        """Returns the VAE."""
        return self._vae

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, layers: list[int], scale: float, start_percent: float, end_percent: float) -> None:
        # 1. Load Model and VAE
        logging.info("Loading Checkpoint...")

        # Free memory before loading the large checkpoint (~10.5GB)
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", self.CHECKPOINT_PATH)
        self._base_model, _a, self._vae, _b = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=False,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        self._layers = ",".join(str(layer) for layer in layers)
        self._scale = scale
        self._start_percent = start_percent
        self._end_percent = end_percent

        if self._layers:
            result: tuple = SkipLayerGuidanceSD3.execute(
                self._base_model,
                self._layers,
                self._scale,
                self._start_percent,
                self._end_percent,
            )
            self._tunned_model = result[0]
        else:
            self._tunned_model = self._base_model

    def __init__(self, layers: list[int], scale: float, start_percent: float, end_percent: float):
        super().__init__()
        self.update(layers=layers, scale=scale, start_percent=start_percent, end_percent=end_percent)


class ControlNetImgPreprocessor(MimicNode, ABC):
    """Abstract base class for ControlNet image preprocessors."""

    def __new__(cls, image_name: str, skip: bool, **kwargs) -> "ControlNetImgPreprocessor":
        if skip:
            logging.info("Skipping ControlNet image preprocessor for %s", image_name)
            return None  # type: ignore
        instance = super(ControlNetImgPreprocessor, cls).__new__(cls)
        return instance

    @property
    @abstractmethod
    def controlnet_path(self) -> str:
        """Returns the ControlNet path."""

    @property
    def skip(self) -> bool:
        """Returns whether to skip this preprocessor."""
        return self._skip

    # pylint: disable=W0221
    def _process_impl(self) -> Any:
        """Processes the image and returns a tensor."""
        res = self._tensor_impl(self._controlnet_img)
        if self._save_tensor:
            self._save_tensor(res)
        return res

    @abstractmethod
    def _tensor_impl(self, cnet_img: torch.Tensor) -> torch.Tensor:
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


class SimpleKSampler(MimicNode):
    """A simple KSampler class for demonstration purposes."""

    @classmethod
    def key(cls) -> str:
        """Returns the key for the SimpleKSampler."""
        return "simple_k_sampler"

    @property
    def use_tune(self) -> bool:
        """Returns whether to use tune."""
        return self._use_tune

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        use_tune: bool,
    ) -> None:
        self._seed = seed
        self._steps = steps
        self._cfg = cfg
        self._sampler_name = sampler_name
        self._scheduler = scheduler
        self._denoise = denoise
        self._use_tune = use_tune

    def __init__(
        self,
        seed: int,
        steps: int,
        cfg: float,
        sampler_name: str,
        scheduler: str,
        denoise: float,
        use_tune: bool,
    ):
        super().__init__()
        if type(self) is SimpleKSampler:  # pylint: disable=C0123
            self.update(
                seed=seed,
                steps=steps,
                cfg=cfg,
                sampler_name=sampler_name,
                scheduler=scheduler,
                denoise=denoise,
                use_tune=use_tune,
            )

    def _to_dict(self) -> dict:
        """Converts the SimpleKSampler instance to a dictionary."""
        logging.info(
            "Sampling 1 with seed=%s, steps=%s, cfg=%s, sampler=%s, scheduler=%s...",
            self._seed,
            self._steps,
            self._cfg,
            self._sampler_name,
            self._scheduler,
        )

        return {
            "seed": self._seed,
            "steps": self._steps,
            "cfg": self._cfg,
            "sampler_name": self._sampler_name,
            "scheduler": self._scheduler,
            "denoise": self._denoise,
        }

    # pylint: disable=W0221
    def _process_impl(
        self, latent_image: torch.Tensor, model: ModelPatcher, cond_pos_cnet: Any, cond_neg_cnet: Any
    ) -> torch.Tensor:
        """A placeholder method to simulate processing."""

        # Prepare noise
        noisy_latent_image = fix_empty_latent_channels(model, latent_image)

        noise = prepare_noise(noisy_latent_image, self._seed, None)

        # Safely get sampler arguments
        sampler_arguments = self._to_dict()

        sampler_arguments.update(
            {
                "model": model,
                "noise": noise,
                "positive": cond_pos_cnet,
                "negative": cond_neg_cnet,
                "latent_image": noisy_latent_image,
                "disable_noise": False,
                "start_step": None,
                "last_step": None,
                "force_full_denoise": False,
                "noise_mask": None,
                "callback": None,
                "disable_pbar": False,
            }
        )

        sampler_signature = inspect.signature(sample)
        for key in sampler_arguments:
            if key not in sampler_signature.parameters:
                raise ValueError(f"Unexpected argument '{key}' for comfy.sample.sample")
        return comfy.sample.sample(**sampler_arguments)


class Prompts(MimicNode):
    """A class representing positive and negative prompts."""

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Prompts."""
        return "prompts"

    @property
    def positive(self) -> str:
        """Returns the positive prompt."""
        return self._positive

    @property
    def negative(self) -> str:
        """Returns the negative prompt."""
        return self._negative

    # pylint: disable=W0221
    def _process_impl(
        self, clip: CLIP
    ) -> tuple[list[tuple[torch.Tensor, dict[str, Any]]], list[tuple[torch.Tensor, dict[str, Any]]]]:
        """Encodes the positive and negative prompts using the provided CLIP model."""
        logging.info("Encoding prompts...")
        tokens_pos = clip.tokenize(self.positive)
        cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

        tokens_neg = clip.tokenize(self.negative)
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

        del tokens_pos
        del tokens_neg
        torch.cuda.empty_cache()
        return (cond_pos, cond_neg)

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, positive: str, negative: str) -> None:
        self._positive = positive
        self._negative = negative

    def __init__(self, positive: str, negative: str):
        super().__init__()
        self.update(positive=positive, negative=negative)


class EmptyLatent(MimicNode):
    """An empty latent class for placeholder purposes."""

    @classmethod
    def key(cls) -> str:
        """Returns the key for the EmptyLatent."""
        return "empty_latent"

    @property
    def start_img(self) -> Optional[torch.Tensor]:
        """Returns the starting image tensor, if any."""
        return self._start_img

    # pylint: disable=W0221
    def _process_impl(self, vae: VAE) -> torch.Tensor:
        """Generates and returns an empty latent tensor."""
        if self.start_img is not None:
            logging.info("Creating latent from start image...")

            if vae is None:
                raise ValueError(
                    "VAE is required to encode start_img to latent space. "
                    "Please provide vae parameter when creating EmptyLatent with image_name."
                )

            # Redim the image  to the expected size
            start_img = self.start_img
            current_height, current_width = start_img.shape[1], start_img.shape[2]

            if current_height != self._height or current_width != self._width:
                logging.info(
                    "Resizing start image from %sx%s to %sx%s...",
                    current_width,
                    current_height,
                    self._width,
                    self._height,
                )
                # Permute: [B, H, W, C] -> [B, C, H, W] for interpolation
                start_img = start_img.permute(0, 3, 1, 2)
                start_img = torch.nn.functional.interpolate(
                    start_img, size=(self._height, self._width), mode="bilinear", align_corners=False
                )
                # Permute back: [B, C, H, W] -> [B, H, W, C]
                start_img = start_img.permute(0, 2, 3, 1)

            logging.info("Encoding start image to latent space with VAE...")
            latent = vae.encode(start_img)
            logging.info("Encoded latent shape: %s", latent.shape)

            return latent

        logging.info("Creating empty latent %sx%s...", self._width, self._height)
        return torch.zeros(
            [self._batch_size, 16, self._height // 8, self._width // 8],
            device=comfy.model_management.intermediate_device(),
        )

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, width: int, height: int, batch_size: int, image_name: Optional[str]) -> None:
        self._width = width
        self._height = height
        self._batch_size = batch_size
        self._start_img = self._upload_image(image_name) if image_name and image_name != "<None>" else None

    def __init__(self, width: int, height: int, batch_size: int, image_name: str):
        super().__init__()
        self.update(width=width, height=height, batch_size=batch_size, image_name=image_name)


class ApplyControlNet(MimicNode):
    """Returns the ControlNet application parameters."""

    @classmethod
    def key(cls) -> str:
        """Returns the key for the ApplyControlNet."""
        return "apply_control_net"

    @property
    def target(self) -> ControlNetImgPreprocessor:
        """Returns the target ControlNet image preprocessor."""
        return self._target

    @SkipLayers.use_class_param(lambda inst: {"vae": cast(SkipLayers, inst).vae})
    # pylint: disable=W0221
    def _process_impl(self, cond_pos: Any, cond_neg: Any, vae: Any) -> tuple[Any, Any]:
        """Returns placeholder conditionals."""

        image_tensor: torch.Tensor = self._target.tensor()

        logging.info("Loading ControlNet...")
        controlnet_full_path = folder_paths.get_full_path_or_raise("controlnet", self._target.controlnet_path)
        controlnet = load_controlnet(controlnet_full_path)

        res = ControlNetApplyAdvanced().apply_controlnet(
            cond_pos,
            cond_neg,
            controlnet,
            image_tensor,
            self._strength,
            self._start_percentage,
            self._end_percentage,
            vae,
        )

        # Note: Don't delete controlnet here - it's copied into conds and
        # will be managed by ComfyUI's memory system via load_models_gpu()
        del image_tensor
        comfy.model_management.soft_empty_cache()

        return res

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(
        self, strength: float, start_percentage: float, end_percentage: float, target: ControlNetImgPreprocessor
    ) -> None:
        self._strength = strength
        self._start_percentage = start_percentage
        self._end_percentage = end_percentage
        self._target = target

    def __init__(
        self, strength: float, start_percentage: float, end_percentage: float, target: ControlNetImgPreprocessor
    ):
        super().__init__()
        self.update(
            strength=strength,
            start_percentage=start_percentage,
            end_percentage=end_percentage,
            target=target,
        )


class OpenPosePose(ControlNetImgPreprocessor):
    """A class representing OpenPose pose settings."""

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
        comfy.model_management.soft_empty_cache()

        # Initialize OpenPose Detector
        openpose_model: OpenposeDetector = OpenposeDetector.from_pretrained().to(
            comfy.model_management.get_torch_device()
        )

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
        comfy.model_management.soft_empty_cache()

        return result

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(
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


class CannyEdge(ControlNetImgPreprocessor):
    """A class representing Canny edge detector settings."""

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
    def _update_impl(
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


class FaceDetailerNode(SimpleKSampler):
    """A class representing face detailer settings."""

    @classmethod
    def key(cls) -> str:
        return "face_detailer"

    @property
    def sam_model_opt(self) -> Sam:
        """Returns the SAM model option."""
        return self._sam_model_opt

    @property
    def bbox_detector(self) -> subcore.UltraBBoxDetector:
        """Returns the batch size."""
        return self._bbox_detector

    def to_dict(self) -> dict:
        """Converts the FaceDetailer instance to a dictionary."""
        base_dict = super()._to_dict()
        base_dict.update(
            {
                "sam_model_opt": self._sam_model_opt,
                "bbox_detector": self._bbox_detector,
                "guide_size": self._guide_size,
                "guide_size_for": self._guide_size_for,
                "max_size": self._max_size,
                "feather": self._feather,
                "noise_mask": self._noise_mask,
                "force_inpaint": self._force_inpaint,
                "drop_size": self._drop_size,
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

    # pylint: disable=W0221
    def _process_impl(self, input_image: torch.Tensor, **kwargs) -> torch.Tensor:
        """Function to process image once rotated."""

        # 10.5 FaceDetailer
        logging.info("Running FaceDetailer...")

        face_detailer = FaceDetailer()

        # FaceDetailer.doit(image, model, clip, vae, guide_size, guide_size_for, max_size,
        # seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint,
        # bbox_threshold, bbox_dilation, bbox_crop_factor, sam_detection_hint, sam_dilation, sam_threshold,
        # sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, bbox_detector,
        # sam_model_opt, segm_detector_opt, detailer_hook)

        # Note: Arguments might vary slightly depending on version, checking signature would be good.
        # Assuming standard arguments based on common usage.

        face_arguments = self.to_dict()

        face_arguments.update(kwargs)

        face_arguments.update(
            {
                "image": input_image,
                "segm_detector_opt": None,  # Not using segm detector here
                "detailer_hook": None,
            }
        )

        # validate face_arguments keys against FaceDetailer.doit signature would be ideal
        face_signature = inspect.signature(face_detailer.doit)
        for key in face_arguments:
            if key not in face_signature.parameters:
                raise ValueError(f"Unexpected argument '{key}' for FaceDetailer.doit")

        result_images, cropped_images, cropped_alpha, mask = face_detailer.doit(**face_arguments)[:4]
        if self._save_tensor:
            for idx, cropped in enumerate(cropped_images):
                self._save_tensor(cropped, f"face-cropped-{idx}")
            for idx, alpha in enumerate(cropped_alpha):
                self._save_tensor(alpha, f"face-alpha-{idx}")
            mask_img_tensor: tuple = MaskToImage().execute(mask).result[0]  # pylint: disable=E1136
            self._save_tensor(mask_img_tensor, "face-mask")
        return result_images

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(
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
        bbox_detector: str,
        sam_model_opt: str,
        wildcard: str,
        use_tune: bool,
    ) -> None:
        super()._update_impl(seed, steps, cfg, sampler_name, scheduler, denoise, use_tune)
        self._guide_size = guide_size
        self._guide_size_for = guide_size_for
        self._max_size = max_size
        self._feather = feather
        self._noise_mask = noise_mask
        self._force_inpaint = force_inpaint
        self._drop_size = drop_size
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

        # Free memory before loading detection models
        comfy.model_management.soft_empty_cache()

        bbox_provider = UltralyticsDetectorProvider()
        # UltralyticsDetectorProvider.doit returns (BBOX_DETECTOR, SEGM_DETECTOR)
        self._bbox_detector, _c = bbox_provider.doit(bbox_detector)

        sam_loader = SAMLoader()
        # SAMLoader.load_model returns (SAM_MODEL,)
        self._sam_model_opt = sam_loader.load_model(sam_model_opt)[0]

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
        bbox_detector: str,
        sam_model_opt: str,
        wildcard: str,
        use_tune: bool,
    ):
        super().__init__(seed, steps, cfg, sampler_name, scheduler, denoise, use_tune)
        self.update(
            seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            denoise=denoise,
            use_tune=use_tune,
            guide_size=guide_size,
            guide_size_for=guide_size_for,
            max_size=max_size,
            feather=feather,
            noise_mask=noise_mask,
            force_inpaint=force_inpaint,
            drop_size=drop_size,
            cycle=cycle,
            bbox_threshold=bbox_threshold,
            bbox_dilation=bbox_dilation,
            bbox_crop_factor=bbox_crop_factor,
            sam_detection_hint=sam_detection_hint,
            sam_dilation=sam_dilation,
            sam_threshold=sam_threshold,
            sam_bbox_expansion=sam_bbox_expansion,
            sam_mask_hint_threshold=sam_mask_hint_threshold,
            sam_mask_hint_use_negative=sam_mask_hint_use_negative,
            bbox_detector=bbox_detector,
            sam_model_opt=sam_model_opt,
            wildcard=wildcard,
        )


class Rotator(MimicNode):
    """A class representing image rotation settings."""

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Rotator."""
        return "rotator"

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, angle: float) -> None:
        self._angle = angle

    def __init__(self, angle: float):
        super().__init__()
        self.update(angle=angle)

    # pylint: disable=W0221
    def _process_impl(self, image: torch.Tensor, func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Rotates the given image tensor by the specified angle.

        Uses PIL with BICUBIC interpolation to minimize quality loss during rotation.
        Converts tensor -> PIL -> rotate -> tensor for better quality preservation.
        """

        if self._angle == 0:
            return func(image)

        # getting original image size (BHWC format)
        batch_size, orig_h, orig_w, _channels = image.shape
        logging.info("Original image shape: %s", image.shape)

        # Process each image in the batch
        rotated_list = []
        for i in range(batch_size):
            # Convert tensor to PIL Image (tensor is 0-1 float, HWC)
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Rotate with PIL using BICUBIC interpolation (expand=True to keep corners)
            rotated_pil = pil_img.rotate(
                -self._angle,  # PIL rotates counter-clockwise, we want clockwise
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

            # Convert back to tensor (0-1 float)
            rotated_np = np.array(rotated_pil).astype(np.float32) / 255.0
            rotated_list.append(torch.from_numpy(rotated_np))

        # Stack batch back together (BHWC)
        rotated_images = torch.stack(rotated_list, dim=0).to(image.device)
        logging.info("Rotated image shape: %s", rotated_images.shape)

        # Run the processing function (e.g., FaceDetailer)
        pre_result: torch.Tensor = func(rotated_images)

        # Rotate result back to original orientation
        logging.info("Rotating results back to original orientation...")
        result_batch = pre_result.shape[0]
        unrotated_list = []
        for i in range(result_batch):
            # Convert tensor to PIL
            img_np = (pre_result[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Rotate back with BICUBIC
            unrotated_pil = pil_img.rotate(
                self._angle,  # Opposite direction
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

            # Convert back to tensor
            unrotated_np = np.array(unrotated_pil).astype(np.float32) / 255.0
            unrotated_list.append(torch.from_numpy(unrotated_np))

        unprocessed_image = torch.stack(unrotated_list, dim=0).to(image.device)

        # Crop to original size (center crop)
        _, h, w, _ = unprocessed_image.shape
        top = (h - orig_h) // 2
        left = (w - orig_w) // 2
        rotated_image = unprocessed_image[:, top : top + orig_h, left : left + orig_w, :]  # noqa: E203

        logging.info("Final cropped image shape: %s", rotated_image.shape)
        return rotated_image
