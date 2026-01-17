"""Mimic child classes for various components."""

import inspect
import logging
from functools import partial
from typing import Any, Callable, Optional, Tuple, cast
import torch
import numpy as np
from segment_anything.build_sam import Sam
import comfy.model_management
from comfy_extras.nodes_sd3 import SkipLayerGuidanceSD3
from comfy_extras.nodes_mask import MaskToImage
from comfy.sample import fix_empty_latent_channels, prepare_noise, sample
from comfy.sd import load_checkpoint_guess_config, VAE, CLIP, load_clip
from comfy.model_patcher import ModelPatcher
from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import UltralyticsDetectorProvider
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import SAMLoader, FaceDetailer
from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import subcore
import folder_paths
from PIL import Image
from json_gui.scripts.mimic import MimicNode, DataWrapper

type Conditional = list[tuple[torch.Tensor, dict[str, Any]]]


class Sd3Clip(MimicNode):
    """A class representing SD3 CLIP settings."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "Sd3Clip"]]:
        return []  # No class params needed for Prompts

    CLIP_G_PATH = "sd35m/clip_g.safetensors"
    CLIP_L_PATH = "sd35m/clip_l.safetensors"
    T5_PATH = "sd35m/t5xxl_fp16.safetensors"

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Sd3Clip."""
        return "sd3_clip"

    # pylint: disable=W0201
    # pylint: disable=W0221
    def _process_impl(self) -> CLIP:
        """Loads the Triple CLIP model."""
        logging.info("Loading CLIPs...")
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", self.CLIP_G_PATH)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", self.CLIP_L_PATH)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", self.T5_PATH)
        return load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

    # pylint: disable=W0221
    def _update_impl(self) -> None:
        """PASS - Loads the CLIP model."""
        self._clip = None

    def __init__(self):
        super().__init__()
        self.update()


class SkipLayers(MimicNode):
    """A class representing skip layer guidance settings."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "SkipLayers"]]:
        return []  # No class params needed for Prompts

    @classmethod
    def key(cls) -> str:
        """Returns the key for the SkipLayers."""
        return "skip_layers_model"

    @property
    def vae(self) -> VAE:
        """Returns the VAE of the model."""
        if self._vae is None:
            raise ValueError("Model has not been processed yet. Call process() first.")
        return self._vae

    @property
    def model(self) -> ModelPatcher:
        """Returns the model."""
        res = self._tunned_model if self.use_tuned else self._base_model
        if res is None:
            raise ValueError("Model has not been processed yet. Call process() first.")
        return res

    @property
    def use_tuned(self) -> bool:
        """Returns whether to use the tuned model."""
        return self._use_tuned

    @use_tuned.setter
    def use_tuned(self, value: bool) -> None:
        """Sets whether to use the tuned model."""
        self._use_tuned = value

    CHECKPOINT_PATH = "sd3.5_medium.safetensors"

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _process_impl(self) -> Tuple[ModelPatcher, VAE]:
        """Returns the tuned model."""
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
        return self.model, self._vae

    def process(self) -> Tuple[ModelPatcher, VAE]:
        """Processes and returns the model and VAE."""
        return super().process()

    # pylint: disable=W0221
    # pylint: disable=W0201
    def _update_impl(self, layers: list[int], scale: float, start_percent: float, end_percent: float) -> None:
        self._layers = ",".join(str(layer) for layer in layers)
        self._scale = scale
        self._start_percent = start_percent
        self._end_percent = end_percent
        self._vae = None
        self._use_tuned = False
        self._base_model = None
        self._tunned_model = None

    def __init__(self, layers: list[int], scale: float, start_percent: float, end_percent: float):
        super().__init__()
        self.update(layers=layers, scale=scale, start_percent=start_percent, end_percent=end_percent)


class SimpleKSampler(MimicNode):
    """A simple KSampler class for demonstration purposes."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "SimpleKSampler"]]:
        res: list[MimicNode.ClassParam[Any, "SimpleKSampler"]] = []
        res.append(
            cls.build_class_param(SkipLayers, lambda inst: cls._set_current_model(inst) or {"node_model": inst})
        )
        return res

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
        self, latent_image: torch.Tensor, node_model: SkipLayers, cond_pos_cnet: Any, cond_neg_cnet: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A placeholder method to simulate processing."""
        try:
            node_model.use_tuned = self.use_tune
            model = node_model.model
            vae = node_model.vae

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
            logging.info("Decoding...")
            res: torch.Tensor = sample(**sampler_arguments)
            images = vae.decode(res.clone())
            logging.info("VAE Output Shape: %s", images.shape)

            # Ensure BHWC (Batch, Height, Width, Channels)
            if images.shape[1] == 3:
                images = images.movedim(1, -1)

            logging.info("Final Image Shape: %s", images.shape)

            self._save_tensor(images, self.key())

            return res, images
        except Exception as e:
            logging.exception("Error in SimpleKSampler processing: %s", e)
            raise e

    def process(
        self, latent_image: torch.Tensor, node_model: SkipLayers, cond_pos_cnet: Any, cond_neg_cnet: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes the latent image and returns the result."""
        return super().process(latent_image, node_model, cond_pos_cnet, cond_neg_cnet)


class Prompts(MimicNode):
    """A class representing positive and negative prompts."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "Prompts"]]:
        return []  # No class params needed for Prompts

    @classmethod
    def key(cls) -> str:
        """Returns the key for the Prompts."""
        return "prompts"

    # pylint: disable=W0221
    def _process_impl(self) -> tuple[DataWrapper[Conditional], DataWrapper[Conditional]]:
        """Encodes the positive and negative prompts using the provided CLIP model."""
        clip: CLIP = Sd3Clip().process()
        logging.info("Encoding prompts...")
        tokens_pos = clip.tokenize(self._positive)
        cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

        tokens_neg = clip.tokenize(self._negative)
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

        del tokens_pos
        del tokens_neg
        torch.cuda.empty_cache()
        return tuple(DataWrapper(value=cond, skip_unwrap=False) for cond in (cond_pos, cond_neg))

    def process(self) -> tuple[DataWrapper[Conditional], DataWrapper[Conditional]]:
        """Processes the prompts using the provided CLIP model."""
        return super().process()

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
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "EmptyLatent"]]:
        res: list[MimicNode.ClassParam[Any, "EmptyLatent"]] = []
        res.append(
            cls.build_class_param(
                SkipLayers, lambda inst: cls._set_current_model(inst) or {"vae": cast(SkipLayers, inst).process()[1]}
            )
        )
        return res

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
    def process(self, vae: VAE) -> torch.Tensor:
        """Generates and returns an empty latent tensor."""
        return super().process(vae)

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


class FaceDetailerNode(SimpleKSampler):
    """A class representing face detailer settings."""

    @classmethod
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "FaceDetailerNode"]]:
        res: list[MimicNode.ClassParam[Any, "FaceDetailerNode"]] = []
        res.append(
            cls.build_class_param(
                SkipLayers, lambda inst: cls._set_current_model(inst) or {"node_model": inst, "node_clip": Sd3Clip()}
            )
        )
        return res

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
    def _process_impl(
        self, input_image: torch.Tensor, positive: Any, negative: Any, node_model: SkipLayers, node_clip: Sd3Clip
    ) -> torch.Tensor:
        """Function to process image once rotated."""
        try:
            node_model.use_tuned = self.use_tune
            model = node_model.model
            vae = node_model.vae
            clip = node_clip.process()

            # FaceDetailer
            logging.info("Running FaceDetailer...")

            face_detailer = FaceDetailer()

            face_arguments = self.to_dict()

            face_arguments.update(
                {
                    "model": model,
                    "vae": vae,
                    "clip": clip,
                    "positive": positive,
                    "negative": negative,
                    "image": input_image,
                    "segm_detector_opt": None,  # Not using segm detector here
                    "detailer_hook": None,
                }
            )
        except Exception as e:
            logging.exception("Error preparing FaceDetailer arguments: %s", e)
            raise e

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
    def process(
        self, input_image: torch.Tensor, positive: Any, negative: Any, node_model: SkipLayers, node_clip: Sd3Clip
    ) -> torch.Tensor:
        """Processes the image using the FaceDetailer."""
        return MimicNode.process(self, input_image, positive, negative, node_model, node_clip)

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
    def _class_param_definitions(cls) -> list[MimicNode.ClassParam[Any, "Rotator"]]:
        return []  # No class params needed for Prompts

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
    def _process_impl(self, image: torch.Tensor) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        """Rotates the given image tensor by the specified angle.

        Uses PIL with BICUBIC interpolation to minimize quality loss during rotation.
        Converts tensor -> PIL -> rotate -> tensor for better quality preservation.
        """

        if self._angle == 0:
            return image, lambda x: x  # No rotation needed

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

        result_fun = partial(Rotator._undo_rotate, angle=self._angle, orig_h=orig_h, orig_w=orig_w)
        self._save_tensor(rotated_images, "rotated_image")

        return rotated_images, result_fun

    @classmethod
    def _undo_rotate(cls, image: torch.Tensor, angle: float, orig_h: int, orig_w: int) -> torch.Tensor:

        # Rotate result back to original orientation
        logging.info("Rotating results back to original orientation...")
        result_batch = image.shape[0]
        unrotated_list = []
        for i in range(result_batch):
            # Convert tensor to PIL
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            # Rotate back with BICUBIC
            unrotated_pil = pil_img.rotate(
                angle,  # Opposite direction
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
