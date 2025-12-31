"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import gc
import inspect
import logging

import torch
import comfy.sd
import folder_paths
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer
from comfy.model_management import unload_all_models, soft_empty_cache
from json_gui.scripts.controlnet_openpose.model import Model
from json_gui.utils import AbsFlow
from comfy_extras.nodes_mask import MaskToImage

# Paths - User to replace these
CLIP_G_PATH = "sd35m/clip_g.safetensors"
CLIP_L_PATH = "sd35m/clip_l.safetensors"
T5_PATH = "sd35m/t5xxl_fp16.safetensors"


class Flow(AbsFlow):
    """ControlNet OpenPose Flow implementation."""

    @property
    def flow(self) -> Model:
        """Returns the Model instance representing the flow."""
        return self._flow

    @flow.deleter
    def flow(self) -> None:
        """Deletes the Model instance and frees resources."""
        del self._flow
        unload_all_models()
        soft_empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @flow.setter
    def flow(self, value: Model) -> None:
        """Sets the Model instance."""
        self._flow = value

    def __init__(self, file_path: str, file_name: str) -> None:
        """Initializes the Flow with specific file paths."""
        super().__init__(file_path, file_name)
        self._steps: int = 0

        def save_call(i: torch.Tensor, n: str) -> torch.Tensor:
            return self.save_image(i, n, self._steps)

        self._flow: Model = Model(self.json_path, save_call)

        # 2. Load Triple CLIP
        logging.info("Loading CLIPs...")
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_G_PATH)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_L_PATH)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", T5_PATH)

        self._clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

    def _run_impl(self, steps: int) -> list[str]:
        """Main function to run the ControlNet flow."""

        self._steps = steps
        # Cleanup model components before deleting to ensure proper VRAM release
        if hasattr(self, "_flow"):
            if hasattr(self._flow, "skip_layers_model"):
                self._flow.skip_layers_model.cleanup()
            if hasattr(self._flow, "face_detailer"):
                self._flow.face_detailer.cleanup()
            if hasattr(self._flow, "apply_control_net"):
                for cnet in self._flow.apply_control_net:
                    if hasattr(cnet, "cleanup"):
                        cnet.cleanup()
        del self.flow
        # Ensure CUDA memory is fully freed before loading new checkpoint
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.flow = Model()

        skip_layers_model = self._flow.skip_layers_model

        # 6. Encode Prompts
        positive_prompt: str = self._flow.positive
        negative_prompt: str = self._flow.negative

        logging.info("Encoding prompts...")
        tokens_pos = self._clip.tokenize(positive_prompt)
        cond_pos = self._clip.encode_from_tokens_scheduled(tokens_pos)

        tokens_neg = self._clip.tokenize(negative_prompt)
        cond_neg = self._clip.encode_from_tokens_scheduled(tokens_neg)

        del tokens_pos
        del tokens_neg
        torch.cuda.empty_cache()

        # Run control net conditionings
        logging.info("Applying ControlNet conditionings...")
        for cnet in self._flow.apply_control_net:
            cond_pos, cond_neg = cnet.conditionals(cond_pos, cond_neg, skip_layers_model.vae)

        latent_image = self._flow.empty_latent.latent

        for sampler_idx, current_sampler in enumerate(self._flow.simple_k_sampler):
            logging.info("Running Sampler %d...", sampler_idx)

            latent_image = current_sampler.process(
                latent_image, skip_layers_model.get_model(current_sampler.use_tune), cond_pos, cond_neg
            )

            # Decode
            logging.info("Decoding...")
            images = skip_layers_model.vae.decode(latent_image.clone())
            logging.info("VAE Output Shape: %s", images.shape)

            # Ensure BHWC (Batch, Height, Width, Channels)
            if images.shape[1] == 3:
                images = images.movedim(1, -1)

            logging.info("Final Image Shape: %s", images.shape)

            self.save_image(images, f"sampler-{sampler_idx}", steps)

        def detailer_func(input_image: torch.Tensor) -> torch.Tensor:
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

            face_arguments = self._flow.face_detailer.to_dict()

            face_arguments.update(
                {
                    "image": input_image,
                    "model": skip_layers_model.get_model(self._flow.face_detailer.use_tune),
                    "clip": self._clip,
                    "vae": skip_layers_model.vae,
                    "positive": cond_pos,
                    "negative": cond_neg,
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

            for idx, cropped in enumerate(cropped_images):
                self.save_image(cropped, f"face-cropped-{idx}", steps)
            for idx, alpha in enumerate(cropped_alpha):
                self.save_image(alpha, f"face-alpha-{idx}", steps)
            mask_img_tensor: tuple = MaskToImage().execute(mask).result[0]  # pylint: disable=E1136
            self.save_image(mask_img_tensor, "face-mask", steps)
            return result_images

        detailed_image: torch.Tensor = self._flow.rotator.rotate_image(images, detailer_func)

        self.save_image(detailed_image, "output", steps, False)

        logging.info("Done.")
