"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import logging
from typing import Callable
from functools import partial
import torch
from comfy.sd import load_clip, CLIP
import folder_paths
from json_gui.scripts.controlnet_openpose.model import Model
from json_gui.utils import AbsFlow
from json_gui.scripts.mimic import NodeExecutor, prepare_for_serialization
import comfy.model_management


class Flow(AbsFlow):
    """ControlNet OpenPose Flow implementation."""

    @property
    def clip(self) -> CLIP:
        """Get the loaded CLIP model."""
        return self._clip

    @property
    def input_model(self) -> Model:
        """Get the flow model inputs."""
        return self._input_model

    @input_model.setter
    def input_model(self, value: int) -> None:
        assert isinstance(value, int), "Flow input value must be an integer."
        self.save_call = value
        self._input_model.set_save_call(self.save_call)
        self._input_model.update_json()

    @property
    def save_call(self) -> Callable:
        """Get the save image callback."""
        return self._save_call

    @save_call.setter
    def save_call(self, value: int) -> None:
        assert isinstance(value, int), "Save call value must be an integer."
        self._save_call = partial(self.save_image, steps=value)

    def __init__(self, file_path: str, filename: str) -> None:
        super().__init__(file_path, filename)
        self._input_model: Model = Model(self.json_path)
        sd_clip = self._input_model.clip
        self._clip: CLIP = NodeExecutor(
            sd_clip, sd_clip.init_args, {}, self.saved_data
        ).execute()
        logging.info("Loaded CLIP model for flow.")

    def _run_impl(self, steps: int) -> list[str]:
        """Main function to run the ControlNet flow."""

        self.input_model: Model = steps
        raw_nodes: dict[type, dict] = {}

        sd_model = self.input_model.skip_layers_model
        raw_nodes.update({sd_model.__class__: sd_model.init_args})

        prms_node = self.input_model.prompts
        raw_nodes.update({prms_node.__class__: prms_node.init_args})

        # Encode Prompts
        other_clip = prepare_for_serialization(self.clip)
        cond_pos, cond_neg = NodeExecutor(
            prms_node, prms_node.process_args_dict(other_clip), {}, self.saved_data
        ).execute(self.save_call)

        # Run control net conditionings
        logging.info("Applying ControlNet conditionings...")
        for cnet in self.input_model.apply_control_net:
            dict_arg: dict = cnet.process_args_dict(cond_pos, cond_neg)
            cond_pos, cond_neg = NodeExecutor(cnet, dict_arg, sd_raw_node, self.saved_data).execute(self.save_call)

        latent_image = self.input_model.empty_latent.latent(skip_layers_model.vae)

        for sampler_idx, current_sampler in enumerate(self.input_model.simple_k_sampler):
            logging.info("Running Sampler %d...", sampler_idx)

            latent_image = current_sampler.process(
                latent_image, skip_layers_model.get_model(current_sampler.use_tune), cond_pos, cond_neg
            )

            # Decode
            logging.info("Decoding...")
            images = skip_layers_model.vae.get().decode(latent_image.clone())
            logging.info("VAE Output Shape: %s", images.shape)

            # Ensure BHWC (Batch, Height, Width, Channels)
            if images.shape[1] == 3:
                images = images.movedim(1, -1)

            logging.info("Final Image Shape: %s", images.shape)

            self.save_image(images, f"sampler-{sampler_idx}", steps)

        input_dict = {
            "model": skip_layers_model.get_model(self.input_model.face_detailer.use_tune),
            "clip": self.clip,
            "vae": skip_layers_model.vae,
            "positive": cond_pos,
            "negative": cond_neg,
        }
        face_task = partial(self.input_model.face_detailer.detailer_func, **input_dict)
        detailed_image: torch.Tensor = self.input_model.rotator.rotate_image(images, face_task)

        self.save_image(detailed_image, "output", steps, False)

        # Cleanup: unload models and free memory after flow execution
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        logging.info("Done.")
