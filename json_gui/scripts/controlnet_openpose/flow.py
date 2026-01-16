"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import logging
from typing import Callable
from functools import partial
import torch
from json_gui.scripts.controlnet_openpose.model import Model
from json_gui.utils import AbsFlow
from json_gui.scripts.mimic import NodeExecutor
import comfy.model_management


class Flow(AbsFlow):
    """ControlNet OpenPose Flow implementation."""

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

    def _run_impl(self, steps: int) -> list[str]:
        """Main function to run the ControlNet flow."""

        self.input_model: Model = steps

        prms_node = self.input_model.prompts
        sd_clip = self._input_model.clip
        clip_raw = {sd_clip.__class__: sd_clip.init_args}
        # Encode Prompts
        cond_pos, cond_neg = NodeExecutor(prms_node, {}, clip_raw, self.saved_data).execute(self.save_call)

        sd_model = self.input_model.skip_layers_model
        model_raw: dict[type, dict] = {sd_model.__class__: sd_model.init_args}

        latent_image: torch.Tensor = NodeExecutor(
            self.input_model.empty_latent, {}, model_raw, self.saved_data
        ).execute()

        # Run control net conditionings
        logging.info("Applying ControlNet conditionings...")
        for cnet in self.input_model.apply_control_net:
            dict_arg: dict = cnet.process_args_dict(cond_pos, cond_neg)
            cond_pos, cond_neg = NodeExecutor(cnet, dict_arg, model_raw, self.saved_data).execute(self.save_call)

        cond_pos.skip_unwrap = False
        cond_neg.skip_unwrap = False

        for sampler_idx, current_sampler in enumerate(self.input_model.simple_k_sampler):
            logging.info("Running Sampler %d...", sampler_idx)
            dict_arg: dict = current_sampler.process_args_dict(
                latent_image, **{"cond_pos_cnet": cond_pos, "cond_neg_cnet": cond_neg}
            )
            latent_image, images = NodeExecutor(current_sampler, dict_arg, model_raw, self.saved_data).execute(
                self.save_call
            )

        rotator = self.input_model.rotator
        rotated, unrotator = NodeExecutor(rotator, rotator.process_args_dict(images), {}, self.saved_data).execute(
            self.save_call
        )

        # full_raw = clip_raw.update(model_raw)

        input_dict = {
            "input_image": rotated,
            "positive": cond_pos,
            "negative": cond_neg,
        }

        detailed_image: torch.Tensor = NodeExecutor(
            self.input_model.face_detailer, input_dict, model_raw, self.saved_data
        ).execute(self.save_call)

        unrotated = unrotator(detailed_image)

        self.save_call(self.saved_data, unrotated, "unrotated", is_temp=False)

        # Cleanup: unload models and free memory after flow execution
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        logging.info("Done.")
