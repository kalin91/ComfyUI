"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import logging
from functools import partial
import torch
from comfy.sd import load_clip
import folder_paths
from json_gui.scripts.controlnet_openpose.model import Model
from json_gui.utils import AbsFlow
import comfy.model_management

# Paths - User to replace these
CLIP_G_PATH = "sd35m/clip_g.safetensors"
CLIP_L_PATH = "sd35m/clip_l.safetensors"
T5_PATH = "sd35m/t5xxl_fp16.safetensors"


class Flow(AbsFlow):
    """ControlNet OpenPose Flow implementation."""

    def _run_impl(self, steps: int) -> list[str]:
        """Main function to run the ControlNet flow."""

        def save_call(i: torch.Tensor, n: str) -> torch.Tensor:
            return self.save_image(i, n, steps)

        flow: Model = Model(self.json_path, save_call)

        skip_layers_model = flow.skip_layers_model

        # 2. Load Triple CLIP
        logging.info("Loading CLIPs...")
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_G_PATH)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_L_PATH)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", T5_PATH)

        clip = load_clip(
            ckpt_paths=[clip_path1, clip_path2, clip_path3],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )

        # 6. Encode Prompts
        positive_prompt: str = flow.positive
        negative_prompt: str = flow.negative

        logging.info("Encoding prompts...")
        tokens_pos = clip.tokenize(positive_prompt)
        cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

        tokens_neg = clip.tokenize(negative_prompt)
        cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

        del tokens_pos
        del tokens_neg
        torch.cuda.empty_cache()

        # Run control net conditionings
        logging.info("Applying ControlNet conditionings...")
        for cnet in flow.apply_control_net:
            cond_pos, cond_neg = cnet.conditionals(cond_pos, cond_neg, skip_layers_model.vae)

        latent_image = flow.empty_latent.latent

        for sampler_idx, current_sampler in enumerate(flow.simple_k_sampler):
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

        input_dict = {
            "model": skip_layers_model.get_model(flow.face_detailer.use_tune),
            "clip": clip,
            "vae": skip_layers_model.vae,
            "positive": cond_pos,
            "negative": cond_neg,
        }
        face_task = partial(flow.face_detailer.detailer_func, input_dict=input_dict)
        detailed_image: torch.Tensor = flow.rotator.rotate_image(images, face_task)

        self.save_image(detailed_image, "output", steps, False)

        # Cleanup: unload models and free memory after flow execution
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        logging.info("Done.")
