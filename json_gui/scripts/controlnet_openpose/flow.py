"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import inspect
import os
import logging

import torch
import comfy.sd
import comfy.sample
import folder_paths
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer
from json_gui.scripts.controlnet_openpose.model import Model
from json_gui.utils import save_image
from comfy_extras.nodes_mask import MaskToImage
from model_patcher import ModelPatcher

# Paths - User to replace these
CHECKPOINT_PATH = "sd3.5_medium.safetensors"
CLIP_G_PATH = "sd35m/clip_g.safetensors"
CLIP_L_PATH = "sd35m/clip_l.safetensors"
T5_PATH = "sd35m/t5xxl_fp16.safetensors"


def main(path_file: str, filename: str, steps: int) -> list[str]:
    """Main function to run the ControlNet flow."""
    created_images: list[str] = []

    flow: Model = Model(os.path.join(path_file, f"{filename}.json"))

    # 1. Load Model and VAE
    logging.info("Loading Checkpoint...")
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", CHECKPOINT_PATH)
    model, _a, vae, _b = comfy.sd.load_checkpoint_guess_config(
        ckpt_path, output_vae=True, output_clip=False, embedding_directory=folder_paths.get_folder_paths("embeddings")
    )
    tunned_model: ModelPatcher = flow.skip_layers_model.tunned_model(model)

    # 2. Load Triple CLIP
    logging.info("Loading CLIPs...")
    clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_G_PATH)
    clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_L_PATH)
    clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", T5_PATH)

    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path1, clip_path2, clip_path3],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    # Run preprocessor
    pose_image_tensor = flow.openpose_pose.tensor()

    save_image(created_images, filename, pose_image_tensor, "pose", steps)

    # 6. Encode Prompts
    positive_prompt = flow.positive
    negative_prompt = flow.negative

    logging.info("Encoding prompts...")
    tokens_pos = clip.tokenize(positive_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

    tokens_neg = clip.tokenize(negative_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # 7. Apply ControlNet Advanced
    cond_pos_cnet, cond_neg_cnet = flow.apply_control_net.conditionals(cond_pos, cond_neg, pose_image_tensor, vae)

    latent_image = flow.empty_latent.latent

    for sampler_idx, current_sampler in enumerate(flow.simple_k_sampler):
        logging.info("Running Sampler %d...", sampler_idx)

        latent_image = current_sampler.process(latent_image, tunned_model, cond_pos_cnet, cond_neg_cnet)

        # Decode
        logging.info("Decoding...")
        images = vae.decode(latent_image.clone())
        logging.info("VAE Output Shape: %s", images.shape)

        # Ensure BHWC (Batch, Height, Width, Channels)
        if images.shape[1] == 3:
            images = images.movedim(1, -1)

        logging.info("Final Image Shape: %s", images.shape)

        save_image(created_images, filename, images, f"sampler-{sampler_idx}", steps)

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

        face_arguments = flow.face_detailer.to_dict()

        face_arguments.update(
            {
                "image": input_image,
                "model": model,
                "clip": clip,
                "vae": vae,
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
            save_image(created_images, filename, cropped, f"face-cropped-{idx}", steps)
        for idx, alpha in enumerate(cropped_alpha):
            save_image(created_images, filename, alpha, f"face-alpha-{idx}", steps)
        mask_img_tensor: tuple = MaskToImage().execute(mask).result[0]  # pylint: disable=E1136
        save_image(created_images, filename, mask_img_tensor, "face-mask", steps)
        return result_images

    detailed_image: torch.Tensor = flow.rotator.rotate_image(images, detailer_func)

    save_image(created_images, filename, detailed_image, "output", steps, False)

    logging.info("Done.")
    return created_images
