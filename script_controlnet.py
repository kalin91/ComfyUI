import inspect
import os
from sys import argv
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

import comfy.sd
import comfy.model_management
import comfy.sample
import comfy.controlnet
import folder_paths
import node_helpers
import prompt_server_mocked as _  # noqa: F401

from custom_nodes.comfyui_controlnet_aux import utils as aux_utils
from custom_nodes.comfyui_controlnet_aux.src.custom_controlnet_aux.open_pose import OpenposeDetector

from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer, SAMLoader
from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import UltralyticsDetectorProvider
from flow import Flow

# Paths - User to replace these
CHECKPOINT_PATH = "models/checkpoints/sd3.5_medium.safetensors"
CLIP_G_PATH = "models/clip/sd35m/clip_g.safetensors"
CLIP_L_PATH = "models/clip/sd35m/clip_l.safetensors"
T5_PATH = "models/clip/sd35m/t5xxl_fp16.safetensors"
CONTROLNET_PATH = "models/controlnet/SD3.5m/bokeh_pose_cn.safetensors"

# Register Model Paths
folder_paths.add_model_folder_path("sams", "/data/home2/kalin/models/sams")
folder_paths.add_model_folder_path("ultralytics", "/data/home2/kalin/models/ultralytics")
folder_paths.add_model_folder_path("ultralytics_bbox", "/data/home2/kalin/models/ultralytics/bbox")

OUTPUT_DIR = "output"
TEMP_DIR = "workspace_temp/temp"


def main(filename: str = None):
    # Helper to resolve path or use as is
    def resolve_path(folder_type, filename):
        try:
            return folder_paths.get_full_path_or_raise(folder_type, filename)
        except Exception:
            return filename

    flow: Flow = Flow(filename)

    # 1. Load Model and VAE
    print("Loading Checkpoint...")
    ckpt_path = resolve_path("checkpoints", CHECKPOINT_PATH)
    model, _, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ckpt_path, output_vae=True, output_clip=False, embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    # 2. Load Triple CLIP
    print("Loading CLIPs...")
    clip_path1 = resolve_path("text_encoders", CLIP_G_PATH)
    clip_path2 = resolve_path("text_encoders", CLIP_L_PATH)
    clip_path3 = resolve_path("text_encoders", T5_PATH)

    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path1, clip_path2, clip_path3],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    # Run preprocessor
    pose_image_tensor = flow.openpose_pose.tensor()

    # Save preview of pose image
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    pose_preview_path = os.path.join(TEMP_DIR, "pose_preview.png")
    pose_img_np = 255.0 * pose_image_tensor[0].cpu().numpy()
    pose_img_pil = Image.fromarray(np.clip(pose_img_np, 0, 255).astype(np.uint8))
    pose_img_pil.save(pose_preview_path)
    print(f"Saved pose preview to {pose_preview_path}")

    # 6. Encode Prompts
    positive_prompt = flow.positive
    negative_prompt = flow.negative

    print("Encoding prompts...")
    tokens_pos = clip.tokenize(positive_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

    tokens_neg = clip.tokenize(negative_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # 7. Apply ControlNet Advanced
    cond_pos_cnet, cond_neg_cnet = flow.apply_control_net.conditionals(cond_pos, cond_neg, pose_image_tensor, vae)

    latent_image = flow.empty_latent.latent

    for current_sampler in flow.simple_k_sampler:

        # Prepare noise
        noisy_latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        noise = comfy.sample.prepare_noise(noisy_latent_image, current_sampler.seed, None)

        sampler_arguments = current_sampler.to_dict()

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

        sampler_signature = inspect.signature(comfy.sample.sample)
        for key in sampler_arguments.keys():
            if key not in sampler_signature.parameters:
                raise ValueError(f"Unexpected argument '{key}' for comfy.sample.sample")

        latent_image = comfy.sample.sample(**sampler_arguments)

    # 10. Decode
    print("Decoding...")
    images = vae.decode(latent_image)
    print(f"VAE Output Shape: {images.shape}")

    # Ensure BHWC (Batch, Height, Width, Channels)
    if images.shape[1] == 3:
        images = images.movedim(1, -1)

    print(f"Final Image Shape: {images.shape}")

    # Save Refiner Output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i, image in enumerate(images):
        img_np = 255.0 * image.cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        output_path = os.path.join(OUTPUT_DIR, f"refiner_output_{i}.png")
        img_pil.save(output_path)
        print(f"Saved refiner output to {output_path}")

    # 10.5 FaceDetailer
    print("Running FaceDetailer...")

    # Load Models
    bbox_model_name = "bbox/yolo11x_face_detect2.pt"
    sam_model_name = "sam_vit_l_0b3195.pth"

    bbox_provider = UltralyticsDetectorProvider()
    # UltralyticsDetectorProvider.doit returns (BBOX_DETECTOR, SEGM_DETECTOR)
    bbox_detector, _ = bbox_provider.doit(bbox_model_name)

    sam_loader = SAMLoader()
    # SAMLoader.load_model returns (SAM_MODEL,)
    sam_model = sam_loader.load_model(sam_model_name)[0]

    face_detailer = FaceDetailer()

    # FaceDetailer.doit(image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook)

    # Note: Arguments might vary slightly depending on version, checking signature would be good.
    # Assuming standard arguments based on common usage.

    face_arguments = flow.face_detailer.to_dict()

    face_arguments.update(
        {
            "image": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": cond_pos,
            "negative": cond_neg,
            "sam_model_opt": sam_model,
            "segm_detector_opt": None,  # Not using segm detector here
            "detailer_hook": None,
            "bbox_detector": bbox_detector,
        }
    )

    # validate face_arguments keys against FaceDetailer.doit signature would be ideal
    face_signature = inspect.signature(face_detailer.doit)
    for key in face_arguments.keys():
        if key not in face_signature.parameters:
            raise ValueError(f"Unexpected argument '{key}' for FaceDetailer.doit")

    result_images = face_detailer.doit(**face_arguments)[0]

    # 11. Save Image
    print("Saving Images...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for i, image in enumerate(result_images):
        img_np = 255.0 * image.cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        output_path = os.path.join(OUTPUT_DIR, f"output_{i}.png")
        img_pil.save(output_path)
        print(f"Saved to {output_path}")

    print("Done.")


if __name__ == "__main__":
    with torch.inference_mode():
        # pass first argument as filename if needed
        assert len(argv) > 1, "Please provide a filename as an argument."
        main(argv[1])
