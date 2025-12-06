"""Script to run a ControlNet flow with Triple CLIP and FaceDetailer integration."""

import inspect
import os
from sys import argv
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.sample
import folder_paths
import prompt_server_mocked as _  # noqa: F401


from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer, SAMLoader
from custom_nodes.ComfyUI_Impact_Subpack.modules.subpack_nodes import UltralyticsDetectorProvider
from flow import Flow

# Paths - User to replace these
CHECKPOINT_PATH = "sd3.5_medium.safetensors"
CLIP_G_PATH = "sd35m/clip_g.safetensors"
CLIP_L_PATH = "sd35m/clip_l.safetensors"
T5_PATH = "sd35m/t5xxl_fp16.safetensors"

# Register Model Paths
folder_paths.add_model_folder_path("sams", "/data/home2/kalin/models/sams")
folder_paths.add_model_folder_path("ultralytics", "/data/home2/kalin/models/ultralytics")
folder_paths.add_model_folder_path("ultralytics_bbox", "/data/home2/kalin/models/ultralytics/bbox")

OUTPUT_DIR = "workspace_temp/output"
TEMP_DIR = "workspace_temp/temp"


def main(filename: str, steps: int) -> list[str]:
    """Main function to run the ControlNet flow."""
    created_images: list[str] = []

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    flow: Flow = Flow(filename)

    # 1. Load Model and VAE
    print("Loading Checkpoint...")
    ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", CHECKPOINT_PATH)
    model, _, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ckpt_path, output_vae=True, output_clip=False, embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    # 2. Load Triple CLIP
    print("Loading CLIPs...")
    clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_G_PATH)
    clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", CLIP_L_PATH)
    clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", T5_PATH)

    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path1, clip_path2, clip_path3],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )

    # Run preprocessor
    pose_image_tensor = flow.openpose_pose.tensor()

    h: int = 0
    pose_file_saved: bool = False
    while not pose_file_saved and h < 40:
        pose_filename = os.path.join(TEMP_DIR, f"{filename}_pose_preview_{h}.png")
        if os.path.exists(pose_filename):
            h += 1
            continue  # Skip if already exists
        img_np = 255.0 * pose_image_tensor[0].cpu().numpy()
        img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
        img_pil.save(pose_filename)
        print(f"Saved pose preview to {pose_filename}")
        pose_file_saved = True
        created_images.append(pose_filename)
        break
    if not pose_file_saved:
        raise RuntimeError("Failed to save pose preview after multiple attempts. clean up temp files.")
    if len(created_images) >= steps:
        return created_images

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

    for sampler_idx, current_sampler in enumerate(flow.simple_k_sampler):

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
        images = vae.decode(latent_image.clone())
        print(f"VAE Output Shape: {images.shape}")

        # Ensure BHWC (Batch, Height, Width, Channels)
        if images.shape[1] == 3:
            images = images.movedim(1, -1)

        print(f"Final Image Shape: {images.shape}")

        j: int = 0
        file_saved: bool = False
        while not file_saved and j < 40:
            for i, image in enumerate(images):
                sampler_file_name = os.path.join(TEMP_DIR, f"{filename}_sampler_{sampler_idx}_{i}_{j}.png")
                if os.path.exists(sampler_file_name):
                    j += 1
                    continue  # Skip if already exists
                img_np = 255.0 * image.cpu().numpy()
                img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
                img_pil.save(sampler_file_name)
                print(f"Saved refiner output to {sampler_file_name}")
                file_saved = True
                created_images.append(sampler_file_name)
                break
        if not file_saved:
            raise RuntimeError("Failed to save refiner output after multiple attempts. clean up temp files.")         
        if len(created_images) >= steps:
            return created_images

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
    k: int = 0
    output_file_saved: bool = False
    while not output_file_saved and k < 40:
        for i, image in enumerate(result_images):
            output_file_name = os.path.join(OUTPUT_DIR, f"{filename}_{i}_{k}.png")
            if os.path.exists(output_file_name):
                k += 1
                continue  # Skip if already exists
            img_np = 255.0 * image.cpu().numpy()
            img_pil = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))
            img_pil.save(output_file_name)
            print(f"Saved refiner output to {output_file_name}")
            output_file_saved = True
            created_images.append(output_file_name)
            break
    if not output_file_saved:
        raise RuntimeError("Failed to save refiner output after multiple attempts. clean up temp files.")
    if len(created_images) >= steps:
        return created_images

    print("Done.")
    return created_images


if __name__ == "__main__":
    with torch.inference_mode():
        # pass first argument as filename if needed
        assert len(argv) > 1, "Please provide a filename as an argument."
        main(argv[1])
