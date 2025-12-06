import os
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


# Paths - User to replace these
CHECKPOINT_PATH = "models/checkpoints/sd3.5_medium.safetensors"
CLIP_G_PATH = "models/clip/sd35m/clip_g.safetensors"
CLIP_L_PATH = "models/clip/sd35m/clip_l.safetensors"
T5_PATH = "models/clip/sd35m/t5xxl_fp16.safetensors"
CONTROLNET_PATH = "models/controlnet/SD3.5m/bokeh_pose_cn.safetensors"
IMAGE_PATH = "input/reina_basto.jpeg"

# Register Model Paths
folder_paths.add_model_folder_path("sams", "/data/home2/kalin/models/sams")
folder_paths.add_model_folder_path("ultralytics", "/data/home2/kalin/models/ultralytics")
folder_paths.add_model_folder_path("ultralytics_bbox", "/data/home2/kalin/models/ultralytics/bbox")

OUTPUT_DIR = "output"
TEMP_DIR = "workspace_temp/temp"


def main():
    # Helper to resolve path or use as is
    def resolve_path(folder_type, filename):
        try:
            return folder_paths.get_full_path_or_raise(folder_type, filename)
        except Exception:
            return filename

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

    # 3. Load ControlNet
    print("Loading ControlNet...")
    controlnet_path = resolve_path("controlnet", CONTROLNET_PATH)
    controlnet = comfy.controlnet.load_controlnet(controlnet_path)

    # 4. Load Image
    print("Loading Image...")
    image_path = resolve_path("input", IMAGE_PATH)
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

    # 5. OpenPose Preprocessor
    print("Running OpenPose Preprocessor...")
    # Parameters from workflow: enable, enable, enable, 768, enable
    detect_hand = True
    detect_body = True
    detect_face = True
    resolution = 768
    scale_stick_for_xinsr_cn = True

    # Initialize OpenPose Detector
    openpose_model = OpenposeDetector.from_pretrained().to(comfy.model_management.get_torch_device())

    def openpose_func(image, **kwargs):
        pose_img, openpose_dict = openpose_model(image, **kwargs)
        return pose_img

    # Run preprocessor
    pose_image_tensor = aux_utils.common_annotator_call(
        openpose_func,
        img_tensor,
        include_hand=detect_hand,
        include_face=detect_face,
        include_body=detect_body,
        image_and_json=True,
        xinsr_stick_scaling=scale_stick_for_xinsr_cn,
        resolution=resolution,
    )

    # Save preview of pose image
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    pose_preview_path = os.path.join(TEMP_DIR, "pose_preview.png")
    pose_img_np = 255.0 * pose_image_tensor[0].cpu().numpy()
    pose_img_pil = Image.fromarray(np.clip(pose_img_np, 0, 255).astype(np.uint8))
    pose_img_pil.save(pose_preview_path)
    print(f"Saved pose preview to {pose_preview_path}")

    # 6. Encode Prompts
    positive_prompt = (
        "Renaissance oil painting style, highly detailed.\n"
        "A beautiful young queen with a small and elegant crown, holding a long dark wood staff that extends from her left hand straight to the ground, with bottom tip firmly on the floor. "
        "Queen's left hand firmly gripping the upper third of the staff, with fingers wrapped naturally around the shaft, that held vertically, no bending, no curved, no twisted, no organic staff shapes. "
        "it has upper part broken, with subtle carved patterns along its shaft. Queen's left hand empty. The queen has a sexy, delicate build with a willpower expression, full of confidence and charisma. \n"
        "The queen is viewed from a low-angle shot, seated on a grand, monolithic high-back marbel seat without armrests, it has subtle gold inlays, and has completely smooth sides, no legs and no separate supports.\n"
        "Behind the seat are large windows with light curtains through which a warm stream of sunlight shines through, wooden and gold architectural pilars glow softly. "
        "Potted plants and tapestries with natural motifs below the windows. A marble lion on a pedestal in the background at the top"
    )
    negative_prompt = "blur, fog, Out-of-focus background, pants"

    print("Encoding prompts...")
    tokens_pos = clip.tokenize(positive_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

    tokens_neg = clip.tokenize(negative_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # 7. Apply ControlNet Advanced
    print("Applying ControlNet...")
    strength = 0.61
    start_percent = 0.0
    end_percent = 0.586

    # Logic from ControlNetApplyAdvanced.apply_controlnet
    if strength == 0:
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
                        control_hint, strength, (start_percent, end_percent), vae=vae
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d["control"] = c_net
                d["control_apply_to_uncond"] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        cond_pos_cnet, cond_neg_cnet = out[0], out[1]

    # 8. Create Empty Latent
    width = 592
    height = 1024
    batch_size = 1
    print(f"Creating empty latent {width}x{height}...")
    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device()
    )
    latent_dict = {"samples": latent}

    # 9. Sample 1
    seed = 878324280905954
    steps = 30
    cfg = 8.4
    sampler_name = "dpm_2_ancestral"
    scheduler = "sgm_uniform"
    denoise = 1.0

    print(f"Sampling 1 with seed={seed}, steps={steps}, cfg={cfg}, sampler={sampler_name}, scheduler={scheduler}...")

    # Prepare noise
    latent_image = latent_dict["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    noise = comfy.sample.prepare_noise(latent_image, seed, None)

    # Sample 1
    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        cond_pos_cnet,
        cond_neg_cnet,
        latent_image,
        denoise=denoise,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=seed,
    )

    # 9.5 Sample 2 (Refiner)
    print("Sampling 2 (Refiner)...")
    seed_2 = 878324280905954
    steps_2 = 34
    cfg_2 = 8.5
    denoise_2 = 0.7

    # Prepare noise for second pass (using output of first pass)
    latent_image_2 = samples
    noise_2 = comfy.sample.prepare_noise(latent_image_2, seed_2, None)

    samples_2 = comfy.sample.sample(
        model,
        noise_2,
        steps_2,
        cfg_2,
        sampler_name,
        scheduler,
        cond_pos_cnet,
        cond_neg_cnet,
        latent_image_2,
        denoise=denoise_2,
        disable_noise=False,
        start_step=None,
        last_step=None,
        force_full_denoise=False,
        noise_mask=None,
        callback=None,
        disable_pbar=False,
        seed=seed_2,
    )

    # 10. Decode
    print("Decoding...")
    images = vae.decode(samples_2)
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

    # Parameters for FaceDetailer
    guide_size = 592
    guide_size_for = True
    max_size = 1024
    seed_3 = 878324280905954
    steps_3 = 20
    cfg_3 = 8.0
    sampler_name_3 = "euler_ancestral"
    scheduler_3 = "normal"
    denoise_3 = 0.5
    feather = 5
    noise_mask = True
    force_inpaint = False
    drop_size = 10
    refiner_ratio = 0.2
    batch_size_3 = 1
    cycle = 1

    # FaceDetailer.doit(image, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, bbox_threshold, bbox_dilation, bbox_crop_factor, sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold, sam_mask_hint_use_negative, drop_size, bbox_detector, sam_model_opt, segm_detector_opt, detailer_hook)

    # Note: Arguments might vary slightly depending on version, checking signature would be good.
    # Assuming standard arguments based on common usage.

    result_images = face_detailer.doit(
        image=images,
        model=model,
        clip=clip,
        vae=vae,
        guide_size=guide_size,
        guide_size_for=guide_size_for,
        max_size=max_size,
        seed=seed_3,
        steps=steps_3,
        cfg=cfg_3,
        sampler_name=sampler_name_3,
        scheduler=scheduler_3,
        positive=cond_pos,
        negative=cond_neg,
        denoise=denoise_3,
        feather=feather,
        noise_mask=noise_mask,
        force_inpaint=force_inpaint,
        bbox_threshold=0.6,
        bbox_dilation=15,
        bbox_crop_factor=3.0,
        sam_detection_hint="center-1",
        sam_dilation=0,
        sam_threshold=0.93,
        sam_bbox_expansion=0,
        sam_mask_hint_threshold=0.7,
        sam_mask_hint_use_negative="False",
        drop_size=drop_size,
        bbox_detector=bbox_detector,
        wildcard="",
        cycle=cycle,
        sam_model_opt=sam_model,
        segm_detector_opt=None,  # Not using segm detector here
        detailer_hook=None,
    )[0]

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
        main()
