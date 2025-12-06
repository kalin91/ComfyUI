import sys
import os
import torch
import numpy as np
from PIL import Image

import comfy.sd
import comfy.model_management
import comfy.sample
import comfy.utils
import folder_paths

# Paths - User to replace these
CHECKPOINT_PATH = "models/checkpoints/sd3.5_medium.safetensors"
CLIP_G_PATH = "models/clip/sd35m/clip_g.safetensors"
CLIP_L_PATH = "models/clip/sd35m/clip_l.safetensors"
T5_PATH = "models/clip/sd35m/t5xxl_fp16.safetensors"

OUTPUT_DIR = "output"

def main():
    # 1. Load Model and VAE
    print("Loading Checkpoint...")
    # Note: load_checkpoint_guess_config returns (model, clip, vae, clipvision)
    # We only need model and vae from here, as we load CLIP separately
    # We use folder_paths to resolve paths if they are just filenames, or use absolute paths if provided
    
    # Helper to resolve path or use as is
    def resolve_path(folder_type, filename):
        try:
            return folder_paths.get_full_path_or_raise(folder_type, filename)
        except:
            return filename

    ckpt_path = resolve_path("checkpoints", CHECKPOINT_PATH)
    
    model, _, vae, _ = comfy.sd.load_checkpoint_guess_config(
        ckpt_path, 
        output_vae=True, 
        output_clip=False, 
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    # 2. Load Triple CLIP
    print("Loading CLIPs...")
    clip_path1 = resolve_path("text_encoders", CLIP_G_PATH)
    clip_path2 = resolve_path("text_encoders", CLIP_L_PATH)
    clip_path3 = resolve_path("text_encoders", T5_PATH)
    
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path1, clip_path2, clip_path3], 
        embedding_directory=folder_paths.get_folder_paths("embeddings")
    )

    # 3. Encode Prompts
    positive_prompt = "an olive skinned queen sitting on her throne with a long dress spreading over the floor, with straight hair, rusty crown, blue tabard with white lion emblem, sitting judging pose.  Renaissance oil painting style"
    negative_prompt = ""

    print("Encoding prompts...")
    tokens_pos = clip.tokenize(positive_prompt)
    cond_pos = clip.encode_from_tokens_scheduled(tokens_pos)

    tokens_neg = clip.tokenize(negative_prompt)
    cond_neg = clip.encode_from_tokens_scheduled(tokens_neg)

    # 4. Create Empty Latent
    width = 1024
    height = 1024
    batch_size = 1
    print(f"Creating empty latent {width}x{height}...")
    latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=comfy.model_management.intermediate_device())
    
    # 5. Sample
    seed = 189895584711893
    steps = 30
    cfg = 8.0
    sampler_name = "euler"
    scheduler = "sgm_uniform"
    denoise = 1.0

    print(f"Sampling with seed={seed}, steps={steps}, cfg={cfg}, sampler={sampler_name}, scheduler={scheduler}...")
    
    # Prepare noise
    # From common_ksampler logic
    latent_image = latent
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
    
    noise = comfy.sample.prepare_noise(latent_image, seed, None)
    
    # Sample
    samples = comfy.sample.sample(
        model, noise, steps, cfg, sampler_name, scheduler, 
        cond_pos, cond_neg, latent_image, 
        denoise=denoise, disable_noise=False, start_step=None, last_step=None, 
        force_full_denoise=False, noise_mask=None, callback=None, disable_pbar=False, seed=seed
    )

    # 6. Decode
    print("Decoding...")
    images = vae.decode(samples)

    # 7. Save Image
    print("Saving image...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    for i, image in enumerate(images):
        img_data = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(img_data, 0, 255).astype(np.uint8))
        output_path = os.path.join(OUTPUT_DIR, f"ComfyUI_{i:05d}_.png")
        img.save(output_path)
        print(f"Saved to {output_path}")

    print("Done.")

if __name__ == "__main__":
    main()
