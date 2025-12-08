from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, ControlNetModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
from tqdm.auto import tqdm

def preprocess_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.BILINEAR)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    return transform(image).unsqueeze(0)


def preprocess_mask(path: str, resolution: int) -> torch.Tensor:
    # V4 Pixel-space: High-res mask input (512x512)
    # Ensure it's 1-channel
    mask = Image.open(path).convert("L")
    mask = mask.resize((resolution, resolution), Image.NEAREST) # Nearest to keep sharp edges
    tensor = transforms.ToTensor()(mask).unsqueeze(0)
    return torch.clamp(tensor, 0.0, 1.0)


def encode_texts(
    tokenizer: AutoTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    num_images: int,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    text_embeddings = text_encoder(text_inputs.input_ids)[0].repeat(num_images, 1, 1)

    uncond_embeddings = None
    if negative_prompt is not None:
        uncond_inputs = tokenizer(
            [negative_prompt],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        uncond_embeddings = text_encoder(uncond_inputs.input_ids)[0].repeat(num_images, 1, 1)

    return text_embeddings, uncond_embeddings


def main():
    parser = argparse.ArgumentParser(description="Inference for the hairline-conditioned V4 (Pixel-space ControlNet).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to trained V4 ControlNet weights.")
    parser.add_argument("--bald_path", type=str, required=True, help="Path to an aligned bald image (used as base canvas).")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the forehead mask.")
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic, detailed hair")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, artificial")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scale", type=float, default=1.0, help="Scale of ControlNet injection.")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--noise_strength", type=float, default=0.9, help="Denoising strength for img2img (1.0 = full noise, 0.0 = no noise).")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dtype == "fp16":
        weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        weight_dtype = torch.float32

    # 1. Load Models
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device, dtype=weight_dtype)
    
    # Load V4 ControlNet (Standard, Pixel-space)
    # Note: trained model should have saved the config with `conditioning_channels=1`
    print(f"Loading ControlNet from {args.controlnet_path}...")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=weight_dtype,
        conditioning_channels=1, # FORCE 1-channel mode
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    ).to(device)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator.seed()

    # 2. Process Input
    # Bald Image -> Base Latents
    image_tensor = preprocess_image(args.bald_path, args.resolution).to(device, dtype=weight_dtype)
    with torch.no_grad():
        z_bald = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
    
    z_bald = z_bald.repeat(args.num_samples, 1, 1, 1)

    # Mask -> ControlNet Condition (1 channel, 512x512)
    mask_tensor = preprocess_mask(args.mask_path, args.resolution).to(device, dtype=weight_dtype)
    controlnet_cond = mask_tensor.repeat(args.num_samples, 1, 1, 1)

    # 3. Initialize Latents (Inpainting-style start)
    # We start from z_bald + noise (SDEdit style) to preserve identity
    # We start from z_bald + noise (SDEdit style) to preserve identity
    noise = torch.randn(
        z_bald.shape, 
        device=device, 
        dtype=weight_dtype, 
        generator=generator
    )
    
    # Calculate starting timestep based on noise_strength
    start_step = int(args.num_inference_steps * args.noise_strength)
    start_timestep = noise_scheduler.timesteps[start_step]
    
    # Add noise to z_bald
    latents = noise_scheduler.add_noise(z_bald, noise, start_timestep)

    # 4. Text Embeddings
    text_embeddings, uncond_embeddings = encode_texts(
        tokenizer,
        text_encoder,
        args.prompt,
        args.negative_prompt,
        device,
        args.num_samples,
    )
    text_embeddings = text_embeddings.to(dtype=weight_dtype)
    if uncond_embeddings is not None:
        uncond_embeddings = uncond_embeddings.to(dtype=weight_dtype)

    # 5. Denoising Loop
    # We only iterate from start_step to end
    timesteps_to_run = noise_scheduler.timesteps[start_step:]
    
    print(f"Starting inference from step {start_step} (Strength: {args.noise_strength})...")

    for t in tqdm(timesteps_to_run):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=device.type == "cuda"):
                
                # Expand latents for classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if args.guidance_scale > 1.0 else latents
                
                # ControlNet Forward
                # We need to pass cond for both text and unconditional branches
                controlnet_cond_input = torch.cat([controlnet_cond] * 2) if args.guidance_scale > 1.0 else controlnet_cond
                
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings]) if args.guidance_scale > 1.0 else text_embeddings,
                    controlnet_cond=controlnet_cond_input,
                    return_dict=False,
                )
                
                # Apply Control Scale
                down_block_res_samples = [res * args.controlnet_scale for res in down_block_res_samples]
                mid_block_res_sample *= args.controlnet_scale

                # UNet Forward
                noise_pred = unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings]) if args.guidance_scale > 1.0 else text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # CFG
                if args.guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Scheduler Step
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # 6. Decode
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(latents).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx in range(args.num_samples):
        img = transforms.ToPILImage()(images[idx])
        img.save(out_dir / f"sample_{timestamp}_{idx:03d}.png")

    print(f"Saved {args.num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
