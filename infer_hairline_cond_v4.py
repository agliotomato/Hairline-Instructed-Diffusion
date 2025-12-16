from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, ControlNetModel as PixelControlNet, MultiControlNetModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
from tqdm.auto import tqdm

# Import Custom Latent ControlNet
from utils.latent_identity_net import ControlNetModel as LatentControlNet


def preprocess_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.BILINEAR)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    return transform(image).unsqueeze(0)


def preprocess_conditions(mask_path: str, bald_path: str, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Geometry Condition: High-Res Mask (1-channel)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    mask_tensor = transforms.ToTensor()(mask).unsqueeze(0) # [1, 1, H, W]
    mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
    
    # 2. Identity Condition: Masked Bald Image (3-channel) for Latent Encoding
    # Bald Image * (1 - Mask)
    bald = Image.open(bald_path).convert("RGB")
    bald = bald.resize((resolution, resolution), Image.BILINEAR)
    bald_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])(bald).unsqueeze(0) # [1, 3, H, W]
    
    # Apply Masking: Hair area (mask=1) becomes Black (-1 in normalized space)
    masked_bald_tensor = bald_tensor * (1.0 - mask_tensor) + (-1.0) * mask_tensor
    
    return mask_tensor, masked_bald_tensor


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
    parser = argparse.ArgumentParser(description="Inference for the Hybrid V4 (Pixel+Latent) Dual-Stream Adapter.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_path", type=str, required=True, help="Path to trained MultiControlNet weights (directory).")
    parser.add_argument("--bald_path", type=str, required=True, help="Path to an aligned bald image (used as base canvas AND Identity condition).")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the forehead mask (Geometry condition).")
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic, detailed hair")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, artificial")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_scales", type=float, nargs='+', default=[1.0, 1.0], help="Scales for [Geometry, Identity].")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--noise_strength", type=float, default=0.9)
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
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    
    # Load Hybrid MultiControlNet
    # NOTE: Since `MultiControlNetModel` loads a LIST of ControlNets, and they have different classes (Pixel vs Latent)
    # auto-loading from a single `from_pretrained` might be tricky if the config doesn't specify the class exactly or if `diffusers` doesn't know about `LatentControlNet`.
    # However, if we saved it as a MultiControlNet, the `model_index.json` or config should help.
    # BUT, `LatentControlNet` is a custom class. `diffusers` generic loading might fail to instantiate it.
    # Safe bet: Load them manually assuming structure.
    # Structure usually: controlnet_path/controlnet, or just controlnet_path if it contains config.json?
    # If saved via `accelerator.unwrap_model(controlnet).save_pretrained(...)`, it likely saved a pipeline-compatible folder structure or just the model.
    # If it's a MultiControlNet, it saves as a list of configs.
    
    print(f"Loading Hybrid MultiControlNet components from {args.controlnet_path}...")
    
    # Path setup based on user observation (controlnet, controlnet_1)
    # We allow args.controlnet_path to be the parent dir containing these, or explicit paths.
    # Assumption: args.controlnet_path points to "hairline_cond_v4_hybrid"
    base_path = Path(args.controlnet_path)
    
    # Try to locate the two controlnets
    # Strategy 1: Look for "controlnet" and "controlnet_1" subfolders
    path_a = base_path / "controlnet"
    path_b = base_path / "controlnet_1"
    
    if not path_a.exists():
        # Maybe the user pointed directly to "controlnet"? then where is "controlnet_1"?
        # Let's fallback to assuming the user provided a directory that MIGHT capture them or standard loading.
        # But per instruction, we expect these two.
        print(f"Warning: {path_a} not found. Trying standard load from {base_path}...")
        try:
             controlnet = MultiControlNetModel.from_pretrained(args.controlnet_path).to(device)
        except Exception as e:
             raise ValueError(f"Could not load MultiControlNet from {args.controlnet_path}. Ensure 'controlnet' and 'controlnet_1' folders exist or path is correct. Error: {e}")
    else:
        print(f"Loading Stream A (Geometry) from {path_a}")
        cnet_a = PixelControlNet.from_pretrained(path_a).to(device)
        
        print(f"Loading Stream B (Identity) from {path_b}")
        # Validate path_b
        if not path_b.exists():
             raise ValueError(f"Found 'controlnet' but not 'controlnet_1' in {base_path}. Both are required for Hybrid V4.")
             
        cnet_b = LatentControlNet.from_pretrained(path_b).to(device)
        
        controlnet = MultiControlNetModel([cnet_a, cnet_b])
        print("Successfully combined separate ControlNets.")

    controlnet.to(device)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator.seed()

    # 2. Process Input
    # Base Latents (from Bald Image)
    image_tensor = preprocess_image(args.bald_path, args.resolution).to(device)
    with torch.no_grad():
        z_bald = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
    z_bald = z_bald.repeat(args.num_samples, 1, 1, 1)

    # Prepare Conditions
    mask_tensor, masked_bald_tensor = preprocess_conditions(args.mask_path, args.bald_path, args.resolution)
    
    # 2.1 [Hybrid] Encode Identity Condition to Latents
    masked_bald_tensor = masked_bald_tensor.to(device)
    with torch.no_grad():
         masked_bald_latents = vae.encode(masked_bald_tensor).latent_dist.sample() * vae.config.scaling_factor
    
    masked_bald_latents = masked_bald_latents.repeat(args.num_samples, 1, 1, 1)
    mask_tensor = mask_tensor.to(device).repeat(args.num_samples, 1, 1, 1)
    
    # MultiControlNet Condition: List [Geometry(Pixel), Identity(Latent)]
    controlnet_cond = [mask_tensor, masked_bald_latents]

    # 3. Initialize Latents
    noise = torch.randn(z_bald.shape, device=device, generator=generator)
    start_step = int(args.num_inference_steps * args.noise_strength)
    start_timestep = noise_scheduler.timesteps[start_step]
    latents = noise_scheduler.add_noise(z_bald, noise, start_timestep)

    # 4. Text Embeddings
    text_embeddings, uncond_embeddings = encode_texts(
        tokenizer, text_encoder, args.prompt, args.negative_prompt, device, args.num_samples
    )
    # text_embeddings = text_embeddings.to(dtype=weight_dtype) # Keep in FP32
    # if uncond_embeddings is not None:
    #     uncond_embeddings = uncond_embeddings.to(dtype=weight_dtype)

    # 5. Denoising Loop
    timesteps_to_run = noise_scheduler.timesteps[start_step:]
    print(f"Starting inference from step {start_step} (Strength: {args.noise_strength})...")
    
    control_scales = args.controlnet_scales if len(args.controlnet_scales) == 2 else [1.0, 1.0]

    for t in tqdm(timesteps_to_run):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=device.type == "cuda"):
                
                # CFG expansion
                latent_model_input = torch.cat([latents] * 2) if args.guidance_scale > 1.0 else latents
                
                # Expand conditions for CFG: List of Tensors
                # Each tensor needs to be doubled if CFG > 1.0
                if args.guidance_scale > 1.0:
                    controlnet_cond_input = [torch.cat([c] * 2) for c in controlnet_cond]
                else:
                    controlnet_cond_input = controlnet_cond
                
                # MultiControlNet Forward
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings]) if args.guidance_scale > 1.0 else text_embeddings,
                    controlnet_cond=controlnet_cond_input,
                    # conditioning_scale=control_scales, # MultiControlNet might handle list of scales? 
                    # checking API: forward doesn't take list of scales usually in `diffusers` versions <= 0.20?
                    # Generally it returns summed residuals. 
                    # If we want scaling, we multiply the output. 
                    # But wait, MultiControlNetModel forward sums them internally.
                    # It accepts `conditioning_scale` which can be a list.
                    conditioning_scale=control_scales,
                    return_dict=False,
                )

                # UNet Forward
                noise_pred = unet(
                    sample=latent_model_input,
                    timestep=t,
                    encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings]) if args.guidance_scale > 1.0 else text_embeddings,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                if args.guidance_scale > 1.0:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

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
