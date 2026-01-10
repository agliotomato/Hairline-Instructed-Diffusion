
import argparse
import torch
import os
from PIL import Image
import numpy as np
import cv2
from diffusers import (
    StableDiffusion3ControlNetPipeline,
    SD3ControlNetModel,
    MultiControlNetModel,
    FlowMatchEulerDiscreteScheduler
)
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for SD3 Hybrid ControlNet")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--controlnet_a_path", type=str, default="output/hairline_sd3_run1/controlnet_a", help="Path to Geometry ControlNet (1ch)")
    parser.add_argument("--controlnet_b_path", type=str, default="output/hairline_sd3_run1/controlnet_b", help="Path to Identity ControlNet (16ch)")
    parser.add_argument("--bald_image", type=str, required=True, help="Path to Bald Image")
    parser.add_argument("--mask_image", type=str, required=True, help="Path to Hair Mask (White=Hair)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--scale_geometry", type=float, default=1.0)
    parser.add_argument("--scale_identity", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def preprocess_mask_smart(mask_pil, resolution=1024):
    """
    Apply Smart Blur logic (Edge-aware) for Inference.
    Similar to training logic but fixed (no random).
    """
    mask_pil = mask_pil.resize((resolution, resolution), Image.NEAREST)
    mask_np = np.array(mask_pil)
    
    # 255: Hair, 127: Face, 0: Background
    # If mask is binary (0/255), treat 0 as bg.
    # We assume the user provides the generated mask (0,127,255).
    # If not, we might need fallback.
    
    unique_vals = np.unique(mask_np)
    if len(unique_vals) <= 2:
        print("Warning: Input mask seems binary. Smart blur might not distinguish Face vs Background.")
        hair_raw = (mask_np > 127).astype(np.float32)
        face_raw = np.zeros_like(hair_raw) # No face info
    else:
        hair_raw = (mask_np == 255).astype(np.float32)
        face_raw = (mask_np == 127).astype(np.float32)

    # 1. Blur Hair (Soft Background)
    k_size = 15
    hair_blur = cv2.GaussianBlur(hair_raw, (k_size, k_size), 0)
    
    # 2. Face Zone
    dilate_k = 21
    face_zone = cv2.dilate(face_raw, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)
    face_zone_soft = cv2.GaussianBlur(face_zone, (15, 15), 0)
    
    # 3. Combine
    # Sharp near face, Soft near background
    # Erode sharp slightly for inference to be safe?
    # Let's keep it sharp (hair_raw).
    
    final_mask = hair_raw * face_zone_soft + hair_blur * (1.0 - face_zone_soft)
    
    return torch.from_numpy(final_mask).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # Load ControlNets
    print("Loading ControlNets...")
    controlnet_a = SD3ControlNetModel.from_pretrained(args.controlnet_a_path, torch_dtype=dtype)
    controlnet_b = SD3ControlNetModel.from_pretrained(args.controlnet_b_path, torch_dtype=dtype)
    
    multi_controlnet = MultiControlNetModel([controlnet_a, controlnet_b])
    
    # Load Pipeline
    print("Loading Pipeline...")
    # SD3 Pipeline accepts list of ControlNets directly
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=[controlnet_a, controlnet_b],
        torch_dtype=dtype
    )
    pipe.to(device)
    
    # Preprocess Inputs
    resolution = 1024 # SD3 standard
    
    # 1. Bald Image (for Identity)
    bald_pil = Image.open(args.bald_image).convert("RGB").resize((resolution, resolution))
    # We need to compute Latents for Identity Net!
    # The pipeline.encode_image functionality might not expose VAE encoding for ControlNet input easily if expecting pixel values.
    # SD3ControlNetPipeline expects 'control_image' argument.
    # For ControlNet B (Identity), it expects 16-channel latents?
    # Standard Pipeline expects Pixel Images and encodes them internally for ControlNet IF extra_conditioning_channels=3.
    # But our ControlNet B has extra_conditioning_channels=16.
    # The pipeline MIGHT crash if we pass an image to a 16-channel ControlNet.
    # Solution: We must pass PRE-ENCODED LATENTS as control condition?
    # Or does the pipeline handle VAE encoding for us?
    # Looking at Diffusers source:
    # prepare_control_image -> if model input is latent, it encodes?
    # Actually, default SD3 ControlNet takes VAE latents (masked image) + Canny/etc.
    # Standard SD3 ControlNet takes RGB image -> VAE -> Latent?
    # No, SD3 ControlNet usually takes "Structure" (Canny/Pose) which might be encoded by VAE or small conv.
    # BUT, if we use `extra_conditioning_channels=16`, it implies we want VAE Latents.
    # The pipeline *likely* doesn't support automatic VAE encoding for custom 16-ch ControlNet out of the box unless we hack it.
    # OR, we pass the 16-ch Latent Tensor directly as `control_image`.
    
    # Let's Encode Manually
    print("Encoding Identity Image...")
    bald_tensor = transforms.ToTensor()(bald_pil).unsqueeze(0).to(device, dtype=dtype)
    bald_tensor = (bald_tensor - 0.5) / 0.5
    with torch.no_grad():
        identity_latents = pipe.vae.encode(bald_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # 2. Mask (for Geometry)
    # 1-channel, but Pipeline usually expects 3-channel RGB for ControlNet.
    # But our ControlNet A has extra_conditioning_channels=1.
    # We should pass 1-channel tensor.
    mask_pil = Image.open(args.mask_image).convert("L")
    mask_tensor = preprocess_mask_smart(mask_pil, resolution).to(device, dtype=dtype)
    
    # Cutout Bald Image using Sharp Mask for Identity Input
    # Identity Latent should be "Masked Bald".
    # Logic: Identity Latent is Full Bald Image.
    # Wait, in training, we passed "Masked Bald".
    # Here we passed "Full Bald".
    # We should apply mask to bald image BEFORE encoding?
    # Training: masked_bald = bald * (1-mask) + (-1)*mask.
    # Yes, we should apply mask to bald image here too.
    
    hair_raw = (np.array(mask_pil.resize((resolution, resolution), Image.NEAREST)) == 255).astype(np.float32)
    hair_tensor = torch.from_numpy(hair_raw).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    
    masked_bald_tensor = bald_tensor * (1.0 - hair_tensor) + (-1.0) * hair_tensor
    with torch.no_grad():
        identity_latents = pipe.vae.encode(masked_bald_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # List of Conditions
    # Pipeline expects list if MultiControlNet
    control_images = [mask_tensor, identity_latents] 
    
    print("Generating...")
    image = pipe(
        prompt=args.prompt,
        control_image=control_images,
        controlnet_conditioning_scale=[args.scale_geometry, args.scale_identity],
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.manual_seed(args.seed),
        height=resolution,
        width=resolution
    ).images[0]
    
    image.save(args.output_path)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
