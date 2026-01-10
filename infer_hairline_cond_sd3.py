
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

class HybridSD3Pipeline(StableDiffusion3ControlNetPipeline):
    def prepare_control_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        # Override to strict bypass VAE if input is Tensor of correct internal shape
        # Standard SD3 ControlNet input: [B, C, H, W]
        
        # Helper to bypass VAE for a single image item
        def _check_bypass(img):
            if isinstance(img, torch.Tensor):
                # If it's 1-channel (Mask) or 16-channel (Latent), we assume it's PREPARED.
                # Standard RGB "image" would be 3-channel.
                if img.shape[1] == 1 or img.shape[1] == 16:
                    return True
            return False

        # If list, check first item?
        is_bypass = False
        if isinstance(image, list):
            if _check_bypass(image[0]):
                is_bypass = True
        elif _check_bypass(image):
            is_bypass = True
            
        if is_bypass:
            # Bypass VAE, just process batch/device
            control_image = image
            if not isinstance(control_image, list):
                control_image = [control_image]
            
            # Align batch size logic (repeat if needed)
            # Simplified: We assume user passed batch size 1 or matching.
            # Just move to device/dtype
            prepared_images = []
            for img in control_image:
                img = img.to(device=device, dtype=dtype)
                # SD3 Pipeline expects control_image to be duplicated for per-prompt?
                # Standard implementation duplicates it by match batch_size * num_images_per_prompt
                if do_classifier_free_guidance: # SD3CFG usually doesn't double control input? 
                    # Actually SD3 uses `joint_attention`, so BS=2 usually (Prompt + Neg).
                    # If passed image is BS=1, we need to repeat?
                    if img.shape[0] < batch_size * num_images_per_prompt:
                         img = torch.cat([img] * (batch_size * num_images_per_prompt), dim=0)
                prepared_images.append(img)
                
            if len(prepared_images) == 1:
                return prepared_images[0]
            return prepared_images
            
        # Fallback to original
        return super().prepare_control_image(
            image, width, height, batch_size, num_images_per_prompt, device, dtype, 
            do_classifier_free_guidance, guess_mode
        )

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # Load ControlNets
    print("Loading ControlNets...")
    controlnet_a = SD3ControlNetModel.from_pretrained(args.controlnet_a_path, torch_dtype=dtype)
    controlnet_b = SD3ControlNetModel.from_pretrained(args.controlnet_b_path, torch_dtype=dtype)
    
    # Load Pipeline
    print("Loading Pipeline...")
    # Use Custom Hybrid Pipeline
    pipe = HybridSD3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=[controlnet_a, controlnet_b],
        torch_dtype=dtype
    )
    pipe.to(device)
    
    # Preprocess Inputs
    resolution = 1024 
    
    # 1. Bald Image (for Identity)
    bald_pil = Image.open(args.bald_image).convert("RGB").resize((resolution, resolution))
    
    # Encode Identity to Latents (16ch)
    print("Encoding Identity Image...")
    bald_tensor = transforms.ToTensor()(bald_pil).unsqueeze(0).to(device, dtype=dtype)
    bald_tensor = (bald_tensor - 0.5) / 0.5
    with torch.no_grad():
        identity_latents = pipe.vae.encode(bald_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # 2. Mask (for Geometry)
    # 1-channel Mask
    mask_pil = Image.open(args.mask_image).convert("L")
    mask_tensor = preprocess_mask_smart(mask_pil, resolution).to(device, dtype=dtype)
    
    # Cutout Bald Logic (Identity)
    # Apply Mask to Bald Image: masked_bald = bald * (1-mask) + (-1)*mask.
    # We use sharp mask for cutout
    hair_raw = (np.array(mask_pil.resize((resolution, resolution), Image.NEAREST)) == 255).astype(np.float32)
    hair_tensor = torch.from_numpy(hair_raw).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    
    masked_bald_tensor = bald_tensor * (1.0 - hair_tensor) + (-1.0) * hair_tensor
    with torch.no_grad():
        identity_latents = pipe.vae.encode(masked_bald_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # List of Conditions [Mask(1ch), IdentityLatents(16ch)]
    control_images = [mask_tensor, identity_latents] 
    
    print("Generating...")
    # Using 'control_image' (singular) argument name as per standard pipeline, but passing list.
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
