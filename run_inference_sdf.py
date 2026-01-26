
import argparse
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
import sys

# Fix imports
sys.path.append(os.getcwd())
from modules.tiny_adapter_native import TinyAdapterNative

def compute_sdf(mask_pil: Image.Image, tau: float = 5.0):
    """
    Convert Binary Mask -> Normalized SDF [-1, 1]
    """
    mask_np = np.array(mask_pil)
    # Binary: 255=Inside, 0=Outside
    binary = (mask_np > 127).astype(np.uint8)
    
    # Distance Transforms
    dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    
    # SDF
    sdf = dist_in - dist_out
    
    # Tanh Normalization
    sdf_norm = np.tanh(sdf / tau)
    
    return torch.from_numpy(sdf_norm).float()

def main():
    parser = argparse.ArgumentParser(description="Inference with TinyAdapterNative (SDF Strategy)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to bald/original image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to binary mask")
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic hair, detailed texture")
    parser.add_argument("--output_path", type=str, default="results/sdf_test/output.png")
    parser.add_argument("--adapter_path", type=str, required=True, help="Checkpoints path")
    parser.add_argument("--tau", type=float, default=5.0, help="SDF Softness. 2.0=Sharp, 10.0=Soft")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--scale", type=float, default=1.0, help="Adapter Strength")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend_pixels", action="store_true", help="Composite result with original image")
    
    args = parser.parse_args()
    
    # Logs
    print(f"--- SDF Inference ---")
    print(f"Image: {args.image_path}")
    print(f"Mask: {args.mask_path} (Tau={args.tau})")
    print(f"Model: {args.adapter_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 1. Load SD3.5
    print("Loading SD 3.5 Pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16
    ).to(device)
    
    # 2. Load Adapter
    print("Loading Adapter...")
    adapter = TinyAdapterNative(input_channels=1, base_channels=32, output_channels=16)
    # Load state dict safely
    state_dict = torch.load(args.adapter_path, map_location=device, weights_only=True)
    adapter.load_state_dict(state_dict)
    adapter.to(device, dtype=torch.bfloat16)
    adapter.eval()
    
    # 3. Prepare Inputs
    # Load Image
    img_pil = Image.open(args.image_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    img_tensor = (transforms.ToTensor()(img_pil).unsqueeze(0).to(device) * 2.0 - 1.0).to(torch.bfloat16)
    
    # Load Mask & Compute SDF
    mask_pil = Image.open(args.mask_path).convert("L").resize((1024, 1024), Image.NEAREST)
    # Fix: ensure 4D tensor (B, C, H, W) for interpolation
    sdf_tensor = compute_sdf(mask_pil, tau=args.tau).unsqueeze(0).unsqueeze(0).to(device).to(torch.bfloat16)
    
    # Compute Adapter Features
    with torch.no_grad():
        adapter_features = adapter(sdf_tensor) * args.scale
        
    # Prepare Latent Mask for Blending
    # Map [-1, 1] -> [0, 1]
    mask_01 = (sdf_tensor + 1.0) / 2.0
    # Downscale to 128x128
    mask_latent = F.interpolate(mask_01, size=(128, 128), mode="bilinear", align_corners=False)
    
    # Encode Image (Background)
    with torch.no_grad():
        bg_latents = pipe.vae.encode(img_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
        
    # Clean Latents (Start Noise)
    generator = torch.Generator(device).manual_seed(args.seed)
    latents = torch.randn(bg_latents.shape, generator=generator, device=device, dtype=bg_latents.dtype)
    
    # Encode Prompt
    (prompt_embeds, negative_prompt_embeds, 
     pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
        prompt=args.prompt,
        prompt_2=args.prompt,
        prompt_3=args.prompt,
        negative_prompt="bad quality, distorted, blurry, ugly",
        device=device,
        do_classifier_free_guidance=True
    )
    
    # 4. Denoising Loop
    print("Generating...")
    pipe.scheduler.set_timesteps(args.steps)
    
    for i, t in enumerate(pipe.scheduler.timesteps):
        # Latent Blending: Preserve background
        # Mix noisy background into current latents
        if i < len(pipe.scheduler.timesteps) - 1:
            # Add noise to background for current timestep
            # In SD3 Flow Match, sigmas go 1.0 -> 0.0
            # We can just linearly interpolate for "Noisy Background"
            # Or simplified: Paste background only at last steps? 
            # No, standard Inpainting: Blend at every step.
            
            # Simple Inpainting Blending:
            # x_t = M * x_t + (1-M) * x_t_background
            # But x_t_background needs to be noisy.
            # SD3 Scheduler usually provides 'sigmas'.
            sigma = pipe.scheduler.sigmas[i]
            noise = torch.randn(bg_latents.shape, generator=generator, device=device, dtype=bg_latents.dtype) # Fixed noise for consistency?
            # Ideally use same noise as initial or new? standard is new noise scaled.
            # Flow Matching: x_t = (1-sigma)*x_0 + sigma*noise
            # Note: SD3 Scheduler sigma is 1.0(noise) -> 0.0(clean)
            
            bg_noisy = (1 - sigma) * bg_latents + sigma * noise
            
            latents = latents * mask_latent + bg_noisy * (1 - mask_latent)

        # Expand for CFG
        latent_model_input = torch.cat([latents] * 2)
        # Add Adapter Features
        model_input = latent_model_input + torch.cat([adapter_features] * 2)
        
        # Predict
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=t.unsqueeze(0).expand(2).to(device),
            encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]),
            pooled_projections=torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds]),
            return_dict=False
        )[0]
        
        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.cfg * (noise_pred_text - noise_pred_uncond)
        
        # Step
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
    # 5. Decode
    print("Decoding...")
    with torch.no_grad():
        latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image = pipe.vae.decode(latents, return_dict=False)[0]
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
    result_pil = Image.fromarray((image * 255).round().astype("uint8"))
    
    # Pixel Blending (Optional)
    if args.blend_pixels:
        print("Applying Pixel Blending...")
        mask_vis = Image.fromarray((mask_01.squeeze().cpu().numpy() * 255).astype(np.uint8)).resize(result_pil.size)
        result_pil = Image.composite(result_pil, img_pil, mask_vis)
        
    result_pil.save(args.output_path)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
