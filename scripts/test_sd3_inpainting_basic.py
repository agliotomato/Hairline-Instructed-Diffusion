
import argparse
import torch
import os
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm
import torchvision # Added for blending

def parse_args():
    parser = argparse.ArgumentParser(description="Test SD3.5 Basic Inpainting (Latent Blending)")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (Bald)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image (White=Generate, Black=Keep)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/test_inpainting_basic.png")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=5.0) # SD3.5 recommended higher guidance
    parser.add_argument("--strength", type=float, default=1.0, help="Denoising strength (1.0 = Full Inpainting)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_blur", type=float, default=0.0, help="Sigma for Gaussian Blur on mask (e.g. 1.0~3.0)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading SD3.5 from {args.model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.enable_model_cpu_offload() 
    # Or pipe.to(device) if enough VRAM
    
    # Enable Tiling for consistency if needed, but not strictly required for test.
    # pipe.vae.enable_tiling()
    
    # 1. Load Images
    resolution = 1024 # SD3 native
    
    image_pil = Image.open(args.image_path).convert("RGB").resize((resolution, resolution))
    mask_pil = Image.open(args.mask_path).convert("L").resize((resolution, resolution), Image.NEAREST)
    
    # Convert Mask to Tensor (1=Generate/Hair, 0=Keep/Bg)
    # Ensure binary
    mask_np = np.array(mask_pil)
    mask_np = (mask_np > 127).astype(np.float32) 
    
    # Prepare Mask for Latents: Resize to 128x128
    mask_tensor_full = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    
    latent_res = resolution // 8
    mask_latent = torch.nn.functional.interpolate(mask_tensor_full, size=(latent_res, latent_res), mode="nearest")

    # IMPROVEMENT: Mask Blurring for Soft Blending
    if args.mask_blur > 0:
        kernel_size = 2 * int(args.mask_blur) + 1
        mask_latent = torchvision.transforms.functional.gaussian_blur(mask_latent, kernel_size=kernel_size, sigma=args.mask_blur)
        print(f"Applied Gaussian Blur to mask with sigma={args.mask_blur}")
    
    # 2. Encode Original Image (Init Latents)
    print("Encoding original image...")
    image_tensor = torch.from_numpy(np.array(image_pil)).float() / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    
    with torch.no_grad():
        init_latents = pipe.vae.encode(image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

    # 3. Prepare Scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler = scheduler
    
    scheduler.set_timesteps(args.steps, device=device)
    timesteps = scheduler.timesteps
    
    # 4. Generate Random Noise
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    
    # Initial Latents (Pure Noise)
    latents = noise 
    
    # 5. Denoising Loop with Blending
    print("Running Inpainting Loop...")
    
    # FIX: Explicitly pass prompt_2 and prompt_3 to handle SD3 pipeline requirements
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt, 
        prompt_2=args.prompt, 
        prompt_3=args.prompt, 
        negative_prompt="bad quality, ugly, distorted", 
        device=device, 
        do_classifier_free_guidance=True
    )

    # CRITICAL FIX: Concatenate embeddings for Classifier-Free Guidance (CFG)
    # The transformer expects [negative, positive] order for the batch
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    for i, t in enumerate(tqdm(timesteps)):
        # 5.1. Repainting / Blending Step
        # At this timestep t, what should the "Background" look like?
        # It should be the Original Image + Noise corresponding to level t.
        # Flow Matching Logic:
        # z_t = (1 - sigma) * x_0 + sigma * epsilon
        
        # Get sigma for current timestep
        sigma = scheduler.sigmas[i]
        sigma = sigma.to(device, dtype=dtype)
        
        # Create noisy version of original image at this step
        # Note: We need a noise sample. Let's use a fixed generator or random.
        # Flow Matching is deterministic ODE usually.
        # Let's try: Noisy Background = (1-sigma)*x0 + sigma*noise_fixed
        noise_bg = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype) 
        
        init_latents_t = (1.0 - sigma) * init_latents + sigma * noise_bg
        
        # BLENDING:
        # Latents = Mask * Latents_Predicted + (1 - Mask) * Init_Latents_T
        # Mask=1 (Hair) -> Use Predicted
        # Mask=0 (Bg)   -> Use Init_T
        latents = mask_latent * latents + (1.0 - mask_latent) * init_latents_t
        
        # 5.2. Standard Step
        with torch.no_grad():
            latent_model_input = torch.cat([latents] * 2)
            
            # Broadcast timestep
            batch_size = latent_model_input.shape[0]
            current_timestep = t.expand(batch_size) if isinstance(t, torch.Tensor) and t.ndim == 0 else torch.tensor([t] * batch_size, device=device)

            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=current_timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Step
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # 6. Decode
    print("Decoding...")
    latents = (latents / pipe.vae.config.scaling_factor)
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
        
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    image_pil.save(args.output_path)
    print(f"Saved result: {args.output_path}")

if __name__ == "__main__":
    main()
