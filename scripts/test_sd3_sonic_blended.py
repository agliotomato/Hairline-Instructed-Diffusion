import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
import torchvision

def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 SONIC + Latent Blending Inpainting")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (Bald)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image (White=Hair, Black=Bg)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/sonic/test_sonic_blended.png")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--mask_blur", type=float, default=3.0)
    # SONIC Params
    parser.add_argument("--opt_steps", type=int, default=10, help="Optimization steps (10-20 enough for hybrid)")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Force Float32 for Optimization Stability, then switch back if needed? 
    # Actually SD3 runs fine in float32, let's keep it safe.
    dtype = torch.float32 

    print(f"Loading SD3.5 from {args.model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.to(device)
    
    # ---------------------------------------------------------
    # 1. Prepare Data (Image, Mask, Latents)
    # ---------------------------------------------------------
    resolution = 1024
    image_pil = Image.open(args.image_path).convert("RGB").resize((resolution, resolution))
    mask_pil = Image.open(args.mask_path).convert("L").resize((resolution, resolution), Image.NEAREST)
    
    mask_np = np.array(mask_pil)
    mask_np = (mask_np > 127).astype(np.float32)
    
    # Latent Mask
    mask_tensor_full = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    latent_res = resolution // 8
    mask_latent = F.interpolate(mask_tensor_full, size=(latent_res, latent_res), mode="nearest")
    
    # Blur Mask for Soft Blending
    if args.mask_blur > 0:
        k = 2 * int(args.mask_blur) + 1
        mask_latent = torchvision.transforms.functional.gaussian_blur(mask_latent, kernel_size=k, sigma=args.mask_blur)
    
    # Background Mask (for SONIC Loss): 1 for Background, 0 for Hair
    bg_mask = 1.0 - mask_latent

    # Encode Image -> Init Latents (Target for Constraint)
    print("Encoding original image...")
    image_tensor = torch.from_numpy(np.array(image_pil)).float() / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    
    with torch.no_grad():
        init_latents = pipe.vae.encode(image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

    # ---------------------------------------------------------
    # 2. SONIC Seed Optimization (The "Pre-Step")
    # ---------------------------------------------------------
    print(f"--- Phase 1: SONIC Seed Optimization ({args.opt_steps} steps) ---")
    
    # Initialize Random Noise
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sonic_noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    sonic_noise.requires_grad_(True)
    
    # Freeze Model
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    # Encode Prompt for Optimization (Use Positive Prompt to guide structure if needed)
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=args.prompt, prompt_2=args.prompt, prompt_3=args.prompt,
            negative_prompt="bad quality", device=device, do_classifier_free_guidance=True
        )
    # Detach to prevent graph issues
    prompt_embeds = prompt_embeds.detach()
    pooled_prompt_embeds = pooled_prompt_embeds.detach()
    
    optimizer = torch.optim.Adam([sonic_noise], lr=args.lr)
    
    # Setup Scheduler for t_start
    pipe.scheduler.set_timesteps(args.steps, device=device)
    t_start = pipe.scheduler.timesteps[0]
    
    for i in tqdm(range(args.opt_steps)):
        optimizer.zero_grad()
        
        # 2-1. Single Step Flow Matching Prediction
        batch_size = sonic_noise.shape[0]
        t_tensor = torch.tensor([t_start] * batch_size, device=device)
        
        noise_pred = pipe.transformer(
            hidden_states=sonic_noise,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds, # Guidance
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]
        
        # 2-2. Linear Approx: pred_x0 = x_t - v (assuming t=1)
        pred_x0 = sonic_noise - noise_pred
        
        # 2-3. Masked Loss: Match Background Only
        loss = F.mse_loss(pred_x0 * bg_mask, init_latents * bg_mask)
        
        # 2-4. Update
        loss.backward()
        torch.nn.utils.clip_grad_norm_([sonic_noise], max_norm=0.1) # Stability
        optimizer.step()
        
        if i % 2 == 0:
            print(f"Opt Step {i}: Loss {loss.item():.6f}")

    optimized_latents = sonic_noise.detach()
    print("Optimization Complete. Starting Generation...")

    # ---------------------------------------------------------
    # 3. Latent Blending Generation (The "Main Loop")
    # ---------------------------------------------------------
    print(f"--- Phase 2: Latent Blending Generation ({args.steps} steps) ---")
    
    # Re-encode prompt with joined embeddings for CFG
    # Note: previous encode_prompt returns separated embeddings.
    # SD3 requires concatenation [negative, positive]
    prompt_embeds_full = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    pooled_prompt_embeds_full = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    
    latents = optimized_latents # Start with our SONIC noise
    
    timesteps = pipe.scheduler.timesteps
    num_steps = len(timesteps)
    
    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            # 3-1. Expand latents for CFG
            latent_model_input = torch.cat([latents] * 2)
            timestep = torch.tensor([t] * 2, device=device)

            # 3-2. Predict Noise
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds_full,
                pooled_projections=pooled_prompt_embeds_full,
                return_dict=False
            )[0]

            # 3-3. Perform CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # 3-4. Compute Previous Latents (Update Step)
            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = latents.to(latents_dtype)

            # -------------------------------------------------------
            # 3-5. LATENT BLENDING (The "Glue")
            # -------------------------------------------------------
            # Inject Noise into Original Image at current timestep t
            # For FlowMatching/RectifiedFlow: t goes 1.0 -> 0.0
            # z_t = t * noise + (1-t) * image
            if i < num_steps - 1:
                # Calculate sigma/t for mixing
                # In Diffusers SD3 scheduler, 'sigmas' track noise level.
                # Usually t corresponds to noise level in RF.
                sigma = pipe.scheduler.sigmas[i+1] if hasattr(pipe.scheduler, 'sigmas') else t/1000.0
                
                # We need fresh noise for blending?
                # Actually, standard blending uses fixed noise x_T mixed with x_0.
                # But here we just use the simple interpolation formula of the scheduler.
                
                # Simple Manual Addition for RF:
                # init_latents_t = (1-t) * init_latents + t * noise
                # But 't' in scheduler is 1000-scale.
                # Let's trust the scheduler's sigma.
                
                # Generate noise for background mixing
                noise_bg = torch.randn_like(init_latents)
                init_latents_t = (1.0 - sigma) * init_latents + sigma * noise_bg

                # BLEND: Mask=1 (Gen), Mask=0 (Bg)
                # latents = Mask * latents + (1-Mask) * init_latents_t
                latents = mask_latent * latents + (1.0 - mask_latent) * init_latents_t

    # 4. Decode Final Image
    print("Decoding result...")
    latents = (latents / pipe.vae.config.scaling_factor)
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(image[0]).save(args.output_path)
    print(f"Saved SONIC-Blended result to {args.output_path}")

if __name__ == "__main__":
    main()
