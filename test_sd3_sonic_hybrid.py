import argparse
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os

# Import our TinyAdapter
from modules.tiny_adapter import TinyAdapter

def main():
    parser = argparse.ArgumentParser(description="SONIC + TinyAdapter + Latent Blending for SD3.5")
    parser.add_argument("--image_path", type=str, required=True, help="Path to bald image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to semantic mask")
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic hair, detailed texture")
    parser.add_argument("--output_path", type=str, default="results/sonic_hybrid/output.png")
    parser.add_argument("--adapter_path", type=str, default="output/tiny_adapter_checkpoints/tiny_adapter_final.pth")
    parser.add_argument("--sonic_steps", type=int, default=15, help="Number of SONIC optimization steps")
    parser.add_argument("--sonic_lr", type=float, default=0.001)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    
    args = parser.parse_args()
    
    # Ensure dirs
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 1. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Load SD3.5 in bfloat16 to save memory (Fixes OOM on 40GB A100)
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16 
    ).to(device)
    
    # Load TinyAdapter
    print(f"Loading TinyAdapter from {args.adapter_path}")
    adapter = TinyAdapter(input_channels=1, output_channels=16)
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=device))
    adapter.to(device, dtype=torch.bfloat16)
    adapter.eval()
    
    # 2. Helper Functions
    def load_image(path, size=(1024, 1024)):
        img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
        # return as bf16
        return (transforms.ToTensor()(img).unsqueeze(0).to(device) * 2.0 - 1.0).to(torch.bfloat16)

    def load_mask(path, size=(1024, 1024)):
        mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
        mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
        return (mask_tensor > 0.5).float().to(torch.bfloat16)

    # 3. Prepare Data
    print("Preparing Data...")
    # Inputs are bf16
    original_image = load_image(args.image_path)    
    mask_highres = load_mask(args.mask_path)        
    
    # Encode Original Image (y)
    with torch.no_grad():
        init_latents = pipe.vae.encode(original_image).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # Prepare Mask for Latent Blending
    mask_latent = F.interpolate(mask_highres, size=init_latents.shape[-2:], mode="nearest")
    
    # Prepare Mask for Adapter
    mask_for_adapter = mask_latent.clone()
    
    # Get Adapter Features
    with torch.no_grad():
        adapter_features = adapter(mask_for_adapter) # [1, 16, 128, 128]
    
    # Encode Prompt
    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt=args.prompt, 
            prompt_2=args.prompt,
            prompt_3=args.prompt,
            negative_prompt="bad quality, ugly", 
            device=device
        )

    # ---------------------------------------------------------
    # Phase 1: SONIC (Seed Optimization)
    # ---------------------------------------------------------
    print("Starting Phase 1: SONIC Optimization...")
    
    # Init Noise (Optimize in float32 for precision)
    init_noise = torch.randn_like(init_latents, dtype=torch.float32)
    
    init_noise.requires_grad = True
    optimizer = torch.optim.Adam([init_noise], lr=args.sonic_lr)
    
    # Freeze minimal models for SONIC loop
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    # SONIC Loop
    pbar = tqdm(range(args.sonic_steps), desc="SONIC")
    for i in pbar:
        # Cast init_noise to bf16 for model forward
        init_noise_bf16 = init_noise.to(torch.bfloat16)
        
        t_val = 999 
        t = torch.tensor([t_val], device=device)
        
        # Inject Adapter
        model_input = init_noise_bf16 + adapter_features
        
        model_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]
        
        # Linear Approximation: pred_x0 = x1 - v
        # Cast back to float32 for loss calculation
        pred_x0 = init_noise - model_pred.to(torch.float32)
        
        # Masked Loss (compare with float32 latents)
        target_latents = init_latents.to(torch.float32)
        mask_float = mask_latent.to(torch.float32)
        
        loss = F.mse_loss(
            pred_x0 * (1 - mask_float), 
            target_latents * (1 - mask_float)
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([init_noise], max_norm=0.1)
        optimizer.step()
        
        pbar.set_postfix({"Loss": loss.item()})
        
    init_noise.requires_grad = False
    print("SONIC Optimization Complete.")
    
    # ---------------------------------------------------------
    # Phase 2: Generation with TinyAdapter + Latent Blending
    # ---------------------------------------------------------
    print("Starting Phase 2: Hybrid Generation...")
    
    # Setup Scheduler
    pipe.scheduler.set_timesteps(args.inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Start Latents
    # SONIC output was float32. Cast to bf16 for Generation loop.
    latents = init_noise.detach().clone().to(torch.bfloat16)
    
    # Manual Denoising Loop
    for i, t in enumerate(progressbar := tqdm(timesteps)):
        # 1. Expand Latents for CFG
        # [latents, latents] for [neg, pos]
        latent_model_input = torch.cat([latents] * 2)
        
        # 2. Add Adapter Injection
        # Add to both? Usually yes.
        # Adapter features are from the mask.
        adapter_input = torch.cat([adapter_features] * 2)
        
        model_input = latent_model_input + adapter_input
        
        # 3. Setup Prompt Embeds
        prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        # 4. Predict
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=torch.tensor([t], device=device).unsqueeze(0).repeat(2),
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_prompt_embeds_input,
            return_dict=False
        )[0]
        
        # 5. CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # 6. Step (Scheduler)
        # Scheduler returns a tuple, usually (prev_sample, )
        step_output = pipe.scheduler.step(noise_pred, t, latents)
        latents = step_output.prev_sample
        
        # 7. Latent Blending (The Safety Net)
        # Enforce background preservation explicitly at every step
        
        # We need "Noisy Original" at this timestep t
        # z_t_background = Noisy(init_latents, t)
        if i < len(timesteps) - 1:
            next_t = timesteps[i+1] # Next timestep in the schedule
            
            # Add noise to original latents to match the noise level of next_t
            # Note: scheduler.add_noise needs sigma/alpha stuff.
            # Easiest way with Flow Match: 
            # interpolate(init_latents, random_noise, t_sigma)
            
            # Since we don't have the exact noise instance the scheduler assumes (it's stochastic ODE in some samplers, but typically deterministic flow),
            # we typically add fresh noise.
            # But consistent noise is better.
            
            # Let's generate a noise instance once
            if i == 0:
                 noise_bg = torch.randn_like(init_latents)
            
            # Get Sigma for next_t
            # Use scheduler logic if possible, or manual interpolation formula
            # pipe.scheduler has sigmas.
            
            # Simpler approach: Use the same scheduler to add noise
            # But scheduler.add_noise takes 'original_samples' and 'noise' and 'timesteps'
            
            latents_bg = pipe.scheduler.add_noise(
                init_latents, 
                noise_bg, 
                torch.tensor([next_t], device=device)
            )
            
            # Blend: Keep Generated in Mask (1), Keep Background in (1-Mask)
            # Soft Blending? We have mask_latent.
            latents = latents * mask_latent + latents_bg * (1 - mask_latent)
            
    # Decode Final
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    
    # Save
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")[0]
    Image.fromarray(image).save(args.output_path)
    print(f"Saved result to {args.output_path}")

if __name__ == "__main__":
    main()
