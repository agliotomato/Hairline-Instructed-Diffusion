import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from tqdm.auto import tqdm
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 SONIC Seed Optimization Prototype")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image (Bald)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to mask image (White=Hair, Black=Bg)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="output/sonic/test_sonic.png")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--opt_steps", type=int, default=30, help="Number of optimization steps for SONIC")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for noise optimization")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading SD3.5 from {args.model_id}...")
    # Load pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    
    # 1. Prepare Images & Masks
    resolution = 1024
    image_pil = Image.open(args.image_path).convert("RGB").resize((resolution, resolution))
    mask_pil = Image.open(args.mask_path).convert("L").resize((resolution, resolution), Image.NEAREST)
    
    # Mask: 1=Hair(Optimize target), 0=Bg(Keep target)
    mask_np = np.array(mask_pil)
    mask_np = (mask_np > 127).astype(np.float32)
    
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    latent_res = resolution // 8
    # Resize mask to latent resolution
    mask_latent = F.interpolate(mask_tensor, size=(latent_res, latent_res), mode="nearest")

    # 2. Prepare Target Latents (Background)
    print("Encoding original image...")
    image_tensor = torch.from_numpy(np.array(image_pil)).float() / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device, dtype=dtype)
    
    with torch.no_grad():
        init_latents = pipe.vae.encode(image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

    # 3. Setup Initial Noise
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # We want to optimize this noise
    # To optimize, we need it to be a leaf tensor with gradients
    init_noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    init_noise.requires_grad_(True)
    
    optimizer = torch.optim.Adam([init_noise], lr=args.lr)

    # 4. SONIC Optimization Loop (Latent Space)
    print(f"Starting SONIC Optimization on Noise ({args.opt_steps} steps)...")
    
    # Prepare text embeddings (Fixed Condition)
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt, 
        prompt_2=args.prompt, 
        prompt_3=args.prompt, 
        negative_prompt="bad quality, ugly", 
        device=device,
        do_classifier_free_guidance=True
    )
    # Concatenate for single batch pass if needed, but for optimization we might just use uncond or cond?
    # SONIC paper: "Optimize x_T such that D(x_T) matches background"
    # Usually we optimize using the UNCONDITIONAL path to ensure structure match, or COND to match prompt context?
    # Let's use COND to be safe, or maybe just UNCOND to get pure structural noise.
    # For now, let's use the positive prompt.
    
    # Optimization loop
    timesteps = pipe.scheduler.timesteps if hasattr(pipe.scheduler, "timesteps") and len(pipe.scheduler.timesteps)>0 else torch.tensor([1000], device=device)
    # We choose a timestep close to T (e.g., 999 or the first step of inference)
    # In Flow Matching, T=1.0. 
    # Let's assume the scheduler is correctly set up.
    
    # We need to initialize scheduler explicitly to get sigmas
    pipe.scheduler.set_timesteps(args.steps, device=device)
    # The first timestep in the schedule (usually corresponding to high noise)
    t_start = pipe.scheduler.timesteps[0] 
    
    for i in tqdm(range(args.opt_steps)):
        optimizer.zero_grad()
        
        # We are at step t_start. 
        # Latent Input = init_noise (Assuming Flow Matching: z_1 is pure noise)
        # Note: In SD3 Flow Matching, the input IS the noise at t=1.0.
        
        # Predict x_0 (Original Sample) from current noise
        # SD3 Transformer predicts 'noise_pred' (velocity v).
        # v = u_t = x_1 - x_0 (in some parameterizations) or similar.
        # Rectified Flow: z_t = t * z_1 + (1-t) * z_0
        # at t=1: z_1 = noise. 
        # We want to know what this noise 'maps to' in terms of x_0.
        # Flow Matching ODE step: dt * v.
        # A full single-step prediction: x_0_pred = z_t - (something) * v_pred
        # For Euler step: prev_sample = sample + v * dt
        # But we want x_0 estimation. Use scheduler.
        
        # Forward pass (Gradient checkpointing might be needed if OOM, but for 1 step it should be fine)
        # We need to ensure logic allows gradient flow back to init_noise
        
        # 1. Transformer Forward
        latent_model_input = init_noise
        
        # Broadcast batch
        batch_size = latent_model_input.shape[0]
        t_tensor = torch.tensor([t_start] * batch_size, device=device)
        
        # We process separately or use CFG? 
        # To save memory, let's just use Positive Prompt for optimization guidance
        # or maybe Unconditional. 
        # Background matching shouldn't depend on "brown hair" prompt, it assumes background content.
        # So using Unconditional (empty text) is probably safer for background structure resonance?
        # Let's try Uncond embeds first (negative_prompt_embeds).
        
        # Actually, if we use uncond, we might lose "man" context. 
        # Let's use the actual prompt embeds to ensure the noise supports the prompt too, 
        # BUT the loss is only on the background.
        
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=t_tensor,
            encoder_hidden_states=prompt_embeds, # Use positive prompt
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]
        
        # 2. Linear Approximation / x_0 prediction
        # In Rectified Flow (SD3), v = x_1 - x_0
        # So x_0 = x_1 - v
        # x_1 is our init_noise (at t=1.0)
        # So pred_original_sample = init_noise - noise_pred
        
        # Note: This is a simplifiction. SD3 scheduler handles scales.
        # Let's trust the vector-field output `noise_pred` is `v`.
        # t=1.0 (sigma=1.0 typically). 
        # Check scheduler logic later if needed. For now assuming v-prediction:
        
        pred_x0 = init_noise - noise_pred 
        
        # 3. Calculate Loss (Masked MSE)
        # Goal: Background (1-Mask) of pred_x0 should match init_latents (Original Background)
        
        # Mask: 1=Hair, 0=Bg. We want (1-Mask) region.
        bg_mask = (1.0 - mask_latent)
        
        loss = F.mse_loss(pred_x0 * bg_mask, init_latents * bg_mask)
        
        # 4. Backward
        loss.backward()
        optimizer.step()
        
        if i % 5 == 0:
            print(f"Step {i}: Loss {loss.item()}")

    # 5. Generate with Optimized Noise
    print("Generating image with Optimized Noise...")
    optimized_noise = init_noise.detach() # Stop gradients
    
    # We now run the standard pipeline but passing our optimized latents
    # SD3 pipeline expects 'latents' argument which overrides internal random noise
    
    # Important: SD3 pipeline 'latents' arg usually expects the starting latents.
    # We must ensure we pass it correctly.
    
    with torch.no_grad():
        output = pipe(
            prompt=args.prompt,
            prompt_2=args.prompt, 
            prompt_3=args.prompt,
            negative_prompt="bad quality, ugly",
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            latents=optimized_noise, # Inject optimized noise here
            output_type="pil" # Return PIL directly
        ).images[0]
    
    output.save(args.output_path)
    print(f"Saved SONIC result to {args.output_path}")

if __name__ == "__main__":
    main()
