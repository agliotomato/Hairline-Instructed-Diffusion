import argparse
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import os

# Import V2 Adapter
from modules.tiny_adapter_v2 import TinyAdapterV2

def main():
    parser = argparse.ArgumentParser(description="SONIC + TinyAdapter V2 + Latent Blending for SD3.5")
    parser.add_argument("--image_path", type=str, required=True, help="Path to bald image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to semantic mask")
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic hair, detailed texture")
    parser.add_argument("--output_path", type=str, default="results/sonic_hybrid_v2/output.png")
    parser.add_argument("--adapter_path", type=str, default="output/tiny_adapter_v2_checkpoints/tiny_adapter_v2_final.pth")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels for V2 adapter")
    parser.add_argument("--sonic_steps", type=int, default=15, help="Number of SONIC optimization steps")
    parser.add_argument("--sonic_lr", type=float, default=0.001)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--adapter_scale", type=float, default=1.0, help="Scale factor for adapter features")
    parser.add_argument("--soft_blending", action="store_true", help="Enable soft blending at pixel level")
    parser.add_argument("--blur_radius", type=float, default=5.0, help="Blur radius for soft mask")
    parser.add_argument("--mask_dilation", type=int, default=0, help="Dilate mask to allow hair growth (pixels)")
    
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
    
    # Load TinyAdapter V2
    print(f"Loading TinyAdapter V2 (128ch) from {args.adapter_path}")
    adapter = TinyAdapterV2(input_channels=1, hidden_channels=args.hidden_channels, output_channels=16)
    adapter.load_state_dict(torch.load(args.adapter_path, map_location=device))
    adapter.to(device, dtype=torch.bfloat16)
    adapter.eval()
    
    # 2. Helper Functions
    def load_image(path, size=(1024, 1024)):
        img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
        # return as bf16
        return (transforms.ToTensor()(img).unsqueeze(0).to(device) * 2.0 - 1.0).to(torch.bfloat16)

    def load_mask(path, size=(1024, 1024), blur_radius=0, dilation=0):
        # Load mask in L mode (0, 127, 255)
        mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
        
        # Threshold: Treat only Hair (255) as mask. Ignore Face (127).
        # Value > 200 (approx 0.8) -> 255, else 0.
        mask_np = np.array(mask)
        mask_bin = (mask_np > 200).astype(np.uint8) * 255
        mask = Image.fromarray(mask_bin)
        
        # Optional: Dilate mask to allow hair to grow slightly outside original mask
        if dilation > 0:
            mask = mask.filter(ImageFilter.MaxFilter(dilation*2 + 1)) # Approx dilation
            
        # Soft Blending: Blur the mask to create a gradient at the edges
        if blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            
        mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
        return mask_tensor.to(torch.bfloat16)

    # 3. Prepare Data
    print("Preparing Data...")
    # Inputs are bf16
    original_image = load_image(args.image_path)    
    
    # Use user specified blur for soft blending
    blur = args.blur_radius if args.soft_blending else 0
    mask_highres = load_mask(args.mask_path, blur_radius=blur, dilation=args.mask_dilation)        
    
    # Encode Original Image (y)
    with torch.no_grad():
        init_latents = pipe.vae.encode(original_image).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # Prepare Mask for Latent Blending
    # SD3 Latents are 1/8th size
    # Using 'bilinear' to avoid jagged edges in latent space
    mask_latent = F.interpolate(mask_highres, size=init_latents.shape[-2:], mode="bilinear", align_corners=False)
    
    # Binarize mask for Adapter? 
    # Usually adapter works better with binary mask structure, but soft mask might help transitions.
    # Let's keep adapter mask sharp-ish but bilinear is fine.
    mask_for_adapter = mask_latent.clone()
    # Optional: Hard threshold for adapter to keep structure strong? 
    # flow: mask_for_adapter = (mask_for_adapter > 0.5).float() 
    # Let's try continuous first as V2 was trained on continuous? 
    # Actually training used 'nearest' resize of binary mask.
    # To match training distribution better, we might want a sharper mask for the adapter input
    # but soft mask for blending.
    # Let's try using the soft mask for adapter too, might help 'seams'.
    
    # Get Adapter Features
    with torch.no_grad():
        adapter_features = adapter(mask_for_adapter) # [1, 16, 128, 128]
        # Apply Scale
        adapter_features = adapter_features * args.adapter_scale
    
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
        # Explicit cast to bf16 to match updated training script fix
        model_input = (init_noise_bf16 + adapter_features).to(torch.bfloat16)
        
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
        # t is a scalar tensor. We need shape [2].
        # Create a 1D tensor [t] and repeat it.
        ts = torch.tensor([t], device=device)
        ts = ts.repeat(2)
        
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=ts,
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
        
        if i < len(timesteps) - 1:
            next_t = timesteps[i+1] # Next timestep in the schedule
        
            # Generate consistent noise instance (simplification: regenerate for now)
            # Consistent noise is better but stochastic is fine for blending usually.
            # Using random noise each step for background blending target
            noise_bg = torch.randn_like(init_latents)
            
            # SD3 Flow Matching Manual Noise Addition:
            # z_t = (1 - sigma) * x + sigma * noise
            
            # pipe.scheduler.sigmas follows the same order as timesteps
            sigma_next = pipe.scheduler.sigmas[i + 1]
            
            # Interpolate
            latents_bg = (1 - sigma_next) * init_latents + sigma_next * noise_bg
            
            # Blend: Keep Generated in Mask (1), Keep Background in (1-Mask)
            # Use soft mask for smooth transition in latent space
            latents = latents * mask_latent + latents_bg * (1 - mask_latent)
            
    # Decode Final
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    
    # Normalize Generated Image (Float 0-1)
    image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0] # [H, W, 3]
    generated_pil = Image.fromarray((image * 255).round().astype("uint8"))
    
    # ---------------------------------------------------------
    # Phase 3: Pixel Space Compositing (Final Color Fix)
    # ---------------------------------------------------------
    if args.soft_blending:
        print("Applying Pixel-Space Compositing with Soft Mask...")
        # Load Original Image as PIL (clean)
        original_pil = Image.open(args.image_path).convert("RGB").resize(generated_pil.size, Image.BILINEAR)
        
        # Load Soft Mask as PIL (0-255)
        # We re-load or use the one we prepared? 
        # Using the same blur radius logic ensures consistency.
        mask_pil = Image.open(args.mask_path).convert("L").resize(generated_pil.size, Image.NEAREST)
        if args.blur_radius > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(args.blur_radius))
        
        # Composite
        # final = generated * mask + original * (1 - mask)
        # PIL.Image.composite(image1, image2, mask) -> image1 where mask is 255
        final_image = Image.composite(generated_pil, original_pil, mask_pil)
    else:
        final_image = generated_pil
        
    final_image.save(args.output_path)
    print(f"Saved result to {args.output_path}")

if __name__ == "__main__":
    main()
