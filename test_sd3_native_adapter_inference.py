
import argparse
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from PIL import Image, ImageFilter
from torchvision import transforms
import numpy as np
from scipy.ndimage import binary_erosion
from tqdm import tqdm
import os
import sys

# Add current dir to path to find modules
sys.path.append(os.getcwd())

# Import Native Adapter (1024px -> 128px)
from modules.tiny_adapter_native import TinyAdapterNative

def main():
    parser = argparse.ArgumentParser(description="Inference with TinyAdapterNative (SD3.5)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to bald image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to semantic mask (1024x1024)")
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic hair, detailed texture")
    parser.add_argument("--output_path", type=str, default="results/native_adapter_test/output.png")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to trained native adapter checkpoint")
    parser.add_argument("--adapter_mask_path", type=str, default=None, help="Optional: Path to target shape mask (e.g. for long->short generation)")
    parser.add_argument("--sonic_steps", type=int, default=15, help="Number of SONIC optimization steps")
    parser.add_argument("--sonic_lr", type=float, default=0.001)
    parser.add_argument("--inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--adapter_scale", type=float, default=1.0, help="Scale factor for adapter features")
    parser.add_argument("--soft_blending", action="store_true", help="Enable soft blending at pixel level")
    parser.add_argument("--blur_radius", type=float, default=5.0, help="Blur radius for soft mask")
    parser.add_argument("--mask_dilation", type=int, default=0, help="Dilate mask to allow hair growth (pixels)")
    parser.add_argument("--smart_blur", action="store_true", help="Apply heavy blur except for the hairline area")
    parser.add_argument("--save_mask_preview", action="store_true", help="Save processed hair mask preview image")
    parser.add_argument("--edge_blur_limit", type=float, default=0.8, help="Maximum blur radius for edge/detail areas (Smart Blur)")
    
    # Adding save_intermediate argument
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate generation steps")
    
    args = parser.parse_args()
    
    # Ensure dirs
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 1. Setup Device & Model
    print("!!! RUNNING UPDATED SCRIPT WITH DEBUG LOGIC !!!")
    print(f"DEBUG: Smart Blur={args.smart_blur}, Blur Radius={args.blur_radius}, Adapter Scale={args.adapter_scale}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Load SD3.5 in bfloat16
    print("Loading SD 3.5...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16 
    ).to(device)
    
    # Load TinyAdapter Native
    print(f"Loading TinyAdapterNative from {args.adapter_path}")
    # Default base_channels=32 (128ch hidden) as used in training
    adapter = TinyAdapterNative(input_channels=1, base_channels=32, output_channels=16)
    
    # Load Weights
    # If checkpoint is full state dict or model state dict? Usually accelerator saves model state dict if unwrapped.
    # Let's try loading safely
    state_dict = torch.load(args.adapter_path, map_location=device)
    adapter.load_state_dict(state_dict)
    
    adapter.to(device, dtype=torch.bfloat16)
    adapter.eval()
    
    # 2. Helper Functions
    def load_image(path, size=(1024, 1024)):
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        # return as bf16
        return (transforms.ToTensor()(img).unsqueeze(0).to(device) * 2.0 - 1.0).to(torch.bfloat16)

    def load_mask(path, size=(1024, 1024), blur_radius=0, dilation=0, smart_blur=False, output_dir=".", edge_limit=1.0):
        # Load Raw Mask
        # Revert to NEAREST for mask to avoid Lanczos ringing artifacts when thresholding
        raw_mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
        raw_np = np.array(raw_mask)
        print(f"DEBUG: Inside load_mask. Smart Blur={smart_blur}, Radius={blur_radius}")
        
        # Define Regions
        # Hair: > 200 (255)
        hair_mask = (raw_np > 200).astype(np.uint8) 
        
        # Face: 127 (Protection)
        face_mask = ((raw_np > 50) & (raw_np < 200)).astype(np.uint8) 
        
        if dilation > 0:
            hair_pil = Image.fromarray(hair_mask * 255)
            dilated_pil = hair_pil.filter(ImageFilter.MaxFilter(dilation*2 + 1))
            # V3 Fix: Allow dilation to expand into the Face area (Forehead).
            # Previous logic `& (~face_mask)` prevented hairline lowering by hard-clipping at the original hairline.
            # We assume if the user asks for dilation, they intend to overwrite the boundary.
            final_mask_np = np.array(dilated_pil) > 127
            mask = Image.fromarray((final_mask_np * 255).astype(np.uint8))
        else:
            mask = Image.fromarray((hair_mask * 255).astype(np.uint8))
            
        if smart_blur and blur_radius > 0:
             print(f"Applying Smart Blur V2 (Erosion-based Protection)...")
             # 1. Heavy Blur for Volume
             # V3 Refinement: Removed 4.0x multiplier.
             # Previous 20.0px blur caused "helmet hair" on adapter trained with sharp masks.
             # Now using user-provided radius (e.g. 5.0) directly.
             heavy_radius = blur_radius
             mask_heavy = mask.filter(ImageFilter.GaussianBlur(heavy_radius))
             
             # 2. Light Blur for Details (Hairline, Sideburns)
             # V2 Improvement: "Thin" areas vanish with large blur.
             # Cap to user specified limit (default 1.0)
             light_radius = min(blur_radius, edge_limit)
             mask_light = mask.filter(ImageFilter.GaussianBlur(light_radius))
             
             # 3. Create Core Mask (Erosion)
             # Identify "thick" areas where heavy blur is safe.
             erosion_size = int(heavy_radius)
             y, x = np.ogrid[-erosion_size:erosion_size+1, -erosion_size:erosion_size+1]
             struct = x*x + y*y <= erosion_size*erosion_size
             
             # hair_mask is boolean 0/1 from earlier
             # (hair_mask > 200) was used to create 'mask' but we need the bool array
             # In logical terms: hair_mask (from line 81) is uint8 0 or 1.
             hair_bool = (hair_mask > 0)
             
             core_mask_np = binary_erosion(hair_bool, structure=struct)
             core_pil = Image.fromarray((core_mask_np * 255).astype(np.uint8))
             
             # Smooth the Core Mask transition
             core_smooth = core_pil.filter(ImageFilter.GaussianBlur(blur_radius * 2))
             
             # 4. Composite
             # Use Heavy where Core is active, Light elsewhere
             mask = Image.composite(mask_heavy, mask_light, core_smooth)

             # DEBUG (V2): Save the actual mask being sent to the adapter
             debug_save_path = os.path.join(output_dir, f"debug_smart_mask_input_{os.path.basename(path)}")
             mask.save(debug_save_path)
             print(f"[DEBUG] Saved Smart Blur Mask to {debug_save_path}")
             
        elif blur_radius > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            
        mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(device)
        # Normalization Step (Vital for blurred masks)
        # Blurring reduces peak intensity (e.g. 255 -> 150).
        # If the adapter expects ~1.0 (255) inputs, this drop causes "weak generation" or "ignored areas".
        # We stretch the intensity so the maximum value hits 1.0 (255) again.
        mask_max = mask_tensor.max()
        if mask_max > 0:
            print(f"DEBUG: Mask Max Intensity before Norm: {mask_max:.4f}")
            mask_tensor = mask_tensor / mask_max
            print(f"DEBUG: Mask Max Intensity after Norm: {mask_tensor.max():.4f}")
            
        return mask_tensor.to(torch.bfloat16)

    # 3. Prepare Data
    print("Preparing Data...")
    original_image = load_image(args.image_path)    
    
    # Pass blur_radius ONLY if Smart Blur is enabled. 
    # For Soft Blending, we want strict generation (sharp mask) but soft pixel composition.
    # So we force blur=0 here if smart_blur is False, and handle soft_blending blur in Phase 3.
    blur = args.blur_radius if args.smart_blur else 0
    mask_highres = load_mask(args.mask_path, blur_radius=blur, dilation=args.mask_dilation, smart_blur=args.smart_blur, output_dir=os.path.dirname(args.output_path), edge_limit=args.edge_blur_limit)        
    if args.save_mask_preview:
        mask_preview = transforms.ToPILImage()(mask_highres[0].float().cpu())
        mask_base, mask_ext = os.path.splitext(args.output_path)
        mask_preview_path = f"{mask_base}_mask{mask_ext or '.png'}"
        mask_preview.save(mask_preview_path)
        print(f"Saved mask preview to {mask_preview_path}")
    
    # Prepare Mask for Latent Blending (Low Res for Blending)
    # SD3 Latents are 1/8th size
    with torch.no_grad():
        init_latents = pipe.vae.encode(original_image).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    mask_latent = F.interpolate(mask_highres, size=init_latents.shape[-2:], mode="bilinear", align_corners=False)

    # Prepare Mask for Adapter (High Res Native)
    # Logic: 
    # - Inpainting Mask (--mask_path): The "Canvas Area" we are allowed to touch (delete old hair + add new hair).
    # - Adapter Mask (--adapter_mask_path): The "Structural Guide" for the new hair shape.
    # If adapter_mask_path is not provided, we assume target shape == original shape.
    
    if args.adapter_mask_path:
        print(f"Loading separate Adapter Mask from {args.adapter_mask_path}")
        # Adapter mask generally shouldn't need dilation aimed at hiding seams, 
        # but we use standard loading for consistency. 
        # We assume the user provides a mask representing the TARGET hair shape.
        mask_for_adapter = load_mask(args.adapter_mask_path, blur_radius=0, dilation=0)
    else:
        # Default behavior: Structure follows the editing area
        mask_for_adapter = mask_highres.clone()

    # Get Adapter Features
    with torch.no_grad():
        print("Running Adapter (Native Resolution)...")
        adapter_features = adapter(mask_for_adapter) # Output should be [1, 16, 128, 128]
        print(f"Adapter Output Shape: {adapter_features.shape}")
        
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
    
    init_noise = torch.randn_like(init_latents, dtype=torch.float32)
    init_noise.requires_grad = True
    optimizer = torch.optim.Adam([init_noise], lr=args.sonic_lr)
    
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    pbar = tqdm(range(args.sonic_steps), desc="SONIC")
    for i in pbar:
        init_noise_bf16 = init_noise.to(torch.bfloat16)
        t = torch.tensor([999], device=device)
        
        # Inject Adapter
        model_input = (init_noise_bf16 + adapter_features).to(torch.bfloat16)
        
        model_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=t,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]
        
        pred_x0 = init_noise - model_pred.to(torch.float32)
        
        target_latents = init_latents.to(torch.float32)
        mask_float = (mask_latent > 0.01).to(torch.float32)
        
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
    # Phase 2: Generation
    # ---------------------------------------------------------
    print("Starting Phase 2: Hybrid Generation...")
    
    pipe.scheduler.set_timesteps(args.inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    latents = init_noise.detach().clone().to(torch.bfloat16)
    noise_bg_base = init_noise.detach().clone().to(torch.bfloat16)

    # Prepare dir for intermediate steps
    if args.save_intermediate:
        intermediate_dir = os.path.join(os.path.dirname(args.output_path), "intermediate_steps")
        os.makedirs(intermediate_dir, exist_ok=True)
        print(f"Saving intermediate steps to {intermediate_dir}")

    for i, t in enumerate(progressbar := tqdm(timesteps)):
        latent_model_input = torch.cat([latents] * 2)
        adapter_input = torch.cat([adapter_features] * 2)
        model_input = latent_model_input + adapter_input
        
        prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_prompt_embeds_input = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        
        ts = torch.tensor([t], device=device).repeat(2)
        
        noise_pred = pipe.transformer(
            hidden_states=model_input,
            timestep=ts,
            encoder_hidden_states=prompt_embeds_input,
            pooled_projections=pooled_prompt_embeds_input,
            return_dict=False
        )[0]
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        step_output = pipe.scheduler.step(noise_pred, t, latents)
        latents = step_output.prev_sample
        
        # Latent Blending
        if i < len(timesteps) - 1:
            sigma_next = pipe.scheduler.sigmas[i + 1]
            latents_bg = (1 - sigma_next) * init_latents + sigma_next * noise_bg_base
            latents = latents * mask_latent + latents_bg * (1 - mask_latent)

        # Save Intermediate Step
        if args.save_intermediate:
            with torch.no_grad():
                # Decode intermediate latents
                decoded = pipe.vae.decode(latents, return_dict=False)[0]
                decoded = (decoded / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
                decoded_pil = Image.fromarray((decoded * 255).round().astype("uint8"))
                
                step_filename = f"step_{i:03d}.png"
                decoded_pil.save(os.path.join(intermediate_dir, step_filename))
            
    # Decode
    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]
    
    image = (image / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).float().numpy()[0]
    generated_pil = Image.fromarray((image * 255).round().astype("uint8"))
    
    # ---------------------------------------------------------
    # Phase 3: Pixel Space Compositing
    # ---------------------------------------------------------
    if args.soft_blending:
        print(f"Applying Pixel-Space Compositing with Soft Edge (Radius={args.blur_radius})...")
        original_pil = Image.open(args.image_path).convert("RGB").resize(generated_pil.size, Image.BILINEAR)
        mask_tensor_proc = mask_highres[0].float()
        mask_pil = transforms.ToPILImage()(mask_tensor_proc)
        
        # Post-process mask: Apply blur ONLY for compositing
        if args.blur_radius > 0:
             mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(args.blur_radius))
        
        if mask_pil.size != generated_pil.size:
            mask_pil = mask_pil.resize(generated_pil.size, Image.BILINEAR)
            
        final_image = Image.composite(generated_pil, original_pil, mask_pil)
    else:
        final_image = generated_pil
        
    final_image.save(args.output_path)
    print(f"Saved result to {args.output_path}")

if __name__ == "__main__":
    main()
