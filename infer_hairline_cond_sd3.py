
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
    parser.add_argument("--controlnet_a_path", type=str, default="output/hairline_sd3_run2_global/controlnet_a", help="Path to Geometry ControlNet (1ch)")
    parser.add_argument("--controlnet_b_path", type=str, default="output/hairline_sd3_run2_global/controlnet_b", help="Path to Identity ControlNet (16ch)")
    parser.add_argument("--bald_image", type=str, required=True, help="Path to Bald Image")
    parser.add_argument("--mask_image", type=str, required=True, help="Path to Hair Mask (White=Hair)")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="low quality, bad anatomy, distorted, ugly, blurry, pixelated")
    parser.add_argument("--output_path", type=str, default="output.png")
    parser.add_argument("--scale_geometry", type=float, default=1.0)
    parser.add_argument("--scale_identity", type=float, default=0.8)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.9, help="Noise Strength for Img2Img (0.0 to 1.0)")
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

class HybridVAE(torch.nn.Module):
    """
    Wrapper around VAE to bypass encoding for already-processed tensors.
    """
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        self.config = vae.config
        # Delegate attributes
    
    def __getattr__(self, name):
        # Prevent recursion by letting nn.Module resolve 'vae' and other standard attrs first
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to inner VAE if not found in wrapper
            # Use _modules directly to avoid recursive lookup of 'vae'
            if 'vae' in self._modules:
                return getattr(self._modules['vae'], name)
            raiseAttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def encode(self, x):
        # Bypass for Mask (1ch) or Latents (16ch)
        # These are already at Latent Resolution/Domain
        if x.shape[1] == 1 or x.shape[1] == 16:
            class MockDist:
                def __init__(self, val): self.val = val
                def sample(self, generator=None): return self.val
            class MockOutput:
                def __init__(self, val): self.latent_dist = MockDist(val)
                
            return MockOutput(x)
        return self.vae.encode(x)

class ConcatenatingSD3ControlNet(torch.nn.Module):
    """
    Wrapper to concatenate noisy_latents (hidden_states) with condition 
    before passing to ControlNet, matching Training Logic.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def __getattr__(self, name):
         # Delegate to inner model
         if name in ["model", "_modules"]:
             return super().__getattr__(name) 
         # Standard lookup
         try:
             return super().__getattr__(name)
         except AttributeError:
             return getattr(self.model, name)
             
    def forward(self, hidden_states, controlnet_cond, **kwargs):
        # Concatenate: [Latents(16) | Condition(N)]
        # This matches training: torch.cat([noisy_latents, mask_cond], dim=1)
        cond_input = torch.cat([hidden_states, controlnet_cond], dim=1)
        return self.model(
            hidden_states=hidden_states, 
            controlnet_cond=cond_input, 
            **kwargs
        )

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # Load ControlNets
    print("Loading ControlNets...")
    controlnet_a = SD3ControlNetModel.from_pretrained(args.controlnet_a_path, torch_dtype=dtype)
    controlnet_b = SD3ControlNetModel.from_pretrained(args.controlnet_b_path, torch_dtype=dtype)
    
    # Wrap them for Concatenation
    controlnet_a = ConcatenatingSD3ControlNet(controlnet_a)
    controlnet_b = ConcatenatingSD3ControlNet(controlnet_b)
    
    # Load Pipeline
    print("Loading Pipeline...")
    # Use Standard Pipeline but wrap VAE later
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=[controlnet_a, controlnet_b],
        torch_dtype=dtype
    )
    pipe.to(device)
    
    # Install VAE Wrapper
    pipe.vae = HybridVAE(pipe.vae)
    
    # Preprocess Inputs
    resolution = 1024 
    latent_res = resolution // 8 # 128
    
    # 1. Bald Image (for Identity)
    bald_pil = Image.open(args.bald_image).convert("RGB").resize((resolution, resolution))
    
    # 2. Mask (for Geometry)
    # 1-channel Mask -> Resize to Latent Res (128x128)
    mask_pil = Image.open(args.mask_image).convert("L")
    
    # We use smart blur at FULL resolution first?
    # No, preprocess_mask_smart does blur at full res.
    # We should run smart blur at 1024, then downsample to 128.
    mask_tensor_full = preprocess_mask_smart(mask_pil, resolution).to(device, dtype=dtype) # [1, 1, 1024, 1024]
    
    # Resize to Latent (128x128)
    mask_tensor_latent = torch.nn.functional.interpolate(
        mask_tensor_full, size=(latent_res, latent_res), mode="bilinear", align_corners=False
    )
    
    # Cutout Bald Logic (Identity)
    # Apply Mask to Bald Image: masked_bald = bald * (1-mask) + (-1)*mask.
    # Use sharp mask at FULL res for cutout
    hair_raw = (np.array(mask_pil.resize((resolution, resolution), Image.NEAREST)) == 255).astype(np.float32)
    hair_tensor = torch.from_numpy(hair_raw).unsqueeze(0).unsqueeze(0).to(device, dtype=dtype)
    
    bald_tensor_for_masking = transforms.ToTensor()(bald_pil).unsqueeze(0).to(device, dtype=dtype)
    bald_tensor_for_masking = (bald_tensor_for_masking - 0.5) / 0.5
    
    masked_bald_tensor = bald_tensor_for_masking * (1.0 - hair_tensor) + (-1.0) * hair_tensor
    with torch.no_grad():
        # This identity_latents is the CONTROLNET CONDITION for identity
        identity_latents_controlnet_cond = pipe.vae.vae.encode(masked_bald_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # 1. Encode Prompts
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=negative_prompt,
        device=device,
        do_classifier_free_guidance=True
    )
    
    if args.guidance_scale > 1.0:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    # 2. Prepare Latents (Start from Bald Image)
    # Encode Bald Image -> VAE -> Latents
    print("Encoding Bald Image as Start Point...")
    bald_tensor_input = transforms.ToTensor()(bald_pil).unsqueeze(0).to(device, dtype=dtype)
    bald_tensor_input = (bald_tensor_input - 0.5) / 0.5
    
    with torch.no_grad():
        # Use inner VAE to avoid wrapper logic if needed, or just pipe.vae.encode works if x is standard 3ch
        # pipe.vae.encode calls HybridVAE.encode -> standard encode for 3ch
        init_latents = pipe.vae.encode(bald_tensor_input).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    # 3. Add Noise (Img2Img Strength)
    strength = args.strength # Add this argument
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config) # Ensure scheduler is initialized
    pipe.scheduler = scheduler # Assign to pipe
    num_inference_steps = args.steps
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    # Calculate Start Step
    # strength 1.0 = pure noise (step 0), strength 0.0 = no noise (step end)
    init_timestep_idx = int(len(scheduler.timesteps) * (1.0 - strength))
    init_timestep_idx = max(0, min(init_timestep_idx, len(scheduler.timesteps) - 1))
    
    timesteps = scheduler.timesteps[init_timestep_idx:]
    print(f"Starting from step {init_timestep_idx}/{num_inference_steps} (Strength: {strength})")
    
    # Add noise to init_latents
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=dtype)
    start_timestep = timesteps[0]
    
    # Add noise using scheduler
    latents = scheduler.add_noise(init_latents, noise, start_timestep)
    
    # 4. Prepare Conditions (Already processed above)
    # mask_tensor_latent (1ch)
    # identity_latents_controlnet_cond (16ch)
    
    # 5. Denoising Loop
    print("Generating...")
    for t in tqdm(timesteps):
        with torch.no_grad():
            # Broadcast params for CFG
            latent_model_input = torch.cat([latents] * 2) if args.guidance_scale > 1.0 else latents
            
            # Prepare ControlNet Condition
            c_masks = mask_tensor_latent
            c_identity = identity_latents_controlnet_cond
            
            if args.guidance_scale > 1.0:
                 c_masks = torch.cat([c_masks] * 2)
                 c_identity = torch.cat([c_identity] * 2)
            
            control_conds = [c_masks, c_identity]
            
            # Block Samples
            # pipe.controlnet is a MultiControlNetModel, it expects a list of control_conds
            block_samples = pipe.controlnet(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                controlnet_cond=control_conds,
                conditioning_scale=[args.scale_geometry, args.scale_identity],
                return_dict=False
            )
            # block_samples is (down_block_res_samples, mid_block_res_sample)
            
            # Transformer Forward
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=t,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                block_controlnet_hidden_states=block_samples,
                return_dict=False
            )[0] # .sample
            
            # CFG
            if args.guidance_scale > 1.0:
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
    
    # Save (01047.png)
    image_pil = Image.fromarray((image[0] * 255).astype(np.uint8))
    
    # Make dir
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    image_pil.save(args.output_path)
    print(f"Saved to {args.output_path}")

if __name__ == "__main__":
    main()
