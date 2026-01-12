import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler

# Import our TinyAdapter
from modules.tiny_adapter import TinyAdapter

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HairDataset(Dataset):
    """
    Simple dataset loading Bald Images and Masks.
    Structure:
    - data_root/bald_images/
    - data_root/semantic_masks/
    """
    def __init__(self, data_root, resolution=1024):
        self.data_root = data_root
        self.resolution = resolution
        self.bald_dir = os.path.join(data_root, "bald_images")
        self.mask_dir = os.path.join(data_root, "semantic_masks")
        
        self.image_names = [f for f in os.listdir(self.bald_dir) if f.endswith(('.png', '.jpg'))]
        
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load Images
        bald_path = os.path.join(self.bald_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        bald_img = Image.open(bald_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L") # 1-channel mask
        
        # Transform
        # Resize to resolution
        T_img = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) # [-1, 1] for SD inputs
        ])
        
        T_mask = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() # [0, 1]
        ])
        
        pixel_values = T_img(bald_img)
        mask = T_mask(mask_img)
        
        # Binary Mask > 0.5 (127/255)
        mask = (mask > 0.5).float()

        return {
            "pixel_values": pixel_values,
            "mask": mask,
            "name": img_name
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output/tiny_adapter_checkpoints")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1) # H100 allows bigger, but start small
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    args = parser.parse_args()

    # Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Components
    logger.info("Loading SD3.5 components...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=weight_dtype)
    pipe.to(device)
    
    # Freeze SD3 parts
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder_3.requires_grad_(False)

    # 2. Initialize TinyAdapter
    logger.info("Initializing TinyAdapter...")
    adapter = TinyAdapter(input_channels=1, output_channels=16)
    adapter.to(device, dtype=weight_dtype)
    adapter.train()

    # Optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

    # Dataset & Dataloader
    dataset = HairDataset(data_root=args.data_root, resolution=args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    # Training Loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps))
    progress_bar.set_description("Steps")

    # Fixed Prompt Embedding (Doing "Unconditional" or Generic Prompt training)
    # For structure guidance, we often train with empty prompt or generic "hair" prompt.
    # Let's use a generic prompt to give some context.
    generic_prompt = "a photo of a person"
    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            generic_prompt, generic_prompt, generic_prompt, device=device
        )
        # Concat for CFG or just use uncond? Training usually uses one.
        # Let's use the positive embedding condition.
    
    logger.info("Starting Training...")
    
    for epoch in range(100): # Hard loop, breaks by max_steps
        for batch in dataloader:
            # Prepare Inputs
            images = batch["pixel_values"].to(device, dtype=weight_dtype)
            masks = batch["mask"].to(device, dtype=weight_dtype) # High-res mask [B, 1, 1024, 1024]
            
            # 1. VAE Encode (Target)
            # We want to reconstruct the image (or part of it).
            # But wait, we are doing ControlNet training: Denoising Loss.
            # We add noise to VAE latents, and try to predict noise, conditioned on Adapter Features.
            
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * pipe.vae.config.scaling_factor
            
            # 2. Resize Mask for Adapter
            # Adapter operates at latent resolution (128x128)
            latent_res = args.resolution // 8
            masks_latent = F.interpolate(masks, size=(latent_res, latent_res), mode="nearest")
            
            # 3. Adapter Forward
            # Input: Mask [B, 1, 128, 128] -> Output: Features [B, 16, 128, 128]
            adapter_features = adapter(masks_latent)
            
            # 4. Add Noise (Timestep Sampling)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            # SD3 uses Flow Matching (t=0..1000 usually in diffusers config, or 0..1 sigmas)
            # Diffusers SD3 scheduler samples timesteps.
            
            # We need valid discrete timesteps
            # Just sample random integers from 0 to 1000
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            # Add noise (Forward diffusion)
            # Diffusers scheduler.add_noise
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # 5. Model Forward (with Adapter injection)
            # SD3 Transformer input: hidden_states + Adapter Features?
            # Standard ControlNet adds residuals. 
            # SD3 ControlNet architecture typically adds to specific blocks.
            # BUT, we are doing a "Lightweight Injection".
            # Simplest approach (T2I-Adapter style): Add directly to noisy_latents input?
            # Or concat? SD3 input is 16 channels. Adapter output is 16 channels.
            # We can ADD it to the input latents: (noisy_latents + adapter_features)
            
            model_input = noisy_latents + adapter_features
            
            # Predict
            model_pred = pipe.transformer(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False
            )[0]
            
            # 6. Loss Calculation (Flow Matching Loss)
            # Target is 'noise - latents' (velocity) or just 'noise'?
            # Check scheduler prediction type.
            # SD3 FlowMatchEuler uses 'rectified_flow' -> target is usually (noise - latents_0) or similar.
            # Pipe uses scheduler.get_velocity. 
            # For simplicity in custom loop: 
            # target = noise - latents (if v-pred)
            # target = noise (if eps-pred)
            # SD3 is Rectified Flow. Target v = x_1 (noise) - x_0 (image).
            
            target = noise - latents
            
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # 7. Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step % 10 == 0:
                logger.info(f"Step {global_step}: Loss {loss.item():.4f}")
            
            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                torch.save(adapter.state_dict(), save_path)
                logger.info(f"Saved Checkpoint to {save_path}")
            
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    # Final Save
    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "tiny_adapter_final.pth"))
    logger.info("Training Finished.")

if __name__ == "__main__":
    main()
