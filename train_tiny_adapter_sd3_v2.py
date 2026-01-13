import argparse
import itertools
import math
import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler

# Import V2 Adapter
from modules.tiny_adapter_v2 import TinyAdapterV2

def main():
    parser = argparse.ArgumentParser(description="Train TinyAdapter V2 for SD3.5")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="output/tiny_adapter_v2_checkpoints")
    parser.add_argument("--hidden_channels", type=int, default=128, help="Hidden channels for V2 adapter")
    args = parser.parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Environment Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fix for 'GET was unable to find an engine' (CUDNN issue)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    weight_dtype = torch.bfloat16 # Use bf16 for SD3.5 model
    
    # 1. Load SD3.5 (Frozen)
    logger.info("Loading SD3.5 components...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=weight_dtype
    )
    pipe.to(device)
    
    # Freeze everything
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder_3.requires_grad_(False)
    
    # 2. Init TinyAdapter V2
    logger.info(f"Initializing TinyAdapter V2 with hidden_channels={args.hidden_channels}...")
    adapter = TinyAdapterV2(input_channels=1, hidden_channels=args.hidden_channels, output_channels=16)
    adapter.to(device, dtype=weight_dtype) # Train in bf16
    adapter.train()

    # 3. Dataset
    class HairDataset(Dataset):
        def __init__(self, root_dir, resolution=1024):
            self.root_dir = root_dir
            self.resolution = resolution
            self.image_dir = os.path.join(root_dir, "bald_images")
            self.mask_dir = os.path.join(root_dir, "semantic_masks")
            self.image_files = sorted(os.listdir(self.image_dir))
            
            self.img_transform = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(), # [0, 1]
                transforms.Normalize([0.5], [0.5]) # [-1, 1]
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ])

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, img_name) # Assuming same name

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            return {
                "pixel_values": self.img_transform(image),
                "mask": (self.mask_transform(mask) > 0.5).float() # Binary 0/1
            }

    dataset = HairDataset(args.data_root, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

    # 5. Training Loop
    logger.info("Starting Training...")
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Pre-compute prompt embeds (Simplified: Use one generic prompt)
    prompt = "a photo of a person"
    with torch.no_grad():
        (prompt_embeds, negative_prompt_embeds, 
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
            prompt, prompt, prompt, device=device
        )

    while global_step < args.max_train_steps:
        for batch in dataloader:
            # Prepare Inputs
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype) # [B, 3, 1024, 1024]
            mask = batch["mask"].to(device, dtype=weight_dtype)                 # [B, 1, 1024, 1024]
            bsz = pixel_values.shape[0]
            
            # VAE Encode
            with torch.no_grad():
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor
            
            # Noise
            noise = torch.randn_like(latents)
            
            # Sample Timesteps
            timesteps = torch.randint(0, 1000, (bsz,), device=device).long()
            
            # Add Noise (Flow Matching Manual)
            sigmas = timesteps.float() / 1000.0
            sigmas = sigmas.view(-1, 1, 1, 1)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            
            # --- Adapter Forward ---
            # Resize mask to latent size (128x128)
            mask_latent = F.interpolate(mask, size=latents.shape[-2:], mode="nearest")
            
            # V2 Adapter: Returns 16ch features
            adapter_features = adapter(mask_latent) # [B, 16, 128, 128]
            
            # Injection
            model_input = noisy_latents + adapter_features
            
            # Target (Flow Matching)
            target = noise - latents
            
            # Predict
            model_pred = pipe.transformer(
                hidden_states=model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.repeat(bsz, 1, 1),
                pooled_projections=pooled_prompt_embeds.repeat(bsz, 1),
                return_dict=False
            )[0]
            
            # Loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            # Backward
            loss.backward()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                progress_bar.set_description(f"Step {global_step+1}: Loss {loss.item():.4f}")
            
            global_step += 1
            
            if global_step >= args.max_train_steps:
                break
                
            if global_step % 500 == 0:
                 torch.save(adapter.state_dict(), os.path.join(args.output_dir, f"checkpoint-{global_step}.pth"))

    # Save Final
    torch.save(adapter.state_dict(), os.path.join(args.output_dir, "tiny_adapter_v2_final.pth"))
    logger.info(f"Training Finished. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
