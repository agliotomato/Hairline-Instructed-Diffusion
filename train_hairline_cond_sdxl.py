#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionXLControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# Add SDXL specific imports
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

# Check versions
check_min_version("0.29.0")

logger = get_logger(__name__)

def compute_embeddings(prompt_batch, proportion_empty_prompts, text_encoder, text_encoder_2, tokenizer, tokenizer_2):
    # SDXL uses two text encoders. We need to implement embedding computation manually to handle dropouts.
    # For simplicity, we can reuse logic from diffusers examples or use pipeline components if available.
    # Here we implement basic encoding.
    
    prompt_embeds_list = []
    
    # We assume prompt_batch is a list of strings
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        else:
            captions.append(caption)
            
    with torch.no_grad():
        # Encoder 1
        text_inputs = tokenizer(
            captions, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder.device)
        prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
        # We need pooled output from encoder 2 usually, but let's check SDXL logic
        # SDXL: 
        # Enc1 -> hidden_states
        # Enc2 -> hidden_states, pooled_output
        # Concatenate hidden_states
        
        pooled_prompt_embeds = prompt_embeds[0] # Placeholder
        prompt_embeds = prompt_embeds.hidden_states[-2] # Penultimate layer
        
        # Encoder 2
        text_inputs_2 = tokenizer_2(
            captions, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt"
        )
        text_input_ids_2 = text_inputs_2.input_ids.to(text_encoder_2.device)
        prompt_embeds_2_out = text_encoder_2(text_input_ids_2, output_hidden_states=True)
        prompt_embeds_2 = prompt_embeds_2_out.hidden_states[-2]
        pooled_prompt_embeds = prompt_embeds_2_out.text_embeds # Pooled output
        
        # Concatenate
        prompt_embeds = torch.cat([prompt_embeds, prompt_embeds_2], dim=-1)
        
    return prompt_embeds, pooled_prompt_embeds

def compute_time_ids(original_size, crops_coords_top_left, target_size, device, weight_dtype):
    # SDXL Micro-Conditioning: (original_size, crops_coords_top_left, target_size)
    # Each is (H, W) or (y, x)? Usually (h, w).
    # We construct the add_time_ids tensor.
    # flatten and concat
    
    add_time_ids = list(original_size) + list(crops_coords_top_left) + list(target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=weight_dtype, device=device)
    return add_time_ids

# Dataset Class
class HairlineDataset(torch.utils.data.Dataset):
    def __init__(self, orig_dir, bald_dir, mask_dir, tokenizer, tokenizer_2, size=1024):
        self.orig_dir = Path(orig_dir)
        self.bald_dir = Path(bald_dir)
        self.mask_dir = Path(mask_dir)
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        
        self.images = sorted([f for f in self.orig_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
        # Default prompt
        self.prompt = "a photo of a person with realistic hairstyle" 

        self.transform_image = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        # Masks (Geometry): 3ch 
        self.transform_mask = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(), # [0, 1]
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        name = img_path.stem
        
        # Load Target Image (Original with Hair)
        image = Image.open(img_path).convert("RGB")
        
        # Load Bald Image (Identity)
        bald_path = self.bald_dir / f"{name}.png" 
        if not bald_path.exists():
             bald_path = self.bald_dir / f"{name}.jpg"
        bald_image = Image.open(bald_path).convert("RGB")
        
        # Load Mask (Geometry)
        mask_path = self.mask_dir / f"{name}.png"
        mask_image = Image.open(mask_path).convert("RGB") # Use RGB for 3ch input compatibility

        # Transform
        pixel_values = self.transform_image(image)
        condition_identity = self.transform_image(bald_image) # Same normalization as image? Yes, it's an image input.
        condition_geometry = self.transform_mask(mask_image)
        
        # Create Masked Bald Image (Identity + Mask applied roughly? No, just pass bald image as condition)
        # Actually, for Identity ControlNet, we usually pass the reference image. 
        # Here 'bald_image' is the reference for identity.
        
        return {
            "pixel_values": pixel_values,
            "condition_geometry": condition_geometry,
            "condition_identity": condition_identity,
            "prompt": self.prompt,
            "original_size": (self.size, self.size),
            "crops_coords_top_left": (0, 0),
            "target_size": (self.size, self.size),
        }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    condition_geometry = torch.stack([example["condition_geometry"] for example in examples])
    condition_identity = torch.stack([example["condition_identity"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    original_sizes = [example["original_size"] for example in examples]
    crop_coords = [example["crops_coords_top_left"] for example in examples]
    target_sizes = [example["target_size"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "condition_geometry": condition_geometry,
        "condition_identity": condition_identity,
        "prompt": prompts,
        "original_sizes": original_sizes,
        "crop_coords": crop_coords,
        "target_sizes": target_sizes,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--output_dir", type=str, default="sdxl_training_output")
    parser.add_argument("--orig_dir", type=str, required=True)
    parser.add_argument("--bald_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--proportion_empty_prompts", type=float, default=0.1)

    args = parser.parse_args()
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 1. Load Models
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16" if weight_dtype==torch.float16 else None)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16" if weight_dtype==torch.float16 else None)
    
    # VAE: Remove variant argument as the repo 'madebyollin/sdxl-vae-fp16-fix' implies the main file IS the fix
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, torch_dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16" if weight_dtype==torch.float16 else None, torch_dtype=weight_dtype)
    
    # 2. ControlNets
    # Initialize from UNet weights for faster convergence (Standard Practice)
    # Using MultiControlNet: List of models
    print("Initializing ControlNet A (Geometry)...")
    controlnet_a = ControlNetModel.from_unet(unet)
    print("Initializing ControlNet B (Identity)...")
    controlnet_b = ControlNetModel.from_unet(unet)
    
    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    
    # Train ControlNets
    controlnet_a.train()
    controlnet_b.train()
    
    # Enable Gradient Checkpointing
    controlnet_a.enable_gradient_checkpointing()
    controlnet_b.enable_gradient_checkpointing()
    unet.enable_gradient_checkpointing() # For memory saving on frozen model too

    if args.mixed_precision == "fp16":
        # Cast frozen models to fp16
        vae.to(dtype=weight_dtype)
        text_encoder.to(dtype=weight_dtype)
        text_encoder_2.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)

    # Optimizer
    params_to_optimize = list(controlnet_a.parameters()) + list(controlnet_b.parameters())
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Dataset & Dataloader
    dataset = HairlineDataset(
        orig_dir=args.orig_dir,
        bald_dir=args.bald_dir,
        mask_dir=args.mask_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Prepare with Accelerator
    controlnet_a, controlnet_b, optimizer, train_dataloader = accelerator.prepare(
        controlnet_a, controlnet_b, optimizer, train_dataloader
    )
    
    # Move frozen components
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    text_encoder_2.to(accelerator.device)
    unet.to(accelerator.device)

    # Training Loop
    global_step = 0
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet_a, controlnet_b):
                # 1. Encode Condition Images
                # Geometry (Mask) -> Pixel values (already matching resolution)
                # Identity (Bald) -> Pixel values
                cond_geo = batch["condition_geometry"].to(dtype=weight_dtype, device=accelerator.device)
                cond_id = batch["condition_identity"].to(dtype=weight_dtype, device=accelerator.device)
                
                # 2. Encode Latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype, device=accelerator.device)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # 3. Sample Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # 4. Add Noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # 5. Get Text Embeddings
                prompt_embeds, pooled_prompt_embeds = compute_embeddings(
                    batch["prompt"], 
                    args.proportion_empty_prompts,
                    text_encoder, text_encoder_2, tokenizer, tokenizer_2
                )
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                
                # 6. Prepare Time IDs (Micro-Conditioning)
                add_time_ids = torch.cat(
                    [compute_time_ids(s, c, t, accelerator.device, weight_dtype) for s, c, t in zip(batch["original_sizes"], batch["crop_coords"], batch["target_sizes"])]
                )
                
                # 7. ControlNet Forward
                # SDXL ControlNet: input=(noisy_latents, timesteps, encoder_hidden_states, controlnet_cond, added_cond_kwargs)
                # We have 2 ControlNets.
                
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }
                
                down_block_res_samples_a, mid_block_res_sample_a = controlnet_a(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    controlnet_cond=cond_geo,
                    return_dict=False,
                )
                
                down_block_res_samples_b, mid_block_res_sample_b = controlnet_b(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    controlnet_cond=cond_id,
                    return_dict=False,
                )
                
                # Merge Residuals (Simple Sum)
                down_block_res_samples = [a + b for a, b in zip(down_block_res_samples_a, down_block_res_samples_b)]
                mid_block_res_sample = mid_block_res_sample_a + mid_block_res_sample_b
                
                # 8. UNet Forward
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]
                
                # 9. Loss
                # SDXL default prediction type is epsilon usually
                # But check scheduler config? default is usually epsilon
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                
                if global_step % args.checkpointing_steps == 0:
                     save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                     accelerator.save_state(save_path)
                     logger.info(f"Saved state to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    main()
