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

# [V4 Philosophy] Use specialized Latent Identity Net
from utils.latent_identity_net import ControlNetModel as LatentIdentityNet

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
        
        self.images = []
        all_images = sorted([f for f in self.orig_dir.glob('*') if f.suffix.lower() in ['.jpg', '.png', '.jpeg']])
        
        # Filter images that exist in all directories
        print(f"Filtering dataset... Found {len(all_images)} candidates.")
        valid_count = 0
        for img_path in all_images:
            name = img_path.stem
            
            # Check Bald
            bald_path = self._find_file(self.bald_dir, name)
            
            # Check Mask
            mask_path = self._find_file(self.mask_dir, name)
            
            if bald_path and mask_path:
                self.images.append({
                    "orig": img_path,
                    "bald": bald_path,
                    "mask": mask_path
                })
                valid_count += 1
        
        print(f"Dataset filtered. Keeping {valid_count}/{len(all_images)} valid samples.")
        
        # Default prompt
        self.prompt = "a photo of a person with realistic hairstyle" 

    def _find_file(self, directory, name):
        for ext in ['.png', '.jpg', '.jpeg']:
            path = directory / f"{name}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        item = self.images[idx]
        
        # Load Images
        orig_image = Image.open(item["orig"]).convert("RGB")
        bald_image = Image.open(item["bald"]).convert("RGB")
        mask_image = Image.open(item["mask"]).convert("L") # V4 Philosophy: 1ch Geometry Mask

        # 1. Synchronized Resize
        orig_image = orig_image.resize((self.size, self.size), Image.BILINEAR)
        bald_image = bald_image.resize((self.size, self.size), Image.BILINEAR)
        mask_image = mask_image.resize((self.size, self.size), Image.NEAREST)

        # 2. Synchronized Horizontal Flip (Augmentation)
        do_flip = random.random() < 0.5
        if do_flip:
            orig_image = orig_image.transpose(Image.FLIP_LEFT_RIGHT)
            bald_image = bald_image.transpose(Image.FLIP_LEFT_RIGHT)
            mask_image = mask_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 3. Base Transform (To Tensor + Normalize)
        # We need mask in [0, 1] for math
        mask_tensor = transforms.ToTensor()(mask_image) # 1ch [0, 1]
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0) 
        
        # Normalize images to [-1, 1]
        norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        orig_pixel_values = norm(transforms.ToTensor()(orig_image))
        bald_pixel_values = norm(transforms.ToTensor()(bald_image))
        
        # 4. Create Masked Identity Condition (V4 Style)
        masked_identity = bald_pixel_values * (1.0 - mask_tensor) + (-1.0) * mask_tensor
        
        return {
            "pixel_values": orig_pixel_values,
            "condition_geometry": mask_tensor, # 1ch Geometry Mask
            "condition_identity": masked_identity, # Masked Bald Images for Identity Net
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
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    
    # --- CUDA Check ---
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA is not available. This script requires a GPU.")
    
    print(f"DEBUG: CUDA Available: {torch.cuda.is_available()}")
    print(f"DEBUG: Current Device Index: {torch.cuda.current_device()}")
    print(f"DEBUG: Device Name: {torch.cuda.get_device_name(0)}")
    
    # Enable TF32 for faster training on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    print("✅ TF32 enabled.")
    # ------------------
    
    if args.seed is not None:
        set_seed(args.seed)

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
    
    # Freeze base models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)
    
    # 2. ControlNets
    # [V4 Philosophy] Hybrid Dual-Stream
    # Stream A: Geometry (Pixel Space, 1ch)
    # Stream B: Identity (Latent Space, 4ch)
    print("Initializing ControlNet A (Geometry, 1ch)...")
    controlnet_a = ControlNetModel.from_unet(unet, conditioning_channels=1)
    
    print("Initializing ControlNet B (Identity, 4ch Latent)...")
    controlnet_b = LatentIdentityNet.from_unet(unet, conditioning_channels=4)
    
    # Train ControlNets
    controlnet_a.train()
    controlnet_b.train()
    
    # Enable Gradient Checkpointing
    if args.gradient_checkpointing:
        print("✅ Enabling Gradient Checkpointing.")
        controlnet_a.enable_gradient_checkpointing()
        controlnet_b.enable_gradient_checkpointing()
        unet.enable_gradient_checkpointing() 

    # Enable Xformers (Crucial for SDXL on <48GB VRAM)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet_a.enable_xformers_memory_efficient_attention()
            controlnet_b.enable_xformers_memory_efficient_attention()
            print("✅ Xformers memory efficient attention enabled.")
        else:
            print("⚠️ Xformers not available. Expect high VRAM usage.")

    if args.mixed_precision == "fp16":
        # Cast frozen models to fp16
        vae.to(dtype=weight_dtype)
        text_encoder.to(dtype=weight_dtype)
        text_encoder_2.to(dtype=weight_dtype)
        unet.to(dtype=weight_dtype)

    # Optimizer: Use 8-bit AdamW to save VRAM (Crucial even at 512px for 2 ControlNets)
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("✅ Using 8-bit AdamW optimizer (bitsandbytes) to fit in VRAM.")
        except ImportError:
            optimizer_class = torch.optim.AdamW
            print("⚠️ bitsandbytes not found. Using standard AdamW (High VRAM risk).")
    else:
        optimizer_class = torch.optim.AdamW
        print("✅ Using standard AdamW optimizer.")

    params_to_optimize = list(controlnet_a.parameters()) + list(controlnet_b.parameters())
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Dataset & Dataloader
    # Note: HairlineDataset transform can be modified for augmentation if needed. 
    # For now we rely on the implementation above.
    dataset = HairlineDataset(
        orig_dir=args.orig_dir,
        bald_dir=args.bald_dir,
        mask_dir=args.mask_dir,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=args.resolution
    )
    
    # Inject RandomHorizontalFlip for augmentation (Quality Boost)
    print("✅ Adding RandomHorizontalFlip augmentation.")
    dataset.transform_image = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5), # Augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    # Note: Mask needs exact same flip? 
    # Yes. Standard RandomHorizontalFlip usually requires setting seed or functional transforms for paired images.
    # The default HairlineDataset is not set up for synchronized transforms.
    # To avoid mismatch, we will skip augmentation for now in this quick patch OR rely on deterministic transforms.
    # Wait, if we flip 'bald' and 'mask' differently, training breaks.
    # Reverting flip for safety unless we rewrite Dataset class fully.
    # Keeping standard resize-only for safety. 
    # To truly enable augmentation, we need a sync transform logic.
    # I will SKIP augmentation to avoid breaking correspondence, but keep standard loading.
    dataset.transform_image = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

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
    # Note: We need to prepare EVERYTHING now including frozen models? No, usually just trainable logic.
    # But Accelerator handles device placement for prepared objects.
    controlnet_a, controlnet_b, optimizer, train_dataloader = accelerator.prepare(
        controlnet_a, controlnet_b, optimizer, train_dataloader
    )
    
    # Move frozen components manually to device
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
                added_cond_kwargs = {
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": add_time_ids,
                }
                
                # Geometry Forward (Pixel Space 1ch)
                down_block_res_samples_a, mid_block_res_sample_a = controlnet_a(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    controlnet_cond=cond_geo,
                    return_dict=False,
                )
                
                # Identity Forward (Latent Space 4ch)
                # [V4 Philosophy] Encode Masked Identity to Latents first
                with torch.no_grad():
                    cond_id_latents = vae.encode(cond_id).latent_dist.sample()
                    cond_id_latents = cond_id_latents * vae.config.scaling_factor

                down_block_res_samples_b, mid_block_res_sample_b = controlnet_b(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs=added_cond_kwargs,
                    controlnet_cond=cond_id_latents,
                    return_dict=False,
                )
                
                # Merge Residuals (Simple Sum) and Cast to weight_dtype
                down_block_res_samples = [ (a + b).to(dtype=weight_dtype) for a, b in zip(down_block_res_samples_a, down_block_res_samples_b) ]
                mid_block_res_sample = (mid_block_res_sample_a + mid_block_res_sample_b).to(dtype=weight_dtype)
                
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
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
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
