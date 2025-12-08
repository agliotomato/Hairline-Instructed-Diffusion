from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, ControlNetModel
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModel
from tqdm.auto import tqdm

from utils.hairline_dataset_v2 import HairlineDatasetV2

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training script for hairline conditioned diffusion v4 (Pixel-space ControlNet).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--orig_dir", type=str, required=True, help="Directory with original hair images.")
    parser.add_argument("--bald_dir", type=str, required=True, help="Directory with aligned bald images.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with 1-channel forehead masks (High Res).")
    parser.add_argument("--metadata_path", type=str, default=None, help="Optional JSON/CSV prompts file.")
    parser.add_argument("--metadata_text_key", type=str, default="prompt")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="hairline_cond_v4")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard", help="TensorBoard/W&B/etc.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to keep.")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None, help="Path to pretrained ControlNet/IdentityNet weights.")
    return parser.parse_args()


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    orig_values = torch.stack([example["orig_pixel_values"] for example in examples])
    bald_values = torch.stack([example["bald_pixel_values"] for example in examples])
    masks = torch.stack([example["hair_mask"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    return {
        "orig_pixel_values": orig_values,
        "bald_pixel_values": bald_values,
        "hair_mask": masks, # (B, 1, 512, 512)
        "prompt": prompts,
    }


def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir, total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.report_to.lower() != "none" else None,
        project_config=accelerator_project_config,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process and args.report_to.lower() != "none":
        accelerator.init_trackers("hairline_cond_v4", config=vars(args))

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    
    # Initialize Standard ControlNet (Pixel-space)
    if args.controlnet_model_name_or_path:
        logger.info(f"Loading pretrained ControlNet from {args.controlnet_model_name_or_path}...")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing ControlNet from UNet weights...")
        # Note: conditioning_channels=1 tells it to expect 1-channel input, 
        # but standard init usually assumes 3. We handle adaptation below.
        controlnet = ControlNetModel.from_unet(unet)
        
        # [Innovation V4 Pixel-space] 3-channel (RGB) -> 1-channel (Mask) Adaptation
        # The 'controlnet_cond_embedding' is the Tiny Encoder.
        # Its first layer 'conv_in' is usually Conv2d(3, 16, ...)
        
        logger.info("Adapting ControlNet Tiny Encoder for 1-channel Mask Input...")
        
        # Access the Tiny Encoder's first layer
        # Standard diffusers ControlNet structure: controlnet.controlnet_cond_embedding.conv_in
        old_conv = controlnet.controlnet_cond_embedding.conv_in
        
        # Create new conv with 1 input channel
        new_conv = nn.Conv2d(
            in_channels=1, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding
        )
        
        # Initialize weights (Kaiming Normal for shape/edge learning)
        nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(new_conv.bias, 0)
        
        # Replace the layer
        controlnet.controlnet_cond_embedding.conv_in = new_conv
        
        # Update config
        controlnet.config.conditioning_channels = 1
        
        logger.info("Successfully adapted ControlNet Tiny Encoder to accept 1-channel inputs.")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False) # Main UNet is FROZEN
    
    controlnet.train() # Only ControlNet trains

    vae.eval()
    text_encoder.eval()
    unet.eval()

    # Optimize only ControlNet parameters
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = HairlineDatasetV2(
        orig_dir=args.orig_dir,
        bald_dir=args.bald_dir,
        mask_dir=args.mask_dir,
        metadata_path=args.metadata_path,
        metadata_text_key=args.metadata_text_key,
        resolution=args.resolution,
    )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare with Accelerator
    # Note: UNet is not passed to prepare() because it's not being optimized
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move frozen models to device
    unet.to(accelerator.device)
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Cast frozen models to weight_dtype
    unet.to(dtype=weight_dtype)
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if accelerator.is_main_process:
        logger.info("***** Running training (V4 - Pixel-space ControlNet) *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size = {total_batch_size}")
        logger.info(f"  Total optimization steps = {max_train_steps}")

    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0

    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        path = Path(args.resume_from_checkpoint)
        if path.is_file():
            global_step = int(path.stem.split("-")[-1])
        else:
            global_step = int(path.name.split("-")[-1])
        progress_bar.update(global_step)

    for epoch in range(args.num_train_epochs):
        controlnet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert inputs to weight_dtype
                orig_pixel_values = batch["orig_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                
                # Hair Mask is used as ControlNet Condition (Pixel Space 512x512)
                hair_masks = batch["hair_mask"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    # Encode input image to latent
                    z_orig = vae.encode(orig_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    
                noise = torch.randn_like(z_orig)
                bsz = z_orig.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=z_orig.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(z_orig, noise, timesteps)

                # Prepare ControlNet Input
                # V4 Pixel-space: Use 512x512 mask DIRECTLY. No downsampling here.
                # The Tiny Encoder inside ControlNet will handle the downsampling.
                controlnet_cond = hair_masks

                # Prepare Text Embeddings
                text_inputs = tokenizer(
                    batch["prompt"],
                    padding="max_length",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device=accelerator.device)
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # 1. ControlNet Forward Pass
                # Returns 13 residuals (12 down + 1 mid)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                    return_dict=False,
                )

                # 2. Main UNet Forward Pass with Additive Injection
                # Using down_block_additional_residuals argument
                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                # 3. Compute Loss
                # Target is the noise
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())
                if accelerator.is_main_process:
                    accelerator.log({"train_loss": loss.detach().item()}, step=global_step)

                if args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(ckpt_dir)

            if global_step >= max_train_steps:
                break

        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save ControlNet
        controlnet_dir = os.path.join(args.output_dir, "controlnet")
        accelerator.unwrap_model(controlnet).save_pretrained(controlnet_dir)
        logger.info(f"Saved ControlNet weights to {controlnet_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
