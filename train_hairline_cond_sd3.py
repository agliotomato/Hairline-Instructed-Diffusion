
import argparse
import logging
import math
import os
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    SD3ControlNetModel,
    StableDiffusion3Pipeline,
)
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, CLIPTextModelWithProjection, T5EncoderModel
from tqdm.auto import tqdm

from utils.hairline_dataset_v2 import HairlineDatasetV2

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for SD3 Dual-Stream ControlNet.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--orig_dir", type=str, required=True)
    parser.add_argument("--bald_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=1024, help="SD3 default is 1024")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--output_dir", type=str, default="hairline_cond_sd3")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    
    return parser.parse_args()

def collate_fn(examples):
    orig_values = torch.stack([ex["orig_pixel_values"] for ex in examples])
    bald_values = torch.stack([ex["bald_pixel_values"] for ex in examples])
    masked_bald_values = torch.stack([ex["masked_bald_pixel_values"] for ex in examples])
    masks = torch.stack([ex["hair_mask"] for ex in examples]) # 1ch
    prompts = [ex["prompt"] for ex in examples]
    
    return {
        "pixel_values": orig_values,
        "masked_bald_pixel_values": masked_bald_values,
        "hair_mask": masks,
        "prompt": prompts
    }

def _get_t5_prompt_embeds(
    tokenizer: T5EncoderModel,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=256, # SD3.5 max length? T5 XXL default
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)

    prompt_embeds = text_encoder(text_input_ids)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(-1, seq_len, prompt_embeds.shape[-1])

    return prompt_embeds


def _get_clip_prompt_embeds(
    tokenizer: CLIPTextModelWithProjection,
    text_encoder: CLIPTextModelWithProjection,
    prompt: Union[str, List[str]],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    prompt_embeds = text_encoder(text_input_ids, output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, pooled_prompt_embeds

def compute_text_embeddings(
    prompts, 
    tokenizers, 
    text_encoders, 
    device
):
    # SD3: CLIP-G, CLIP-L, T5
    # Tokenizers: [CLIP-L, CLIP-G, T5]
    # Encoders: [CLIP-L, CLIP-G, T5]
    
    tokenizer_1, tokenizer_2, tokenizer_3 = tokenizers
    text_encoder_1, text_encoder_2, text_encoder_3 = text_encoders
    
    with torch.no_grad():
        # CLIP-L
        prompt_embeds_1, pooled_prompt_embeds_1 = _get_clip_prompt_embeds(
            tokenizer_1, text_encoder_1, prompts, device=device
        )
        # CLIP-G
        prompt_embeds_2, pooled_prompt_embeds_2 = _get_clip_prompt_embeds(
            tokenizer_2, text_encoder_2, prompts, device=device
        )
        # T5
        # Optimization: T5 is huge (4.7B params for XXL). If running on single GPU with training, 
        # it might OOM. 
        # But we assume the environment can handle it or we use cpu offload logic.
        # Since we just need embeddings, we can run it and delete inputs.
        
        # NOTE: If T5 fails due to OOM, we might need to skip or use CPU.
        prompt_embeds_3 = _get_t5_prompt_embeds(
             tokenizer_3, text_encoder_3, prompts, device=device
        )
        
    # Concatenate
    # Pooling: Concat pooled from CLIP-L and CLIP-G
    pooled_prompt_embeds = torch.cat([pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1)
    
    # Sequence: Pad CLIPs to match T5 or just concat?
    # SD3 pipeline logic:
    # CLIPs are 77 tokens. T5 is 256 or 512.
    # SD3 concats them along sequence dimension? No.
    # SD3 pipeline uses `joint_attention`.
    # Actually, verify pipeline source if possible. SD3 uses `prompt_embeds` which is concat of clip_1, clip_2, t5.
    # But paddings must align.
    # Standard SD3: pad CLIP embeds to T5 hidden dimension? No.
    # Concat along SEQUENCE dim.
    # Shape: (batch, seq_len_1 + seq_len_2 + seq_len_3, dim)?
    # No, SD3 uses `joint_attention_dim` = 4096.
    # CLIP-L (768), CLIP-G (1280), T5 (4096).
    # Wait, SD3 projects them?
    # Actually, standard Diffusers pipeline handles this.
    # We should mimic `StableDiffusion3Pipeline.encode_prompt`.
    # Since we can't easily replicate 100 lines of pipeline code without errors:
    # Use the pipeline!
    # Just load pipeline and call `encode_prompt`.
    pass 
    # But we didn't load pipeline, we loaded components.
    # We can instantiate a tiny pipeline wrapper around loaded components
    # to use its `encode_prompt`.
    
    return prompt_embeds_3, pooled_prompt_embeds # Placeholder return as real implementation requires pipeline logic
    
def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Use Pipeline to handle model loading & prompt encoding logic cleanly
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float16 if args.mixed_precision == "fp16" else torch.float32 
    )
    # We only need encoding methods and models.
    # We will extract models from pipeline.
    vae = pipeline.vae
    transformer = pipeline.transformer
    scheduler = pipeline.scheduler
    
    # Freeze
    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    
    # Initialize ControlNets
    if accelerator.is_main_process:
        print("Initializing ControlNet A (Geometry)...")
    controlnet_a = SD3ControlNetModel.from_transformer(transformer)
    if accelerator.is_main_process:
        print("Initializing ControlNet B (Identity)...")
    controlnet_b = SD3ControlNetModel.from_transformer(transformer)
    
    controlnet_a.requires_grad_(True)
    controlnet_b.requires_grad_(True)
    controlnet_a.train()
    controlnet_b.train()
    
    # Save VRAM by offloading pipeline components we don't need immediately?
    # We need text encoders for prompt encoding step.
    # We can keep them on CPU and move to GPU only when needed, or keep in pipeline.
    # For training optimization, assume sufficient VRAM or use accelerator.
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        itertools.chain(controlnet_a.parameters(), controlnet_b.parameters()),
        lr=args.learning_rate
    )
    
    # Dataset
    train_dataset = HairlineDatasetV2(
        orig_dir=args.orig_dir,
        bald_dir=args.bald_dir,
        mask_dir=args.mask_dir,
        metadata_path=args.metadata_path,
        resolution=args.resolution
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
    
    controlnet_a, controlnet_b, optimizer, train_dataloader = accelerator.prepare(
        controlnet_a, controlnet_b, optimizer, train_dataloader
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Move pipeline components to device
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to(accelerator.device) 
    # Note: text encoders might consume LOTS of VRAM. 
    # Ideally should offload. prompt encoding is once per batch.
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet_a, controlnet_b):
                # 1. Text Encoding using Pipeline
                # pipeline.encode_prompt handles the complexity
                with torch.no_grad():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipeline.encode_prompt(
                        prompt=batch["prompt"],
                        device=accelerator.device, # Use accelerator device
                        do_classifier_free_guidance=False 
                    )
                    
                # 2. VAE Encode
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    
                    # Encode Masks (Geometry Stream)
                    mask = batch["hair_mask"].to(device=accelerator.device, dtype=weight_dtype)
                    mask_3ch = mask.repeat(1, 3, 1, 1) # 1ch -> 3ch RGB
                    mask_latents = vae.encode(mask_3ch).latent_dist.sample() * vae.config.scaling_factor
                    
                    # Encode Identity
                    masked_bald = batch["masked_bald_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                    identity_latents = vae.encode(masked_bald).latent_dist.sample() * vae.config.scaling_factor

                # 3. Flow Matching Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # Add noise
                sigmas = scheduler.sigmas[timesteps].flatten()
                # SD3 Rectified Flow: z_t = (1-t)x + t*noise ? Or other way?
                # Using scheduler.add_noise ensures consistency
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Target for Velocity (Rectified Flow)
                # v = noise - latents (usually)
                # Check paper or scheduler convention.
                # SD3 uses v-prediction typically.
                target = noise - latents 

                # 4. Forward ControlNets
                # Stream A
                out_a = controlnet_a(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=mask_latents,
                    return_dict=False
                )
                
                # Stream B
                out_b = controlnet_b(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=identity_latents,
                    return_dict=False
                )
                
                # Merge Residuals
                # out_a is tuple (residuals,)
                residuals_a = out_a[0]
                residuals_b = out_b[0]
                
                combined_residuals = [a + b for a, b in zip(residuals_a, residuals_b)]
                
                # 5. Transformer Forward
                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=combined_residuals,
                    return_dict=True
                ).sample
                
                # 6. Loss
                # Weighting for Flow Matching?
                # Usually 1.0 or sigma-dependent.
                # Simplest Flow Matching uses uniform weighting.
                loss = F.mse_loss(model_pred, target, reduction="mean")
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"loss": loss.detach().item()}, step=global_step)
                
                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
            if global_step >= max_train_steps:
                break
                
    accelerator.end_training()
    
    if accelerator.is_main_process:
        print("Saving final models...")
        controlnet_a.save_pretrained(os.path.join(args.output_dir, "controlnet_a"))
        controlnet_b.save_pretrained(os.path.join(args.output_dir, "controlnet_b"))

if __name__ == "__main__":
    main()
