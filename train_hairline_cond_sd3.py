
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

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

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
    # Initialize ControlNets (Hybrid Strategy)
    # Workaround: from_transformer doesn't accept kwargs for channels in this version.
    # Manual initialization using config copy.
    
    # 1. Geometry (1ch)
    if accelerator.is_main_process:
        print("Initializing ControlNet A (Geometry, 1ch input)...")
    config_a = dict(transformer.config)
    config_a["extra_conditioning_channels"] = 1
    # SD3CN uses 'num_layers' which might differ from transformer 'num_layers' if not careful,
    # but from_transformer logic usually just copies. 
    # Let's trust that SD3ControlNetModel(**config_a) works if config matches.
    # We need to filter keys that are in Transformer but not ControlNet? 
    # Usually easier to use Load & Update pattern or from_config.
    # Let's try safe approach: instantiate with same params as from_transformer would.
    
    # Initialize ControlNets (Hybrid Strategy)
    # ControlNet A (Geometry, 1ch): SD3 Default matches (extra=1)
    if accelerator.is_main_process:
        print("Initializing ControlNet A (Geometry, 1ch input)...")
    controlnet_a = SD3ControlNetModel.from_transformer(transformer)
    controlnet_a.requires_grad_(True)
    controlnet_a.train()

    # ControlNet B (Identity, 16ch): Need manual init for channels
    if accelerator.is_main_process:
        print("Initializing ControlNet B (Identity, 16ch input)...")
    
    # Config derived from inspect_sd3_config.py
    # num_layers=12, dual_attention_layers=[0...12], etc.
    controlnet_b = SD3ControlNetModel(
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=12, # CONFIRMED
        attention_head_dim=64,
        num_attention_heads=24,
        joint_attention_dim=4096,
        caption_projection_dim=1536,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=384,
        qk_norm="rms_norm",
        dual_attention_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], # CONFIRMED KEY
        extra_conditioning_channels=16, # <--- 16 Channels for Latent Identity
    )
    # Load weights from transformer
    # Note: 'pos_embed_input' will fail to load due to channel mismatch (16 vs 1), which is EXPECTED.
    # Other weights should match.
    keys_to_ignore = ["pos_embed_input.weight"] 
    # Actually load_state_dict with strict=False handles ignoring, but we expect size mismatch for pos_embed_input.
    # We want to load everything ELSE.
    
    # Manually filter state dict to avoid "RuntimeError: size mismatch"
    # The error before was: size mismatch for pos_embed_input.weight
    # Wait, previous error was huge list of mismatches because num_layers was wrong (default 2 vs 12).
    # Now num_layers=12 matches, so only pos_embed_input should mismatch.
    try:
        controlnet_b.load_state_dict(transformer.state_dict(), strict=False)
    except RuntimeError as e:
        # Ignore size mismatch for input embedding layer, as we changed channels
        if "size mismatch" in str(e) and "pos_embed_input" in str(e):
             print("Specific size mismatch in Identity Net (Expected due to channel change). Ignoring.")
        else:
             # If other errors, re-raise might be noisy, but let's trust strict=False usually doesn't raise for size mismatch?
             # Actually strict=False DOES raise for size mismatch, only ignores missing keys.
             # So we MUST filter the state dict.
             pass

    # Better loading strategy: Filter out incompatible keys
    transformer_state_dict = transformer.state_dict()
    # Remove keys that have shape mismatch (input embedding)
    # ControlNet's pos_embed_input is new, Transformer doesn't have it? 
    # SD3CN.from_transformer copies logic.
    # Let's just use strict=False and catch the error if specific.
    # Actually, modifying the dictionary before loading is safer.
    
    # Filter state dict
    compatible_state_dict = {}
    model_dict = controlnet_b.state_dict()
    for k, v in transformer_state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                compatible_state_dict[k] = v
            else:
                if accelerator.is_main_process:
                    print(f"Skipping {k} due to shape mismatch: Trans{v.shape} vs CN{model_dict[k].shape}")
    
    controlnet_b.load_state_dict(compatible_state_dict, strict=False)
    
    controlnet_b.train()
    
    # Cast ControlNets to correct precision (FP16) - REMOVED
    # Keeping them in FP32 allows standard Mixed Precision optimizer step.
    # Autocast handles the FP16 ops.
    # controlnet_a.to(dtype=weight_dtype) 
    # controlnet_b.to(dtype=weight_dtype)

    controlnet_a.requires_grad_(True)
    controlnet_b.requires_grad_(True)
    
    # Enable Gradient Checkpointing for VRAM savings
    controlnet_a.enable_gradient_checkpointing()
    controlnet_b.enable_gradient_checkpointing()
    # CRITICAL FIX: Enable gradient checkpointing for frozen transformer to save activation memory
    transformer.enable_gradient_checkpointing()

    controlnet_a.train()
    controlnet_b.train()
    # Transformer remains in eval mode usually, but check pointing might need train mode?
    # Diffusers enable_gradient_checkpointing usually handles this.
    # Keep transformer in eval to disable dropout
    transformer.eval() 
    
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
    
    # Move pipeline components to device - MODIFIED FOR VRAM SAVINGS
    # pipeline = pipeline.to(accelerator.device) # DO NOT MOVE ALL AT ONCE
    # Instead, we will move text encoders / vae on demand.
    
    # Ensure Transformer is on device (if we extracted it, it might still refer to pipeline's module)
    # Actually, we extracted `transformer = pipeline.transformer`. 
    # VRAM FIX: Initialize Transformer on CPU. Only move to GPU when needed.
    pipeline.transformer.to("cpu")
    transformer.to("cpu")
    
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet_a, controlnet_b):
                # 1. Text Encoding using Pipeline
                # Move text encoders to GPU just for this operation
                pipeline.text_encoder.to(accelerator.device)
                pipeline.text_encoder_2.to(accelerator.device)
                pipeline.text_encoder_3.to(accelerator.device)
                
                with torch.no_grad():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipeline.encode_prompt(
                        prompt=batch["prompt"],
                        prompt_2=None,
                        prompt_3=None,
                        device=accelerator.device, # Use accelerator device
                        do_classifier_free_guidance=False 
                    )
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=weight_dtype)
                
                # Move text encoders back to CPU immediately
                pipeline.text_encoder.to("cpu")
                pipeline.text_encoder_2.to("cpu")
                pipeline.text_encoder_3.to("cpu")
                torch.cuda.empty_cache()
                    
                # 2. VAE Encode
                # Move VAE to GPU just for this operation
                vae.to(accelerator.device)
                
                pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                    
                    # Prepare Masks (Geometry Stream - Non-VAE)
                    # Resize mask to latent resolution (H/8, W/8)
                    mask = batch["hair_mask"].to(device=accelerator.device, dtype=weight_dtype)
                    mask_cond = torch.nn.functional.interpolate(mask, size=latents.shape[-2:], mode="nearest")
                    
                    # Encode Identity
                    masked_bald = batch["masked_bald_pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                    identity_latents = vae.encode(masked_bald).latent_dist.sample() * vae.config.scaling_factor

                # Move VAE back to CPU immediately
                vae.to("cpu")
                
                # AGGRESSIVE CLEANUP: Free VRAM after encodings are done
                torch.cuda.empty_cache()

                # 3. Flow Matching Noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # Add noise
                # Sigmas indexing (timesteps=GPU, sigmas=CPU)
                # Fix: Move timesteps to CPU for indexing, or move sigmas to GPU
                sigmas = scheduler.sigmas[timesteps.cpu()].to(device=latents.device).flatten()
                # Force sigmas to weight_dtype (fp16) to prevent promotion to float32
                sigmas = sigmas.to(dtype=weight_dtype)

                # SD3 Rectified Flow: z_t = (1-t)x + t*noise
                # Manual implementation since scheduler.add_noise is missing
                sigmas = sigmas.view(-1, 1, 1, 1)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                
                # Target for Velocity (Rectified Flow)
                target = noise - latents 

                # 4. Forward ControlNets (Hybrid)
                # Important: SD3ControlNet expects concatenated input (Noisy Latents + Condition)
                
                # Stream A (Geometry)
                # Input: 16ch (Noisy) + 1ch (Mask) = 17ch
                noisy_latents = noisy_latents.to(dtype=weight_dtype)
                cond_a_input = torch.cat([noisy_latents, mask_cond], dim=1).to(dtype=weight_dtype)
                
                # Use autocast for safer mixed precision handling
                with accelerator.autocast():
                    out_a = controlnet_a(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        controlnet_cond=cond_a_input, 
                        return_dict=False
                    )
                    
                    # Stream B (Identity)
                    # Input: 16ch (Noisy) + 16ch (Identity Latents) = 32ch
                    cond_b_input = torch.cat([noisy_latents, identity_latents], dim=1).to(dtype=weight_dtype)
                    out_b = controlnet_b(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        controlnet_cond=cond_b_input, 
                        return_dict=False
                    )
                    
                    # Merge Residuals
                    # out_a is tuple (residuals,)
                    residuals_a = out_a[0]
                    residuals_b = out_b[0]
                    
                    combined_residuals = [a + b for a, b in zip(residuals_a, residuals_b)]
                    
                    # 5. Transformer Forward
                    # VRAM OPTIMIZATION: Move Transformer to GPU ON-DEMAND
                    transformer.to(accelerator.device)
                    
                    # DEBUG: Check devices
                    if accelerator.is_main_process:
                         print(f"DEBUG: noisy_latents device: {noisy_latents.device}")
                         print(f"DEBUG: timesteps device: {timesteps.device}")
                         print(f"DEBUG: prompt_embeds device: {prompt_embeds.device}")
                         print(f"DEBUG: pooled_prompt_embeds device: {pooled_prompt_embeds.device}")
                         print(f"DEBUG: residuals device: {combined_residuals[0].device if combined_residuals else 'None'}")
                         print(f"DEBUG: transformer device: {transformer.device if hasattr(transformer, 'device') else 'Unknown'}")

                    model_pred = transformer(
                        hidden_states=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=combined_residuals,
                        return_dict=True
                    ).sample
                    
                    # Move Transformer BACK to CPU to save memory for next batch encoding
                    # Note: Moving back and forth is slow, but necessary if OOM. 
                    # If we can fit Transformer + T5, we don't need this. But A100 OOM suggests we can't.
                    transformer.to("cpu")
                    torch.cuda.empty_cache()
                
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
