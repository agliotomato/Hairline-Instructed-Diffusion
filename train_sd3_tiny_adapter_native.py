
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import StableDiffusion3Pipeline, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast
import sys
import os

# Fix for ModuleNotFoundError when running via accelerate
sys.path.append(os.getcwd())

# Custom Imports
from modules.tiny_adapter_native import TinyAdapterNative
from utils.hairline_dataset_v2 import HairlineDatasetV2

def main():
    parser = argparse.ArgumentParser(description="Train TinyAdapterNative for SD 3.5")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--output_dir", type=str, default="output/tiny_adapter_native_checkpoints")
    parser.add_argument("--orig_dir", type=str, required=True)
    parser.add_argument("--bald_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=1024, help="Native resolution for SD3.5")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        log_with="tensorboard",
        project_dir=args.output_dir
    )
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training TinyAdapterNative on {args.resolution}x{args.resolution} inputs...")
        accelerator.init_trackers("tiny_adapter_native")

    # 2. Load Models
    # Load SD3.5 Components
    # We only need VAE and Transformer for training. Text Encoders are frozen.
    # Actually we need Text Encoders to encode prompts.
    
    # Load Pipeline to easily get components (efficient way for now)
    # Using float16/bf16 for base model to save memory
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    print("Loading SD3.5 Pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    )
    
    # Freeze Pipeline Components
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder_3.requires_grad_(False)
    
    # Move to device handled by accelerator later, but for now lets keep them accessible
    # We will strip components from pipe for easier handling
    vae = pipe.vae
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    
    # Clean up pipe to save memory if possible (optional)
    del pipe
    
    # 3. Initialize Adapter (Trainable)
    print("Initializing TinyAdapterNative...")
    adapter = TinyAdapterNative(input_channels=1, base_channels=32, output_channels=16)
    adapter.train()

    # 4. Dataset
    dataset = HairlineDatasetV2(
        orig_dir=args.orig_dir,
        bald_dir=args.bald_dir, # Required by V2 but we might only use orig/mask for simple training
        mask_dir=args.mask_dir,
        resolution=args.resolution
    )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

    # 6. Prepare with Accelerator
    adapter, optimizer, dataloader = accelerator.prepare(
        adapter, optimizer, dataloader
    )
    
    # Move frozen models to device
    vae.to(accelerator.device)
    transformer.to(accelerator.device)
    # Text encoders need to be on device too. 
    # Since we deleted pipe, we need to handle text encoding carefully.
    # Actually, let's keep pipe for text encoding to avoid complexity re-implementing it.
    # Reloading pipe for simplicity of 'encode_prompt'
    # NOTE: Efficient way is to pre-compute embeddings if dataset is small.
    # For simplicity here, we re-instantiate pipe or just use it from before.
    # Let's revert the 'del pipe' strategy and use pipe.encode_prompt.
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype
    ).to(accelerator.device)
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.text_encoder_3.requires_grad_(False)

    # 7. Training Loop
    global_step = 0
    
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}")
        for step, batch in enumerate(tqdm(dataloader)):
            with accelerator.accumulate(adapter):
                # A. Encode Images to Latents
                pixel_values = batch["orig_pixel_values"].to(dtype=weight_dtype)
                
                with torch.no_grad():
                    # SD3 VAE Encoding
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
                
                # B. Sample Noise & Timesteps (Flow Matching)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random times u in [0, 1] - equivalent to sigma in Rectified Flow
                # Bias towards middle/high noise for better learning? Standard is Uniform.
                u = torch.rand((bsz,), device=latents.device)
                
                # In SD3, timestep is effectively 1000 * (1 - sigma)? Or 1000 * u?
                # FlowMatchEulerDiscreteScheduler: sigmas go from 1.0 down to 0.0.
                # timestep 1000 -> sigma 1.0 (noise)
                # timestep 0 -> sigma 0.0 (image)
                # So t = u * 1000.
                # sigma = u (if we define u=1 as noise)
                # But typically scheduler.sigmas[t] maps t to sigma.
                # Let's align with SD3 conventions:
                # t input to model is 0-1000.
                timesteps = (u * 1000).long()
                
                # Sigma for interpolation
                # We align sigma with u. sigma=0(image), sigma=1(noise).
                # Noisy = (1-sigma)x + sigma*epsilon
                # This matches "timestep 1000" being pure noise.
                sigmas = u.view(bsz, 1, 1, 1)
                
                # C. Add Noise (Manual Rectified Flow)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                
                # D. Adapter Forward (Native Resolution 1024 -> 128)
                mask = batch["hair_mask"].to(dtype=weight_dtype) 
                # Mask [B, 1, 1024, 1024]
                
                adapter_features = adapter(mask) 
                # Adapter Output [B, 16, 128, 128]
                
                # E. Injection (Additive)
                model_input = noisy_latents + adapter_features
                
                # CASTING: Ensure model_input matches Transformer's dtype (e.g. bf16)
                model_input = model_input.to(dtype=weight_dtype)

                # F. Text Encoding
                prompts = batch["prompt"]
                with torch.no_grad():
                    # SD3 Encoding
                    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                        prompt=prompts,
                        prompt_2=prompts,
                        prompt_3=prompts,
                        device=accelerator.device,
                        do_classifier_free_guidance=False
                    )
                
                # G. Model Prediction
                # SD3 Transformer Forward
                noise_pred = pipe.transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False
                )[0]
                
                # H. Loss Calculation (Flow Matching: Target is Noise - Latents usually, depends on formulation)
                # SD3 Flow Match Euler: Target is (noise - latents) usually "v-prediction"
                # But simplest is to ask scheduler what the target is for MSE
                # For Flow Matching, target is (noise - x_0) or (x_0 - noise).
                # Official SD3 training target is 'noise - x_start' (Flow velocity).
                target = noise - latents
                
                loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                
                # Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(adapter.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            
            # Logging
            if global_step % 10 == 0 and accelerator.is_main_process:
                print(f"Step {global_step} | Loss: {loss.item():.4f}")
                accelerator.log({"train_loss": loss.item()}, step=global_step)
            
            global_step += 1
            
            # Checkpointing
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pth")
                    torch.save(adapter.state_dict(), save_path)
                    print(f"Saved checkpoint to {save_path}")

    # Final Save
        torch.save(adapter.state_dict(), save_path)
        print(f"Saved final model to {save_path}")
    
    accelerator.end_training()

if __name__ == "__main__":
    main()
