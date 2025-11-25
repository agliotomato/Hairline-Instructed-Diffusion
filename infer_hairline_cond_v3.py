from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms

from utils.hair_mask_utils import enable_hairline_conditioning
from utils.latent_identity_net import ControlNetModel


def preprocess_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.BILINEAR)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    return transform(image).unsqueeze(0)


def preprocess_mask(path: str, resolution: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = mask.resize((resolution, resolution), Image.BILINEAR)
    tensor = transforms.ToTensor()(mask).unsqueeze(0)
    return torch.clamp(tensor, 0.0, 1.0)


def encode_texts(
    tokenizer: AutoTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    num_images: int,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).to(device)
    text_embeddings = text_encoder(text_inputs.input_ids)[0].repeat(num_images, 1, 1)

    uncond_embeddings = None
    if negative_prompt is not None:
        uncond_inputs = tokenizer(
            [negative_prompt],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)
        uncond_embeddings = text_encoder(uncond_inputs.input_ids)[0].repeat(num_images, 1, 1)

    return text_embeddings, uncond_embeddings


def main():
    parser = argparse.ArgumentParser(description="Inference for the hairline-conditioned UNet v3 (Latent IdentityNet).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--unet_path", type=str, default=None, help="Path to fine-tuned UNet weights.")
    parser.add_argument("--controlnet_path", type=str, default=None, help="Path to trained ControlNet weights.")
    parser.add_argument("--bald_path", type=str, required=True, help="Path to an aligned bald image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the forehead mask.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--init_latent", type=str, choices=["noise", "zbald"], default="noise")
    parser.add_argument("--noise_strength", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.dtype == "fp16":
        weight_dtype = torch.float16 if device.type == "cuda" else torch.float32
    elif args.dtype == "bf16":
        weight_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        weight_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    ).to(device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(
        device, dtype=weight_dtype
    )

    # Load Main UNet
    if args.unet_path:
        unet = UNet2DConditionModel.from_pretrained(
            args.unet_path,
            in_channels=5,
            low_cpu_mem_usage=False,
        )
        unet.config.base_in_channels = 4
        unet.config.hair_conditioning_channels = 1
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
        )
        unet = enable_hairline_conditioning(unet, mask_channels=1)
    unet.to(device, dtype=weight_dtype)

    # Load Latent IdentityNet (ControlNet)
    if args.controlnet_path:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_path).to(device, dtype=weight_dtype)
    else:
        print("Warning: No ControlNet path provided. Initializing from UNet (untrained).")
        controlnet = ControlNetModel.from_unet(unet, load_weights_from_unet=True).to(device, dtype=weight_dtype)

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator.seed()

    image_tensor = preprocess_image(args.bald_path, args.resolution).to(device, dtype=weight_dtype)
    mask_tensor = preprocess_mask(args.mask_path, args.resolution).to(device, dtype=weight_dtype)

    with torch.no_grad():
        z_bald = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor

    z_bald = z_bald.repeat(args.num_samples, 1, 1, 1)
    mask_latent = F.interpolate(mask_tensor, size=(64, 64), mode="bilinear", align_corners=False)
    mask_latent = mask_latent.repeat(args.num_samples, 1, 1, 1).to(device=device, dtype=weight_dtype)

    # Init Latents
    if args.init_latent == "noise":
        latents = torch.randn(
            (args.num_samples, 4, 64, 64), device=device, dtype=weight_dtype, generator=generator
        )
        latents = latents * noise_scheduler.init_noise_sigma
    elif args.init_latent == "zbald":
        noise = torch.randn_like(z_bald, generator=generator)
        noise = noise * noise_scheduler.init_noise_sigma
        latents = z_bald + args.noise_strength * noise
    else:
        raise ValueError(f"Unsupported init_latent option: {args.init_latent}")

    use_guidance = args.guidance_scale > 1.0

    text_embeddings, uncond_embeddings = encode_texts(
        tokenizer,
        text_encoder,
        args.prompt,
        args.negative_prompt if use_guidance else None,
        device,
        args.num_samples,
    )
    text_embeddings = text_embeddings.to(dtype=weight_dtype)
    if uncond_embeddings is not None:
        uncond_embeddings = uncond_embeddings.to(dtype=weight_dtype)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=weight_dtype, enabled=device.type == "cuda"
            ):
                # Prepare Main UNet Input
                model_input = torch.cat([latents, mask_latent], dim=1)
                
                # Latent IdentityNet Forward
                # controlnet_cond = z_bald
                down_block_res_samples, mid_block_res_sample = controlnet(
                    sample=latents,
                    timestep=t,
                    encoder_hidden_states=text_embeddings, # Using text embeddings for ControlNet too? Stable-Hair uses it.
                    controlnet_cond=z_bald,
                    return_dict=False,
                )

                if use_guidance:
                    # Unconditional pass
                    # For ControlNet, we might want to drop condition or use uncond embeddings?
                    # Stable-Hair implementation usually runs ControlNet twice or uses cfg on ControlNet output.
                    # Here, let's run ControlNet for uncond as well if we want full CFG.
                    # Or we can reuse the same control features if we assume Identity is constant?
                    # Standard ControlNet practice: apply to both cond and uncond, or just cond.
                    # Let's apply to both for consistency.
                    
                    down_block_res_samples_uncond, mid_block_res_sample_uncond = controlnet(
                        sample=latents,
                        timestep=t,
                        encoder_hidden_states=uncond_embeddings,
                        controlnet_cond=z_bald, # Should we drop z_bald for uncond? Usually no, we want to keep structure.
                        return_dict=False,
                    )

                    noise_pred_uncond = unet(
                        model_input,
                        t,
                        encoder_hidden_states=uncond_embeddings,
                        down_block_additional_residuals=down_block_res_samples_uncond,
                        mid_block_additional_residual=mid_block_res_sample_uncond,
                    ).sample

                    noise_pred_text = unet(
                        model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = unet(
                        model_input,
                        t,
                        encoder_hidden_states=text_embeddings,
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
                    ).sample

                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(latents).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx in range(args.num_samples):
        img = transforms.ToPILImage()(images[idx])
        img.save(out_dir / f"sample_{timestamp}_{idx:03d}.png")

    print(f"Saved {args.num_samples} samples to {out_dir}")


if __name__ == "__main__":
    main()
