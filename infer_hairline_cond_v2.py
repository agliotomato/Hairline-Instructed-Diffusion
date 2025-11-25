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
from utils.hairline_conditioning import HairlineConditioningEmbeddings


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


def load_conditioner(
    path: Path | None, hidden_size: int, device: torch.device, dtype: torch.dtype
) -> HairlineConditioningEmbeddings:
    state = None
    use_bald_token = True
    if path and path.exists():
        state = torch.load(path, map_location="cpu")
        hidden_size = state.get("hidden_size", hidden_size)
        if state.get("use_bald_token", True) is False:
            print("conditioner checkpoint had use_bald_token=False, overriding to True for inference.")

    conditioner = HairlineConditioningEmbeddings(hidden_size=hidden_size, use_bald_token=use_bald_token)
    if state:
        conditioner.load_state_dict(state["state_dict"], strict=False)
    return conditioner.to(device=device, dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="Inference for the hairline-conditioned UNet (x0 = z_orig).")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to fine-tuned UNet weights.")
    parser.add_argument("--conditioner_path", type=str, default=None, help="Path to conditioner weights (conditioner.pt).")
    parser.add_argument("--bald_path", type=str, required=True, help="Path to an aligned bald image.")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the forehead mask.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
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

    conditioner_path = Path(args.conditioner_path) if args.conditioner_path else None
    if conditioner_path is None and args.model_dir:
        maybe = Path(args.model_dir).parent / "conditioner.pt"
        if maybe.exists():
            conditioner_path = maybe
    conditioner = load_conditioner(conditioner_path, text_encoder.config.hidden_size, device, weight_dtype)

    if args.model_dir:
        unet = UNet2DConditionModel.from_pretrained(
            args.model_dir,
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

    if args.init_latent == "noise":
        latents = torch.randn(
            (args.num_samples, 4, 64, 64), device=device, dtype=weight_dtype, generator=generator
        )
        latents = latents * noise_scheduler.init_noise_sigma
    elif args.init_latent == "zbald":
        noise = torch.randn_like(z_bald, generator=generator)
        noise = noise * noise_scheduler.init_noise_sigma
        latents = z_bald + args.noise_strength * noise  # identity-preserving init: start near z_bald
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

    cond_tokens = conditioner(mask_latent, z_bald)
    cond_states = torch.cat([cond_tokens, text_embeddings], dim=1)
    uncond_states = None
    if use_guidance:
        uncond_states = torch.cat([cond_tokens, uncond_embeddings], dim=1)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=weight_dtype, enabled=device.type == "cuda"
            ):
                model_input = torch.cat([latents, mask_latent], dim=1)
                if use_guidance:
                    noise_pred_uncond = unet(
                        model_input,
                        t,
                        encoder_hidden_states=uncond_states,
                    ).sample
                    noise_pred_text = unet(
                        model_input,
                        t,
                        encoder_hidden_states=cond_states,
                    ).sample
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = unet(
                        model_input,
                        t,
                        encoder_hidden_states=cond_states,
                    ).sample

                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        images = vae.decode(latents).sample


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

    cond_tokens = conditioner(mask_latent, z_bald)
    cond_states = torch.cat([cond_tokens, text_embeddings], dim=1)
    uncond_states = None
    if use_guidance:
        uncond_states = torch.cat([cond_tokens, uncond_embeddings], dim=1)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda", dtype=weight_dtype, enabled=device.type == "cuda"
            ):
                model_input = torch.cat([latents, mask_latent], dim=1)
                if use_guidance:
                    noise_pred_uncond = unet(
                        model_input,
                        t,
                        encoder_hidden_states=uncond_states,
                    ).sample
                    noise_pred_text = unet(
                        model_input,
                        t,
                        encoder_hidden_states=cond_states,
                    ).sample
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = unet(
                        model_input,
                        t,
                        encoder_hidden_states=cond_states,
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
