import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from torch.cuda.amp import autocast
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTextModel

from utils.hair_mask_utils import enable_hairline_conditioning


def preprocess_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    tensor = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )(image)
    return tensor.unsqueeze(0)


def preprocess_mask(path: str, resolution: int) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize((resolution, resolution), Image.BILINEAR)
    tensor = transforms.ToTensor()(mask).unsqueeze(0)
    return torch.clamp(tensor, 0.0, 1.0)


def encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: Optional[str],
    device: torch.device,
    num_images: int,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    texts: List[str] = [prompt]
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    cond = text_encoder(inputs.input_ids.to(device))[0].to(dtype=dtype)
    cond = cond.repeat(num_images, 1, 1)

    uncond = None
    if negative_prompt is not None:
        neg_inputs = tokenizer(
            [negative_prompt],
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond = text_encoder(neg_inputs.input_ids.to(device))[0].to(dtype=dtype)
        uncond = uncond.repeat(num_images, 1, 1)

    return cond, uncond


def main():
    parser = argparse.ArgumentParser(description="Memory-friendly inference for hairline-conditioned UNet.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to fine-tuned UNet weights.")
    parser.add_argument("--bald_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=0.7, help="Noise strength for img2img [0, 1].")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    if args.num_samples != 1:
        raise ValueError("infer_hairline_cond2.py is optimized for batch size 1. Set --num_samples 1.")

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
    text_encoder.eval()

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(
        device, dtype=weight_dtype
    )
    vae.eval()

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
    unet.eval()

    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    image_tensor = preprocess_image(args.bald_path, args.resolution).to(device, dtype=weight_dtype)
    mask_tensor = preprocess_mask(args.mask_path, args.resolution).to(device, dtype=weight_dtype)

    generator = torch.Generator(device=device)
    if args.seed is not None:
        generator.manual_seed(args.seed)
    else:
        generator.seed()

    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

    mask_latent = F.interpolate(
        mask_tensor, size=latents.shape[-2:], mode="bilinear", align_corners=False
    )

    timesteps = noise_scheduler.timesteps
    strength = max(0.0, min(1.0, args.strength))
    init_timestep = min(max(1, int(len(timesteps) * strength)), len(timesteps))
    t_start = max(len(timesteps) - init_timestep, 0)
    timesteps = timesteps[t_start:]
    noise = torch.randn(latents.shape, device=latents.device, dtype=latents.dtype, generator=generator)
    if t_start > 0:
        latents = noise_scheduler.add_noise(latents, noise, timesteps[0])
    else:
        latents = noise * noise_scheduler.init_noise_sigma

    cond_embeds, uncond_embeds = encode_prompt(
        tokenizer,
        text_encoder,
        args.prompt,
        args.negative_prompt if args.guidance_scale > 1.0 else None,
        device,
        args.num_samples,
        weight_dtype,
    )

    use_guidance = args.guidance_scale > 1.0 and uncond_embeds is not None

    def amp_context():
        if device.type == "cuda" and weight_dtype in (torch.float16, torch.bfloat16):
            return autocast(dtype=weight_dtype)
        return nullcontext()

    with torch.no_grad():
        for t in timesteps:
            with amp_context():
                if use_guidance:
                    noise_pred_uncond = unet(
                        latents,
                        t,
                        encoder_hidden_states=uncond_embeds,
                        hair_mask=mask_latent,
                    ).sample
                    noise_pred_text = unet(
                        latents,
                        t,
                        encoder_hidden_states=cond_embeds,
                        hair_mask=mask_latent,
                    ).sample
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred = unet(
                        latents,
                        t,
                        encoder_hidden_states=cond_embeds,
                        hair_mask=mask_latent,
                    ).sample
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

        latents = latents / vae.config.scaling_factor
        with amp_context():
            images = vae.decode(latents).sample

    images = (images / 2 + 0.5).clamp(0, 1).cpu()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img = transforms.ToPILImage()(images[0])
    img.save(out_dir / "sample_000.png")
    print(f"Saved sample_000.png to {out_dir}")


if __name__ == "__main__":
    main()
