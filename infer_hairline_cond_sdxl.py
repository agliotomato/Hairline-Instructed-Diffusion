import argparse
import os
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    AutoencoderKL,
    DDPMScheduler
)
from diffusers.utils import load_image
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="SDXL Hybrid ControlNet Inference")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--controlnet_a_path", type=str, required=True, help="Path to Geometry ControlNet")
    parser.add_argument("--controlnet_b_path", type=str, required=True, help="Path to Identity ControlNet")
    parser.add_argument("--bald_path", type=str, required=True, help="Path to input bald image (used as base and condition)")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to semantic mask")
    parser.add_argument("--prompt", type=str, default="high quality, realistic hairstyle, detailed hair, 8k")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, deformed, artificial")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--strength", type=float, default=0.9, help="Noise strength (0.0=original, 1.0=full generation)")
    parser.add_argument("--controlnet_scales", type=float, nargs="+", default=[1.0, 1.0])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output_sdxl_inference.png")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device == "cuda" else torch.float32

    # 1. Load Models
    print("⏳ Loading Models...")
    vae = AutoencoderKL.from_pretrained(args.vae_model_name_or_path, torch_dtype=weight_dtype).to(device)
    controlnet_a = ControlNetModel.from_pretrained(args.controlnet_a_path, torch_dtype=weight_dtype).to(device)
    controlnet_b = ControlNetModel.from_pretrained(args.controlnet_b_path, torch_dtype=weight_dtype).to(device)
    
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=[controlnet_a, controlnet_b],
        vae=vae,
        torch_dtype=weight_dtype,
        variant="fp16",
        use_safetensors=True
    ).to(device)
    
    pipe.enable_xformers_memory_efficient_attention()

    # 2. Preprocess Inputs
    print("⏳ Preprocessing Inputs...")
    bald_image = Image.open(args.bald_path).convert("RGB").resize((args.resolution, args.resolution))
    mask_image = Image.open(args.mask_path).convert("RGB").resize((args.resolution, args.resolution))
    
    # SDXL ControlNet Img2Img Pipeline
    # image: base image to add noise to
    # control_image: list of images for controlnets
    control_images = [mask_image, bald_image]

    # 3. Inference
    print(f"🚀 Generating Image with prompt: {args.prompt} (Strength: {args.strength})")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        image=bald_image,
        control_image=control_images,
        strength=args.strength,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scales,
        generator=generator,
    ).images[0]

    # 4. Save
    image.save(args.output_path)
    print(f"✅ Image saved to {args.output_path}")

if __name__ == "__main__":
    main()
