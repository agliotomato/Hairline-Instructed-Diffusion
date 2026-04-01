import argparse
import gradio as gr
import torch
from PIL import Image
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import os
import cv2
from pathlib import Path
from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel
from ref_encoder.adapter import *
from ref_encoder.reference_unet import ref_unet
from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline

def concatenate_images(image_files, output_file, type="pil"):
    if type == "np":
        image_files = [Image.fromarray(img) for img in image_files]
    images = image_files  # list
    max_height = max(img.height for img in images)
    images = [img.resize((img.width, max_height)) for img in images]
    total_width = sum(img.width for img in images)
    combined = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    combined.save(output_file)

class StableHair:
    def __init__(self, config="stable_hair/configs/hair_transfer.yaml", device="cuda", weight_dtype=torch.float16) -> None:
        print("Initializing Stable Hair Pipeline...")
        self.config = OmegaConf.load(config)
        self.device = device

        ### Load controlnet
        unet = UNet2DConditionModel.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        controlnet = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.controlnet_path))
        controlnet.load_state_dict(_state_dict, strict=False)
        controlnet.to(weight_dtype)

        ### >>> create pipeline >>> ###
        self.pipeline = StableHairPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=weight_dtype,
        ).to(device)
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)

        ### load Hair encoder/adapter
        self.hair_encoder = ref_unet.from_pretrained(self.config.pretrained_model_path, subfolder="unet").to(device)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.encoder_path))
        self.hair_encoder.load_state_dict(_state_dict, strict=False)
        self.hair_adapter = adapter_injection(self.pipeline.unet, device=self.device, dtype=torch.float16, use_resampler=False)
        _state_dict = torch.load(os.path.join(self.config.pretrained_folder, self.config.adapter_path))
        self.hair_adapter.load_state_dict(_state_dict, strict=False)

        ### load bald converter
        bald_converter = ControlNetModel.from_unet(unet).to(device)
        _state_dict = torch.load(self.config.bald_converter_path)
        bald_converter.load_state_dict(_state_dict, strict=False)
        bald_converter.to(dtype=weight_dtype)
        del unet

        ### create pipeline for hair removal
        self.remove_hair_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.config.pretrained_model_path,
            controlnet=bald_converter,
            safety_checker=None,
            torch_dtype=weight_dtype,
        )
        self.remove_hair_pipeline.scheduler = UniPCMultistepScheduler.from_config(
            self.remove_hair_pipeline.scheduler.config)
        self.remove_hair_pipeline = self.remove_hair_pipeline.to(device)

        ### move to fp16
        self.hair_encoder.to(weight_dtype)
        self.hair_adapter.to(weight_dtype)

        print("Initialization Done!")

    def Hair_Transfer(self, source_image, reference_image, random_seed, step, guidance_scale, scale, controlnet_conditioning_scale, size=512):
        prompt = ""
        n_prompt = ""
        random_seed = int(random_seed)
        step = int(step)
        guidance_scale = float(guidance_scale)
        scale = float(scale)

        # load imgs
        source_image = Image.open(source_image).convert("RGB").resize((size, size))
        id = np.array(source_image)
        reference_image = np.array(Image.open(reference_image).convert("RGB").resize((size, size)))
        source_image_bald = np.array(self.get_bald(source_image, scale=0.9))
        H, W, C = source_image_bald.shape

        # generate images
        set_scale(self.pipeline.unet, scale)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(random_seed)
        sample = self.pipeline(
            prompt,
            negative_prompt=n_prompt,
            num_inference_steps=step,
            guidance_scale=guidance_scale,
            width=W,
            height=H,
            controlnet_condition=source_image_bald,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            reference_encoder=self.hair_encoder,
            ref_image=reference_image,
        ).samples
        return id, sample, source_image_bald, reference_image

    def get_bald(self, id_image, scale, return_latents=False):
        H, W = id_image.size
        scale = float(scale)
        output_type = "latent" if return_latents else "pil"
        result = self.remove_hair_pipeline(
            prompt="",
            negative_prompt="",
            num_inference_steps=30,
            guidance_scale=1.5,
            width=W,
            height=H,
            image=id_image,
            controlnet_conditioning_scale=scale,
            generator=None,
            output_type=output_type,
        )

        if not return_latents:
            return result.images[0]

        latents = result.images[0]
        latents_for_decode = latents.unsqueeze(0) if latents.ndim == 3 else latents
        with torch.no_grad():
            decoded = self.remove_hair_pipeline.vae.decode(
                latents_for_decode / self.remove_hair_pipeline.vae.config.scaling_factor, return_dict=False
            )[0]
        if decoded.ndim == 3:
            decoded = decoded.unsqueeze(0)
        image = self.remove_hair_pipeline.image_processor.postprocess(decoded, output_type="pil")[0]
        return image, latents


def run_batch_bald(model, input_dir, output_dir, size=512, scale=0.9, latent_output_dir=None):
    """Run Stage1 bald conversion on a folder of images and save the outputs (and optionally latents)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_output_dir = Path(latent_output_dir) if latent_output_dir else None
    if latent_output_dir:
        latent_output_dir.mkdir(parents=True, exist_ok=True)
    allowed_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed_ext]
    )
    if not image_paths:
        raise ValueError(f"No images with extensions {allowed_ext} found in {input_dir}")

    total = len(image_paths)
    for idx, path in enumerate(image_paths, start=1):
        img = Image.open(path).convert("RGB").resize((size, size))
        if latent_output_dir:
            bald, latents = model.get_bald(img, scale=scale, return_latents=True)
            torch.save(latents.cpu(), latent_output_dir / f"{path.stem}.pt")
        else:
            bald = model.get_bald(img, scale=scale)
        out_path = output_dir / path.name
        bald.save(out_path)
        print(f"[{idx}/{total}] Saved {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stable-Hair inference and bald conversion utility")
    parser.add_argument('--config', default="./configs/hair_transfer.yaml", help="Path to hair transfer config")
    parser.add_argument('--device', default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument(
        '--dtype',
        choices=['fp16', 'fp32'],
        default='fp32',
        help="Computation precision for the pipelines",
    )
    parser.add_argument('--batch_input_dir', help="Directory with source images for Stage1 bald conversion")
    parser.add_argument('--batch_output_dir', help="Directory to save Stage1 bald images")
    parser.add_argument('--batch_latent_dir', help="Directory to save Stage1 bald latents")
    parser.add_argument('--batch_size', type=int, default=512, help="Resize dimension for batch conversion")
    parser.add_argument('--batch_scale', type=float, default=0.9, help="ControlNet conditioning scale for Stage1")
    args = parser.parse_args()

    weight_dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    model = StableHair(config=args.config, device=args.device, weight_dtype=weight_dtype)

    if args.batch_input_dir:
        if not args.batch_output_dir:
            raise ValueError("batch_output_dir must be provided when batch_input_dir is set.")
        run_batch_bald(
            model,
            input_dir=args.batch_input_dir,
            output_dir=args.batch_output_dir,
            size=args.batch_size,
            scale=args.batch_scale,
            latent_output_dir=args.batch_latent_dir,
        )
    else:
        kwargs = OmegaConf.to_container(model.config.inference_kwargs)
        id, image, source_image_bald, reference_image = model.Hair_Transfer(**kwargs)
        os.makedirs(model.config.output_path, exist_ok=True)
        output_file = os.path.join(model.config.output_path, model.config.save_name)
        concatenate_images(
            [id, source_image_bald, reference_image, (image * 255.).astype(np.uint8)],
            output_file=output_file,
            type="np",
        )
