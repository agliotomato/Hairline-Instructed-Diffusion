import os
import argparse
import time
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel

from utils.pipeline_cn import StableDiffusionControlNetPipeline as LatentCNStage1
from utils.pipeline_cn_pixel import StableDiffusionControlNetPixelPipeline as PixelCNStage1
from utils.pipeline import StableHairPipeline as LatentStage2
from utils.pipeline_stage2_pixel import StableHairPipelinePixel as PixelStage2

from ref_encoder.latent_controlnet import ControlNetModel as LatentControlNetModel
from diffusers.models.controlnet import ControlNetModel as PixelControlNetModel
from ref_encoder.reference_unet import ref_unet
from ref_encoder.adapter import adapter_injection, set_scale


def load_image(path, size):
    img = Image.open(path).convert('RGB').resize((size, size))
    return img


def to_numpy(img: Image.Image):
    return np.array(img)


def run_stage1_latent(config, device, weight_dtype, id_image_pil, scale=1.0):
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_path, subfolder="unet").to(device)
    controlnet = LatentControlNetModel.from_unet(unet).to(device)
    state = torch.load(config.bald_converter_path, map_location='cpu')
    controlnet.load_state_dict(state, strict=False)
    controlnet.to(dtype=weight_dtype)
    pipe = LatentCNStage1.from_pretrained(
        config.pretrained_model_path, controlnet=controlnet, safety_checker=None, torch_dtype=weight_dtype
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    H, W = id_image_pil.size[1], id_image_pil.size[0]
    out = pipe(
        prompt="",
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=1.5,
        width=W,
        height=H,
        image=id_image_pil,
        controlnet_conditioning_scale=scale,
    ).images[0]
    return out


def run_stage1_pixel(config, device, weight_dtype, id_image_pil, pixel_model_id_or_dir, scale=1.0):
    controlnet = PixelControlNetModel.from_pretrained(pixel_model_id_or_dir, torch_dtype=weight_dtype)
    pipe = PixelCNStage1.from_pretrained(
        config.pretrained_model_path, controlnet=controlnet, safety_checker=None, torch_dtype=weight_dtype
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    H, W = id_image_pil.size[1], id_image_pil.size[0]
    out = pipe(
        prompt="",
        negative_prompt="",
        num_inference_steps=30,
        guidance_scale=1.5,
        width=W,
        height=H,
        image=id_image_pil,
        controlnet_conditioning_scale=scale,
    ).images[0]
    return out


def build_stage2_latent(config, device, weight_dtype):
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_path, subfolder="unet").to(device)
    controlnet = LatentControlNetModel.from_unet(unet).to(device)
    state = torch.load(os.path.join(config.pretrained_folder, config.controlnet_path), map_location='cpu')
    controlnet.load_state_dict(state, strict=False)
    controlnet.to(weight_dtype)

    pipe = LatentStage2.from_pretrained(
        config.pretrained_model_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    hair_encoder = ref_unet.from_pretrained(config.pretrained_model_path, subfolder="unet").to(device)
    state = torch.load(os.path.join(config.pretrained_folder, config.encoder_path), map_location='cpu')
    hair_encoder.load_state_dict(state, strict=False)

    hair_adapter = adapter_injection(pipe.unet, device=device, dtype=torch.float16, use_resampler=False)
    state = torch.load(os.path.join(config.pretrained_folder, config.adapter_path), map_location='cpu')
    hair_adapter.load_state_dict(state, strict=False)

    hair_encoder.to(weight_dtype)
    hair_adapter.to(weight_dtype)
    return pipe, hair_encoder


def build_stage2_pixel(config, device, weight_dtype, pixel_model_id_or_dir):
    controlnet = PixelControlNetModel.from_pretrained(pixel_model_id_or_dir, torch_dtype=weight_dtype)
    pipe = PixelStage2.from_pretrained(
        config.pretrained_model_path,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=weight_dtype,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    hair_encoder = ref_unet.from_pretrained(config.pretrained_model_path, subfolder="unet").to(device)
    state = torch.load(os.path.join(config.pretrained_folder, config.encoder_path), map_location='cpu')
    hair_encoder.load_state_dict(state, strict=False)

    hair_adapter = adapter_injection(pipe.unet, device=device, dtype=torch.float16, use_resampler=False)
    state = torch.load(os.path.join(config.pretrained_folder, config.adapter_path), map_location='cpu')
    hair_adapter.load_state_dict(state, strict=False)

    hair_encoder.to(weight_dtype)
    hair_adapter.to(weight_dtype)
    return pipe, hair_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/hair_transfer.yaml')
    parser.add_argument('--id_image', required=True)
    parser.add_argument('--ref_image', required=True)
    parser.add_argument('--output_dir', default='./experiments')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--stage1_pixel_model', required=True, help='HF id or local dir for pixel ControlNet (stage1)')
    parser.add_argument('--stage2_pixel_model', required=True, help='HF id or local dir for pixel ControlNet (stage2)')
    parser.add_argument('--control_scale', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    config = OmegaConf.load(args.config)

    id_img = load_image(args.id_image, args.size)
    ref_img = load_image(args.ref_image, args.size)

    # Stage1: Bald conversion
    print('Stage1-Latent...')
    bald_latent = run_stage1_latent(config, device, dtype, id_img, args.control_scale)
    bald_latent.save(os.path.join(args.output_dir, 'stage1_latent.png'))

    print('Stage1-Pixel...')
    bald_pixel = run_stage1_pixel(config, device, dtype, id_img, args.stage1_pixel_model, args.control_scale)
    bald_pixel.save(os.path.join(args.output_dir, 'stage1_pixel.png'))

    # Stage2: Hair transfer
    print('Stage2-Latent...')
    pipe2_latent, hair_enc_latent = build_stage2_latent(config, device, dtype)
    set_scale(pipe2_latent.unet, 1.0)
    out_latent = pipe2_latent(
        prompt="",
        negative_prompt="",
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.size,
        height=args.size,
        controlnet_condition=np.array(bald_latent),
        controlnet_conditioning_scale=args.control_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        reference_encoder=hair_enc_latent,
        ref_image=np.array(ref_img),
    ).samples
    out_latent_img = Image.fromarray((out_latent * 255.).astype(np.uint8))
    out_latent_img.save(os.path.join(args.output_dir, 'stage2_latent.png'))

    print('Stage2-Pixel...')
    pipe2_pixel, hair_enc_pixel = build_stage2_pixel(config, device, dtype, args.stage2_pixel_model)
    set_scale(pipe2_pixel.unet, 1.0)
    out_pixel = pipe2_pixel(
        prompt="",
        negative_prompt="",
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.size,
        height=args.size,
        controlnet_condition=np.array(bald_pixel),
        controlnet_conditioning_scale=args.control_scale,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        reference_encoder=hair_enc_pixel,
        ref_image=np.array(ref_img),
    ).samples
    out_pixel_img = Image.fromarray((out_pixel * 255.).astype(np.uint8))
    out_pixel_img.save(os.path.join(args.output_dir, 'stage2_pixel.png'))

    print('Done. Saved to', args.output_dir)


if __name__ == '__main__':
    main()

