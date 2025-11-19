import os
import argparse
import json
import random
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusers import DDIMScheduler, UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel

from utils.pipeline import StableHairPipeline
from utils.pipeline_cn import StableDiffusionControlNetPipeline
from ref_encoder.latent_controlnet import ControlNetModel as LatentControlNetModel
from ref_encoder.reference_unet import ref_unet
from ref_encoder.adapter import adapter_injection, set_scale


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def aug_variants(img: Image.Image, n: int):
    variants = []
    w, h = img.size
    for i in range(n):
        im = img.copy()
        if i % 2 == 1:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        angle = random.uniform(-8, 8)
        im = im.rotate(angle, resample=Image.BICUBIC, expand=False)
        scale = random.uniform(0.9, 1.05)
        nw, nh = int(w * scale), int(h * scale)
        im = im.resize((nw, nh), Image.BICUBIC).resize((w, h), Image.BICUBIC)
        variants.append(im)
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='./configs/hair_transfer.yaml')
    ap.add_argument('--id_image', default='./test_imgs/ID/0.jpg')
    ap.add_argument('--ref_image', default='./test_imgs/Ref/0.jpg')
    ap.add_argument('--out_dir', default='./datasets/stage2_pixel')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--num_samples', type=int, default=16)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, 'images'))

    cfg = OmegaConf.load(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build latent stage1 (bald) converter
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_path, subfolder='unet').to(device)
    bald_cn = LatentControlNetModel.from_unet(unet).to(device)
    state = torch.load(cfg.bald_converter_path, map_location='cpu')
    bald_cn.load_state_dict(state, strict=False)
    bald_cn.to(dtype=torch.float32)
    remove_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.pretrained_model_path, controlnet=bald_cn, safety_checker=None, torch_dtype=torch.float32
    ).to(device)
    remove_pipe.scheduler = UniPCMultistepScheduler.from_config(remove_pipe.scheduler.config)

    # Build latent stage2 (hair transfer)
    hair_cn = LatentControlNetModel.from_unet(unet).to(device)
    state = torch.load(os.path.join(cfg.pretrained_folder, cfg.controlnet_path), map_location='cpu')
    hair_cn.load_state_dict(state, strict=False)
    hair_cn.to(dtype=torch.float32)
    transfer_pipe = StableHairPipeline.from_pretrained(
        cfg.pretrained_model_path, controlnet=hair_cn, safety_checker=None, torch_dtype=torch.float32
    ).to(device)
    transfer_pipe.scheduler = DDIMScheduler.from_config(transfer_pipe.scheduler.config)

    hair_encoder = ref_unet.from_pretrained(cfg.pretrained_model_path, subfolder='unet').to(device)
    state = torch.load(os.path.join(cfg.pretrained_folder, cfg.encoder_path), map_location='cpu')
    hair_encoder.load_state_dict(state, strict=False)
    hair_adapter = adapter_injection(transfer_pipe.unet, device=device, dtype=torch.float16, use_resampler=False)
    state = torch.load(os.path.join(cfg.pretrained_folder, cfg.adapter_path), map_location='cpu')
    hair_adapter.load_state_dict(state, strict=False)
    hair_encoder.to(torch.float32)
    hair_adapter.to(torch.float32)

    id_img = Image.open(args.id_image).convert('RGB').resize((args.size, args.size))
    ref_img = Image.open(args.ref_image).convert('RGB').resize((args.size, args.size))
    id_imgs = aug_variants(id_img, args.num_samples)
    ref_imgs = aug_variants(ref_img, args.num_samples)

    jsonl = []
    for i, (src, ref) in enumerate(zip(id_imgs, ref_imgs)):
        # stage1: bald
        bald = remove_pipe(
            prompt='', negative_prompt='', num_inference_steps=30, guidance_scale=1.5,
            width=args.size, height=args.size, image=src, controlnet_conditioning_scale=1.0
        ).images[0]

        # stage2: transfer
        out = transfer_pipe(
            prompt='', negative_prompt='', num_inference_steps=30, guidance_scale=1.5,
            width=args.size, height=args.size,
            controlnet_condition=np.array(bald), controlnet_conditioning_scale=1.0,
            generator=None, reference_encoder=hair_encoder, ref_image=np.array(ref)
        ).samples
        out_img = Image.fromarray((out * 255.).astype(np.uint8))

        src_path = os.path.join(args.out_dir, 'images', f'stage2_src_{i:04d}.png')
        ref_path = os.path.join(args.out_dir, 'images', f'stage2_ref_{i:04d}.png')
        tgt_path = os.path.join(args.out_dir, 'images', f'stage2_tgt_{i:04d}.png')
        bald.save(src_path)
        ref.save(ref_path)
        out_img.save(tgt_path)
        jsonl.append({'source': src_path, 'reference': ref_path, 'target': tgt_path})

    with open(os.path.join(args.out_dir, 'train.jsonl'), 'w') as f:
        for row in jsonl:
            f.write(json.dumps(row) + '\n')

    print('Wrote dataset to', args.out_dir)


if __name__ == '__main__':
    main()
