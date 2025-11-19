import os
import argparse
import json
import random
from PIL import Image
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusers import UniPCMultistepScheduler
from diffusers.models import UNet2DConditionModel
from ref_encoder.latent_controlnet import ControlNetModel as LatentControlNetModel
from utils.pipeline_cn import StableDiffusionControlNetPipeline


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def aug_variants(img: Image.Image, n: int):
    variants = []
    w, h = img.size
    for i in range(n):
        im = img.copy()
        # simple deterministic-ish augmentations
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
    ap.add_argument('--out_dir', default='./datasets/stage1_pixel')
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

    # build latent stage1 (bald) pipeline
    unet = UNet2DConditionModel.from_pretrained(cfg.pretrained_model_path, subfolder='unet').to(device)
    controlnet = LatentControlNetModel.from_unet(unet).to(device)
    state = torch.load(cfg.bald_converter_path, map_location='cpu')
    controlnet.load_state_dict(state, strict=False)
    controlnet.to(dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.pretrained_model_path, controlnet=controlnet, safety_checker=None, torch_dtype=torch.float32
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    id_img = Image.open(args.id_image).convert('RGB').resize((args.size, args.size))
    imgs = aug_variants(id_img, args.num_samples)

    jsonl = []
    for i, src in enumerate(imgs):
        bald = pipe(
            prompt='', negative_prompt='', num_inference_steps=30, guidance_scale=1.5,
            width=args.size, height=args.size, image=src, controlnet_conditioning_scale=1.0
        ).images[0]

        src_path = os.path.join(args.out_dir, 'images', f'stage1_src_{i:04d}.png')
        tgt_path = os.path.join(args.out_dir, 'images', f'stage1_tgt_{i:04d}.png')
        src.save(src_path)
        bald.save(tgt_path)
        jsonl.append({'source': src_path, 'target': tgt_path})

    with open(os.path.join(args.out_dir, 'train.jsonl'), 'w') as f:
        for row in jsonl:
            f.write(json.dumps(row) + '\n')

    print('Wrote dataset to', args.out_dir)


if __name__ == '__main__':
    main()

