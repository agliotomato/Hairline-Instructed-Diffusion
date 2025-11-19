import argparse
import logging
import math
import os
from pathlib import Path
import itertools
import numpy as np
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from ref_encoder.adapter import adapter_injection
from ref_encoder.reference_control import ReferenceAttentionControl
from ref_encoder.reference_unet import ref_unet
from diffusers.models.controlnet import ControlNetModel
import albumentations as A
import cv2
import torch.nn.functional as F

check_min_version("0.23.0")
logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation
        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train pixel-space ControlNet for Stage2 (hair transfer)")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None)
    parser.add_argument("--train_data_dir", type=str, required=True)
    parser.add_argument("--refer_column", type=str, default="reference")
    parser.add_argument("--source_column", type=str, default="source")
    parser.add_argument("--target_column", type=str, default="target")
    parser.add_argument("--output_dir", type=str, default="models/stage2_pixel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--revision", type=str, default=None)

    args = parser.parse_args(input_args) if input_args is not None else parser.parse_args()
    if args.resolution % 8 != 0:
        raise ValueError("`--resolution` must be divisible by 8.")
    return args


def make_train_dataset(args, accelerator):
    dataset = load_dataset('json', data_files=args.train_data_dir)
    column_names = dataset["train"].column_names
    refer_column = args.refer_column
    source_column = args.source_column
    target_column = args.target_column
    for c in [refer_column, source_column, target_column]:
        if c not in column_names:
            raise ValueError(f"Dataset missing column: {c}")

    norm = transforms.Normalize([0.5], [0.5])
    to_tensor = transforms.ToTensor()
    pixel_transform = A.Compose([
        A.SmallestMaxSize(max_size=args.resolution),
        A.CenterCrop(args.resolution, args.resolution),
    ], additional_targets={'image0': 'image', 'image1': 'image'})

    def refer_imgaug(image):
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [args.resolution, args.resolution])
        image = norm(to_tensor(image/255.))
        return image

    def imgaug(source_image, target_image):
        source_image = cv2.resize(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB), [args.resolution, args.resolution])
        target_image = cv2.resize(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB), [args.resolution, args.resolution])
        results = pixel_transform(image=source_image, image0=target_image)
        source_image, target_image = norm(to_tensor(results["image"]/255.)), norm(to_tensor(results["image0"]/255.))
        return source_image, target_image

    def preprocess_train(examples):
        source_images = [cv2.imread(image) for image in examples[source_column]]
        refer_images = [cv2.imread(image) for image in examples[refer_column]]
        target_images = [cv2.imread(image) for image in examples[target_column]]
        s, t = zip(*[imgaug(a, b) for a, b in zip(source_images, target_images)])
        r = [refer_imgaug(im) for im in refer_images]
        examples["source_pixel_values"] = list(s)
        examples["refer_pixel_values"] = list(r)
        examples["target_pixel_values"] = list(t)
        return examples

    with accelerator.main_process_first():
        train_dataset = dataset["train"].with_transform(preprocess_train)
    return train_dataset


def collate_fn(examples):
    spv = torch.stack([e["source_pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    rpv = torch.stack([e["refer_pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    tpv = torch.stack([e["target_pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float()
    return {"source_pixel_values": spv, "refer_pixel_values": rpv, "target_pixel_values": tpv}


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir),
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False)
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision).to(accelerator.device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision).to(accelerator.device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision).to(accelerator.device)

    if args.controlnet_model_name_or_path:
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path).to(accelerator.device)
    else:
        controlnet = ControlNetModel.from_unet(unet).to(accelerator.device)

    Hair_Encoder = ref_unet.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision).to(accelerator.device)
    Hair_Adapter = adapter_injection(unet, dtype=torch.float32).to(accelerator.device)

    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)
    Hair_Encoder.requires_grad_(True); Hair_Adapter.requires_grad_(True); controlnet.requires_grad_(True)

    params_to_optimize = itertools.chain(controlnet.parameters(), Hair_Encoder.parameters(), Hair_Adapter.parameters())
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)

    train_dataset = make_train_dataset(args, accelerator)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer, num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes, num_training_steps=args.max_train_steps * accelerator.num_processes)

    Hair_Encoder, Hair_Adapter, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(Hair_Encoder, Hair_Adapter, controlnet, optimizer, train_dataloader, lr_scheduler)

    weight_dtype = torch.float32 if accelerator.mixed_precision == "no" else (torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    Hair_Encoder.to(accelerator.device, dtype=torch.float32)
    Hair_Adapter.to(accelerator.device, dtype=torch.float32)
    controlnet.to(accelerator.device, dtype=torch.float32)

    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0

    null_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    encoder_hidden_states = text_encoder(null_ids.to(device=accelerator.device))[0]

    progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_local_main_process)
    # reference control hooks
    reference_control_writer = ReferenceAttentionControl()
    reference_control_reader = ReferenceAttentionControl()
    reference_control_writer.hook(Hair_Encoder)
    reference_control_reader.hook(unet)
    reference_control_reader_train = reference_control_reader
    reference_control_writer_train = reference_control_writer

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # target latents
                latents = vae.encode(batch["target_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # pixel control image (bald) in [-1,1]
                control_image = batch["source_pixel_values"].to(dtype=weight_dtype)
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(latents.shape[0], 1, 1),
                    controlnet_cond=control_image,
                    return_dict=False,
                )

                # write reference features
                Hair_Encoder(
                    vae.encode(batch["refer_pixel_values"].to(dtype=weight_dtype)).latent_dist.mean * vae.config.scaling_factor,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(latents.shape[0], 1, 1),
                )
                reference_control_reader_train.update(reference_control_writer_train)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.repeat(latents.shape[0], 1, 1).to(dtype=weight_dtype),
                    down_block_additional_residuals=[s.to(dtype=weight_dtype) for s in down_block_res_samples],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                reference_control_reader_train.clear()

                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1); global_step += 1
                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path, safe_serialization=False)
                    logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs); accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone(); accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

