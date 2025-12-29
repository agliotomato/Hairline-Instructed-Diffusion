from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, ControlNetModel as PixelControlNet
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from transformers import AutoTokenizer, CLIPTextModel
from torchvision import transforms
from tqdm.auto import tqdm

# Import Custom Latent ControlNet
from utils.latent_identity_net import ControlNetModel as LatentControlNet


class MaskedCrossAttnProcessor:
    def __init__(self, mask_pyramid: Dict[int, torch.Tensor], target_indices: List[int], scaling_factor: float = 1.0):
        self.mask_pyramid = mask_pyramid
        self.target_indices = target_indices
        self.scaling_factor = scaling_factor

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Standard Attention Calculation
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # attention_probs shape: (batch_size * heads, query_len, key_len)

        # --- MASK INJECTION logic ---
        if self.target_indices and encoder_hidden_states.shape[1] > max(self.target_indices):
            # Check resolution to pick correct mask
            dim = int(sequence_length ** 0.5)
            
            if dim in self.mask_pyramid:
                 # mask: (1, 1, dim, dim) -> (1, 1, dim*dim) -> tranpose -> (1, dim*dim, 1)
                 mask = self.mask_pyramid[dim].view(1, -1, 1).to(attention_probs.device)
                 
                 for idx in self.target_indices:
                     # Suppress pixels outside mask (where mask is 0)
                     # We multiply the attention score column for the target token by the mask.
                     # Pixels with mask=0 become 0 attention (or low energy).
                     # Actually, multiplying probability by 0 makes it 0.
                     # But we are operating on scores (softmax input)?
                     # Diffusers 'get_attention_scores' usually returns softmaxed probs?
                     # Let's check source code of `get_attention_scores` or usage.
                     # It usually calls `softmax`.
                     
                     # If `attention_probs` is already softmaxed, multiplying by 0 is fine (probability becomes 0).
                     # The distribution won't sum to 1 anymore, but that's okay for suppressing.
                     
                     attention_probs[:, :, idx] *= mask.squeeze(-1)

        # ----------------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def preprocess_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.BILINEAR)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    return transform(image).unsqueeze(0)


def preprocess_conditions(mask_path: str, bald_path: str, resolution: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # 1. Geometry Condition: High-Res Mask (1-channel)
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize((resolution, resolution), Image.NEAREST)
    mask_tensor = transforms.ToTensor()(mask).unsqueeze(0) # [1, 1, H, W]
    mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
    
    # 2. Identity Condition: Masked Bald Image (3-channel)
    bald = Image.open(bald_path).convert("RGB")
    bald = bald.resize((resolution, resolution), Image.BILINEAR)
    bald_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])(bald).unsqueeze(0) # [1, 3, H, W]
    
    masked_bald_tensor = bald_tensor * (1.0 - mask_tensor) + (-1.0) * mask_tensor
    
    return mask_tensor, masked_bald_tensor


def encode_texts(tokenizer, text_encoder, prompt, negative_prompt, device, num_images):
    text_inputs = tokenizer(
        [prompt], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
    ).to(device)
    text_embeddings = text_encoder(text_inputs.input_ids)[0].repeat(num_images, 1, 1)

    uncond_embeddings = None
    if negative_prompt is not None:
        uncond_inputs = tokenizer(
            [negative_prompt], padding="max_length", truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt"
        ).to(device)
        uncond_embeddings = text_encoder(uncond_inputs.input_ids)[0].repeat(num_images, 1, 1)

    return text_embeddings, uncond_embeddings


def get_token_indices(tokenizer, prompt: str, trigger_word: str):
    if not trigger_word:
        return []
    
    input_ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True).input_ids
    trigger_ids = tokenizer(trigger_word, add_special_tokens=False).input_ids
    
    indices = []
    len_trigger = len(trigger_ids)
    if len_trigger == 0:
        return []
        
    for i in range(len(input_ids) - len_trigger + 1):
        if input_ids[i : i + len_trigger] == trigger_ids:
            indices.extend(list(range(i, i + len_trigger)))
            
    print(f"DEBUG: Found trigger '{trigger_word}' at indices {indices} in prompt.")
    return list(set(indices))


def create_mask_pyramid(mask_tensor: torch.Tensor, max_res: int = 64) -> Dict[int, torch.Tensor]:
    # mask_tensor: [1, 1, H, W]
    pyramid = {}
    current_res = max_res
    while current_res >= 8:
        m = F.interpolate(mask_tensor, size=(current_res, current_res), mode='nearest')
        pyramid[current_res] = m
        current_res //= 2
    return pyramid


def main():
    parser = argparse.ArgumentParser(description="Inference with Mask-Guided Cross-Attention Color Control.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--controlnet_path", type=str, required=True)
    parser.add_argument("--bald_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="high quality, 8k, realistic, detailed hair, black hair")
    parser.add_argument("--negative_prompt", type=str, default="blurry, low quality, artificial")
    parser.add_argument("--color_trigger", type=str, default="", help="Words to restrict to mask area (e.g. 'black hair')")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # 1. Load Models
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet").to(device)
    
    # Load ControlNets
    base_path = Path(args.controlnet_path)
    path_a = base_path / "controlnet"
    path_b = base_path / "controlnet_1"
    
    if path_a.exists() and path_b.exists():
        cnet_a = PixelControlNet.from_pretrained(path_a).to(device)
        cnet_b = LatentControlNet.from_pretrained(path_b).to(device)
        controlnet = MultiControlNetModel([cnet_a, cnet_b]).to(device)
    else:
        # Fallback for single or weird structure if needed
        controlnet = MultiControlNetModel.from_pretrained(args.controlnet_path).to(device)
    
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    # 2. Process Conditions
    mask_tensor, masked_bald_tensor = preprocess_conditions(args.mask_path, args.bald_path, 512)
    mask_tensor = mask_tensor.to(device)
    
    with torch.no_grad():
        image_tensor = preprocess_image(args.bald_path, 512).to(device)
        z_bald = vae.encode(image_tensor).latent_dist.sample() * vae.config.scaling_factor
        
        masked_bald_tensor = masked_bald_tensor.to(device)
        masked_bald_latents = vae.encode(masked_bald_tensor).latent_dist.sample() * vae.config.scaling_factor

    # 3. Setup Mask Guided Attention
    if args.color_trigger:
        target_indices = get_token_indices(tokenizer, args.prompt, args.color_trigger)
        if target_indices:
            print(f"Enabling Mask-Guided Attention for tokens: {target_indices}")
            mask_pyramid = create_mask_pyramid(mask_tensor, max_res=64)
            
            attn_procs = {}
            for name in unet.attn_processors.keys():
                if "attn2" in name:
                    attn_procs[name] = MaskedCrossAttnProcessor(mask_pyramid, target_indices)
                else:
                    attn_procs[name] = unet.attn_processors[name]
            
            unet.set_attn_processor(attn_procs)
        else:
            print(f"Warning: Trigger word '{args.color_trigger}' not found in prompt. Skipping Mask Guidance.")

    # 4. Run Inference
    generator = torch.utils.data.RandomSampler(None) # Dummy
    latents = z_bald.clone()
    noise = torch.randn_like(latents)
    
    noise_strength = 0.9
    init_timestep_idx = int(args.num_inference_steps * (1.0 - noise_strength))
    start_timestep = noise_scheduler.timesteps[init_timestep_idx]
    latents = noise_scheduler.add_noise(latents, noise, start_timestep)
    
    text_embeddings, uncond_embeddings = encode_texts(tokenizer, text_encoder, args.prompt, args.negative_prompt, device, 1)
    
    controlnet_cond = [mask_tensor, masked_bald_latents]
    
    timesteps = noise_scheduler.timesteps[init_timestep_idx:]
    print("Running inference...")
    
    for t in tqdm(timesteps):
        with torch.no_grad():
            with torch.autocast(device.type, dtype=weight_dtype):
                latent_input = torch.cat([latents] * 2)
                cond_input = [torch.cat([c]*2) for c in controlnet_cond]
                
                down, mid = controlnet(
                    latent_input, t, 
                    encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings]),
                    controlnet_cond=cond_input,
                    return_dict=False
                )
                
                noise_pred = unet(
                    latent_input, t, 
                    encoder_hidden_states=torch.cat([uncond_embeddings, text_embeddings]),
                    down_block_additional_residuals=down,
                    mid_block_additional_residual=mid
                ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
                
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                
    latents = latents / vae.config.scaling_factor
    with torch.no_grad():
        image = vae.decode(latents).sample
        
    image = (image / 2 + 0.5).clamp(0, 1).cpu()
    img_pil = transforms.ToPILImage()(image[0])
    
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_dir) / "output_color_control.png"
    img_pil.save(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
