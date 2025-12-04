import torch
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import os
import argparse
from torchvision import transforms

def load_vae(model_id="runwayml/stable-diffusion-v1-5"):
    print(f"Loading VAE from {model_id}...")
    try:
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    except OSError:
        print(f"Could not load from {model_id}, trying local cache or full path if provided.")
        vae = AutoencoderKL.from_pretrained(model_id) # Try without subfolder if it's a direct path to VAE
    
    vae.to("cuda" if torch.cuda.is_available() else "cpu")
    vae.eval()
    return vae

def preprocess_image(image_path, size=512):
    image = Image.open(image_path).convert("RGB")
    # Resize and center crop to size
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(image).unsqueeze(0)

def tensor_to_image(tensor):
    image = tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 0.5 + 0.5).clip(0, 1)
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image[0])

def reconstruct(vae, image_tensor):
    device = vae.device
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        # Encode
        latents = vae.encode(image_tensor).latent_dist.mode() # Use mode for deterministic check
        
        # Decode
        decoding = vae.decode(latents).sample
    return decoding

def main():
    parser = argparse.ArgumentParser(description="VAE Reconstruction Experiment")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="vae_reconstruction_results", help="Output directory")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to process")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5", help="Model ID or path")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    vae = load_vae(args.model_id)
    
    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    import random
    random.seed(42)
    if len(image_files) > args.num_images:
        selected_files = random.sample(image_files, args.num_images)
    else:
        selected_files = image_files
        
    print(f"Processing {len(selected_files)} images...")
    
    for filename in selected_files:
        img_path = os.path.join(args.image_dir, filename)
        
        # Preprocess
        input_tensor = preprocess_image(img_path)
        
        # Reconstruct
        recon_tensor = reconstruct(vae, input_tensor)
        
        # Convert back to PIL
        original_pil = tensor_to_image(input_tensor)
        recon_pil = tensor_to_image(recon_tensor)
        
        # Concatenate side-by-side
        w, h = original_pil.size
        combined = Image.new("RGB", (w * 2, h))
        combined.paste(original_pil, (0, 0))
        combined.paste(recon_pil, (w, 0))
        
        # Save
        save_path = os.path.join(args.output_dir, f"recon_{filename}")
        combined.save(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
