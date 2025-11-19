import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from diffusers import AutoencoderKL
from torchvision import transforms


def load_image(path: str, resolution: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((resolution, resolution), Image.BILINEAR)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    return transform(image).unsqueeze(0)


def encode_latent(vae: AutoencoderKL, image_tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.sample()
    return latent * vae.config.scaling_factor


def save_grid(latent: torch.Tensor, out_path: Path, title: str):
    latent = latent.squeeze(0)
    channels = latent.shape[0]
    fig, axes = plt.subplots(1, channels, figsize=(4 * channels, 4))
    for idx in range(channels):
        ax = axes[idx] if channels > 1 else axes
        channel = latent[idx].cpu().numpy()
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        ax.imshow(channel, cmap="viridis")
        ax.axis("off")
        ax.set_title(f"{title} c{idx}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize VAE latents for two images.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--image_a", type=str, required=True, help="Original image path.")
    parser.add_argument("--image_b", type=str, required=True, help="Bald image path.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out_dir", type=str, default="latent_viz")
    parser.add_argument("--save_npz", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.eval()

    image_a = load_image(args.image_a, args.resolution)
    image_b = load_image(args.image_b, args.resolution)

    latent_a = encode_latent(vae, image_a)
    latent_b = encode_latent(vae, image_b)

    save_grid(latent_a, out_dir / "latent_original.png", "orig")
    save_grid(latent_b, out_dir / "latent_bald.png", "bald")

    if args.save_npz:
        torch.save({"original": latent_a.cpu(), "bald": latent_b.cpu()}, out_dir / "latents.pt")

    print(f"Saved latent visualizations to {out_dir}")


if __name__ == "__main__":
    main()
