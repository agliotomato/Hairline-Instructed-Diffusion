import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 1-channel hairline masks from aligned portraits.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with aligned RGB portraits.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where PNG/NPY/PT hairline masks will be written.",
    )
    parser.add_argument(
        "--face_parsing_ckpt",
        type=str,
        required=True,
        help="Checkpoint file for the face parsing network (e.g., BiSeNet weights).",
    )
    parser.add_argument(
        "--model_module",
        type=str,
        default="models.face_parsing.model",
        help="Python module that exposes the segmentation model class (default matches HairFusion layout).",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="BiSeNet",
        help="Class name inside --model_module that instantiates the segmentation network.",
    )
    parser.add_argument(
        "--model_root",
        type=str,
        default=None,
        help="Optional root to append to sys.path so --model_module can be imported.",
    )
    parser.add_argument("--num_classes", type=int, default=16, help="Number of semantic classes the parser predicts.")
    parser.add_argument("--image_size", type=int, default=512, help="Resolution to run the parser at.")
    parser.add_argument(
        "--hair_label",
        type=int,
        default=10,
        help="Semantic label id that represents hair in the parser output (CelebAMask-HQ/BiSeNet uses 10).",
    )
    parser.add_argument(
        "--skin_labels",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="Comma separated label ids that approximate the skin/face area. Used for forehead refinement.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=21,
        help="Kernel size for cv2.erode when carving the hairline band (use odd values).",
    )
    parser.add_argument("--erosion_iters", type=int, default=1, help="How many erosion iterations to run.")
    parser.add_argument(
        "--forehead_only",
        action="store_true",
        help="If set, intersect the hairline band with the provided skin_labels to keep only forehead-adjacent pixels.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run segmentation on (cuda/cpu).")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Weights / activation dtype used during segmentation.",
    )
    parser.add_argument(
        "--save_npy",
        action="store_true",
        help="Additionally store each mask as <name>.npy for numpy-based pipelines.",
    )
    parser.add_argument(
        "--save_pt",
        action="store_true",
        help="Additionally store each mask as <name>.pt for torch.load convenience.",
    )
    return parser.parse_args()


def list_images(root: Path) -> List[Path]:
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(path)
    return files


def load_parser_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model_root:
        sys.path.append(args.model_root)
    module = importlib.import_module(args.model_module)
    model_cls = getattr(module, args.model_class)
    model = model_cls(args.num_classes)
    state = torch.load(args.face_parsing_ckpt, map_location="cpu")
    missing = model.load_state_dict(state, strict=False)
    if missing.missing_keys:
        print(f"[WARN] Missing keys when loading parser: {missing.missing_keys}")
    if missing.unexpected_keys:
        print(f"[WARN] Unexpected keys when loading parser: {missing.unexpected_keys}")

    dtype = torch.float16 if args.precision == "fp16" and torch.cuda.is_available() else torch.float32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def preprocessing_transform(resolution: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def run_parser(
    model: torch.nn.Module,
    image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
        parsing = torch.argmax(logits, dim=1)
    return parsing.squeeze(0).cpu().numpy().astype(np.uint8)


def build_hairline_band(
    parsing: np.ndarray,
    hair_label: int,
    kernel_size: int,
    erosion_iters: int,
    image_size: int,
) -> np.ndarray:
    hair_mask = (parsing == hair_label).astype(np.uint8)
    if hair_mask.shape != (image_size, image_size):
        hair_mask = cv2.resize(hair_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    inner = cv2.erode(hair_mask, kernel, iterations=max(1, erosion_iters))
    border = cv2.subtract(hair_mask, inner)
    return border.astype(np.float32)


def refine_with_skin(
    hairline: np.ndarray,
    parsing: np.ndarray,
    skin_labels: Sequence[int],
    image_size: int,
) -> np.ndarray:
    skin_mask = np.isin(parsing, skin_labels).astype(np.uint8)
    if skin_mask.shape != (image_size, image_size):
        skin_mask = cv2.resize(skin_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
    return hairline * skin_mask.astype(np.float32)


def save_mask(base_path: Path, mask: np.ndarray, save_npy: bool, save_pt: bool) -> None:
    png = (mask * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(png, mode="L").save(base_path.with_suffix(".png"))
    if save_npy:
        np.save(base_path.with_suffix(".npy"), mask)
    if save_pt:
        torch.save(torch.from_numpy(mask), base_path.with_suffix(".pt"))


def process_directory(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_images(input_dir)
    if not files:
        raise RuntimeError(f"No images found in {input_dir}")

    model = load_parser_model(args)
    device = next(model.parameters()).device
    transform = preprocessing_transform(args.image_size)
    skin_labels = [int(x.strip()) for x in args.skin_labels.split(",") if x.strip()]

    for path in files:
        image = Image.open(path).convert("RGB")
        parsing = run_parser(model, image, transform, device)
        hairline = build_hairline_band(
            parsing=parsing,
            hair_label=args.hair_label,
            kernel_size=args.kernel_size,
            erosion_iters=args.erosion_iters,
            image_size=args.image_size,
        )
        if args.forehead_only and skin_labels:
            hairline = refine_with_skin(
                hairline=hairline,
                parsing=parsing,
                skin_labels=skin_labels,
                image_size=args.image_size,
            )
        base = output_dir / path.stem
        save_mask(base, hairline, save_npy=args.save_npy, save_pt=args.save_pt)
        print(f"[INFO] Saved hairline mask for {path.name} -> {base.with_suffix('.png')}")


def main():
    args = parse_args()
    process_directory(args)


if __name__ == "__main__":
    main()
