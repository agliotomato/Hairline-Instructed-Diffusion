from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HairlineDatasetV2(Dataset):
    """
    Triplet dataset returning (I_orig, I_bald, M, prompt) matched by filename stem.
    """

    def __init__(
        self,
        orig_dir: str,
        bald_dir: str,
        mask_dir: str,
        metadata_path: Optional[str] = None,
        metadata_text_key: str = "prompt",
        resolution: int = 512,
        aug_prob: float = 0.5,
        aug_blur_max: float = 8.0,
        aug_morph_max: int = 2,
    ) -> None:
        super().__init__()
        orig_dir = Path(orig_dir)
        bald_dir = Path(bald_dir)
        mask_dir = Path(mask_dir)
        if not orig_dir.exists():
            raise FileNotFoundError(f"Could not find orig_dir: {orig_dir}")
        if not bald_dir.exists():
            raise FileNotFoundError(f"Could not find bald_dir: {bald_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Could not find mask_dir: {mask_dir}")

        bald_files = sorted([p for p in bald_dir.iterdir() if p.is_file()])
        orig_map = {p.stem: p for p in orig_dir.iterdir() if p.is_file()}
        mask_map = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}

        metadata = self._load_metadata(metadata_path, metadata_text_key)

        self.items: List[Dict[str, Path]] = []
        for bald_path in bald_files:
            mask_path = mask_map.get(bald_path.stem)
            orig_path = orig_map.get(bald_path.stem)
            if mask_path is None or orig_path is None:
                continue
            prompt = metadata.get(bald_path.stem, "")
            self.items.append(
                {"orig": orig_path, "bald": bald_path, "mask": mask_path, "prompt": prompt}
            )

        if not self.items:
            raise RuntimeError(
                "No paired samples found between orig_dir, bald_dir, and mask_dir. "
                "Ensure file names (without extension) match between all folders."
            )

        self.resolution = resolution
        self.aug_prob = aug_prob
        self.aug_blur_max = aug_blur_max
        self.aug_morph_max = aug_morph_max

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        # Note: Mask transform for augmentation is handled manually before ToTensor
        self.mask_resize = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _load_metadata(path: Optional[str], text_key: str) -> Dict[str, str]:
        if not path:
            return {}
        meta_path = Path(path)
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata_path {meta_path} does not exist")

        def _stem_for(entry: Dict[str, str]) -> Optional[str]:
            for key in ("file_name", "filename", "image", "path"):
                if key in entry:
                    return Path(entry[key]).stem
            return None

        prompts: Dict[str, str] = {}
        suffix = meta_path.suffix.lower()
        if suffix == ".jsonl":
            with meta_path.open("r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    stem = _stem_for(entry)
                    if stem is None:
                        continue
                    prompts[stem] = entry.get(text_key, "")
        elif suffix == ".json":
            with meta_path.open("r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    stem = Path(key).stem
                    prompts[stem] = value if isinstance(value, str) else ""
            elif isinstance(data, list):
                for entry in data:
                    if not isinstance(entry, dict):
                        continue
                    stem = _stem_for(entry)
                    if stem is None:
                        continue
                    prompts[stem] = entry.get(text_key, "")
        elif suffix == ".csv":
            with meta_path.open("r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    stem = _stem_for(row)
                    if stem is None:
                        continue
                    prompts[stem] = row.get(text_key, "")
        else:
            raise ValueError(f"Unsupported metadata file extension: {meta_path.suffix}")

        return prompts

    def __len__(self) -> int:
        return len(self.items)
    
    def _apply_augmentation(self, mask_pil: Image.Image) -> Image.Image:
        """Apply Randomized Morphological Augmentation"""
        import numpy as np
        from PIL import ImageFilter
        from scipy.ndimage import binary_erosion, binary_dilation
        
        # 1. Decide if we apply augmentation
        if np.random.rand() > self.aug_prob:
            return mask_pil
            
        mask_np = np.array(mask_pil) > 127 # Binary
        
        # 2. Random Morphology (Erosion/Dilation)
        if self.aug_morph_max > 0:
            # Sample iteration: e.g. -2, -1, 0, 1, 2
            # Negative = Erosion, Positive = Dilation
            morph_iter = np.random.randint(-self.aug_morph_max, self.aug_morph_max + 1)
            
            if morph_iter != 0:
                struct = np.ones((3, 3)) # 3x3 kernel
                if morph_iter < 0:
                    # Erosion
                    mask_np = binary_erosion(mask_np, structure=struct, iterations=abs(morph_iter))
                else:
                    # Dilation
                    mask_np = binary_dilation(mask_np, structure=struct, iterations=morph_iter)
        
        # Convert back to PIL for blurring
        # Mask is now binary boolean
        mask_aug = Image.fromarray((mask_np * 255).astype(np.uint8))
        
        # 3. Random Blur
        if self.aug_blur_max > 0.0:
            sigma = np.random.uniform(0.0, self.aug_blur_max)
            if sigma >= 0.5:
                mask_aug = mask_aug.filter(ImageFilter.GaussianBlur(sigma))
                
                # 4. Peak Normalization (Critical for Adapter)
                # If blur < 255 peak, stretch it back to 255
                # But PIL GaussianBlur usually preserves total energy? No, it diffuses intensity.
                # Peak decreases.
                arr_blurred = np.array(mask_aug).astype(np.float32)
                peak = arr_blurred.max()
                if peak > 0 and peak < 255:
                    arr_blurred = (arr_blurred / peak) * 255.0
                    mask_aug = Image.fromarray(arr_blurred.astype(np.uint8))
        
        return mask_aug

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.items[idx]
        orig_image = Image.open(sample["orig"]).convert("RGB")
        bald_image = Image.open(sample["bald"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        
        orig_pixel_values = self.image_transform(orig_image)
        bald_pixel_values = self.image_transform(bald_image)
        
        # Augmentation Pipeline for Mask
        # 1. Resize first (to ensure morph/blur is resolution consistent)
        mask_resized = self.mask_resize(mask)
        
        # 2. Augment (PIL based)
        mask_augmented = self._apply_augmentation(mask_resized)
        
        # 3. To Tensor
        mask_tensor_01 = self.to_tensor(mask_augmented)
        mask_tensor_01 = torch.clamp(mask_tensor_01, 0.0, 1.0)
        
        # [V4 Dual-Stream] Create Masked Bald Image (Identity Stream Input)
        # Hair region (1.0 in mask) becomes 0.0 (Black) in masked_bald
        # NOTE: pixel_values are normalized to [-1, 1], so we need careful masking.
        # But here valid pixels are [-1, 1], masked pixels should probably be -1 (Black) or 0 (Grey)?
        # Usually standard ControlNet expects [-1, 1]. -1 is black.
        # Let's interact in [0, 1] space first then normalize?
        # Actually bald_pixel_values is already simplified. 
        # Let's do it on the raw tensor before normalization if possible, but here we have normalized values.
        # Check normalization: transforms.Normalize([0.5...], [0.5...]) -> (x - 0.5) / 0.5 = 2x - 1.
        # So Black (0) -> -1.
        
        # Formula: M = 1 where hair is.
        # We want Result = Original where M=0, and Black where M=1.
        # Result = Original * (1 - M) + Black * M
        # Black is -1.
        # So Result = Original * (1 - M) + (-1) * M
        
        
        masked_bald = bald_pixel_values * (1.0 - mask_tensor_01) + (-1.0) * mask_tensor_01
        
        # [Latent Distribution Matching]
        # Normalize Mask to [-1, 1] for Adapter Input
        # Current: [0, 1] -> (x - 0.5) / 0.5 = 2x - 1
        mask_tensor_norm = (mask_tensor_01 - 0.5) / 0.5
        
        return {
            "orig_pixel_values": orig_pixel_values,
            "bald_pixel_values": bald_pixel_values,
            "masked_bald_pixel_values": masked_bald,
            "hair_mask": mask_tensor_norm, # Normalized Input for Adapter
            "prompt": sample["prompt"],
            "orig_path": str(sample["orig"]),
            "bald_path": str(sample["bald"]),
            "mask_path": str(sample["mask"]),
        }
