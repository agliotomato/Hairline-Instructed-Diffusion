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
    
    
    
    def _compute_sdf(self, mask_pil: Image.Image, tau: float = 5.0) -> torch.Tensor:
        """
        Compute Normalized Signed Distance Function.
        Values mapped to [-1, 1] using tanh(SDF / tau).
        """
        import cv2
        import numpy as np
        
        mask_np = np.array(mask_pil)
        # Binary: 255=Inside, 0=Outside
        binary = (mask_np > 127).astype(np.uint8)
        
        # Distance to Nearest Zero (Outside) for Inside pixels
        # DIST_L2: Euclidean Distance
        # 5: Mask Size (3 or 5 usually)
        dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Distance to Nearest Zero (Inside) for Outside pixels
        # Invert binary
        dist_out = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
        
        # Signed Distance: Inside is positive, Outside is negative
        # Boundary is 0.
        sdf = dist_in - dist_out
        
        # Normalization with Tanh
        # tau controls the "slope" width.
        # sdf / tau
        sdf_norm = np.tanh(sdf / tau)
        
        # Convert to Tensor [1, H, W]
        timestamp_tensor = torch.from_numpy(sdf_norm).float().unsqueeze(0)
        return timestamp_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.items[idx]
        orig_image = Image.open(sample["orig"]).convert("RGB")
        bald_image = Image.open(sample["bald"]).convert("RGB")
        
        # Load Mask - Try to preserve values for Semantic Parsing
        raw_mask = Image.open(sample["mask"])
        if raw_mask.mode != "L":
            raw_mask = raw_mask.convert("L")
            
        # Extract Hair (Assuming > 200 is Hair)
        mask_np = np.array(raw_mask)
        hair_mask_np = (mask_np > 200).astype(np.uint8) * 255
        hair_mask_pil = Image.fromarray(hair_mask_np)
        
        orig_pixel_values = self.image_transform(orig_image)
        bald_pixel_values = self.image_transform(bald_image)
        
        # Resize inputs
        hair_mask_resized = self.mask_resize(hair_mask_pil)
        
        # [SDF Transformation]
        # We use SDF instead of binary mask for Adapter Input
        # Random Tau for augmentation (Softness variation)
        # Range: 2.0 (Sharp) ~ 10.0 (Soft)
        tau = np.random.uniform(2.0, 10.0)
        mask_tensor_sdf = self._compute_sdf(hair_mask_resized, tau=tau)
        
        # For Compositing (making masked_bald), we still need a hard 0/1 mask or soft [0,1].
        # We can derive it from SDF or just calculate binary from resized.
        # Tanh(SDF) is [-1, 1].
        # 0 to 1 mapping: (Tanh + 1) / 2
        mask_01 = (mask_tensor_sdf + 1.0) / 2.0
        
        # Create Masked Bald:
        # Result = Original * (1 - M) + Black * M
        # Black in [-1, 1] is -1.
        # But wait, masked_bald should be the input to the identity stream.
        # It represents "Bald Image with Missing Area".
        # It should be Black (-1) where hair will be.
        masked_bald = bald_pixel_values * (1.0 - mask_01) + (-1.0) * mask_01
        
        return {
            "orig_pixel_values": orig_pixel_values,
            "bald_pixel_values": bald_pixel_values,
            "masked_bald_pixel_values": masked_bald,
            "hair_mask": mask_tensor_sdf, # [-1, 1] Normalized SDF
            "prompt": sample["prompt"],
            "orig_path": str(sample["orig"]),
            "bald_path": str(sample["bald"]),
            "mask_path": str(sample["mask"]),
        }
