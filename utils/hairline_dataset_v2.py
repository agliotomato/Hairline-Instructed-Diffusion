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
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ]
        )

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.items[idx]
        orig_image = Image.open(sample["orig"]).convert("RGB")
        bald_image = Image.open(sample["bald"]).convert("RGB")
        mask_pil = Image.open(sample["mask"]).convert("L") # 0, 127, 255
        
        # Preprocess logic (Resize first to match resolution)
        # Note: We resize images and masks to target resolution BEOFRE advanced processing
        # to ensure pixel-level logic (dilation/blur) works on the final scale.
        
        orig_image = orig_image.resize((self.resolution, self.resolution), Image.BILINEAR)
        bald_image = bald_image.resize((self.resolution, self.resolution), Image.BILINEAR)
        mask_pil = mask_pil.resize((self.resolution, self.resolution), Image.NEAREST) # Keep labels exact
        
        # Convert to arrays
        orig_np = np.array(orig_image)
        bald_np = np.array(bald_image)
        mask_np = np.array(mask_pil)
        
        # Transforms for Images (To Tensor + Normalize)
        # We implementation manual normalize to control it better or use standard logic
        # Standard: ToTensor -> Normalize([-1, 1])
        def to_tensor_norm(img_np):
            t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            return (t - 0.5) / 0.5
            
        orig_pixel_values = to_tensor_norm(orig_np)
        bald_pixel_values = to_tensor_norm(bald_np)
        
        # --- Smart Mask Logic ---
        
        # Extract Classes
        # 255: Hair, 127: Face, 0: Background
        hair_raw = (mask_np == 255).astype(np.floatㅌ`32)
        face_raw = (mask_np == 127).astype(np.float32)
        
        # Augmentation: Smart Blur vs Sharp (50:50)
        # We always compute sharp mask for cutout, but geom_mask varies.
        
        import random
        import cv2
        
        is_smart_blur = random.random() < 0.5
        
        if is_smart_blur:
            # 1. Blur the hair mask (Soft edges everywhere)
            # Kernel size 15~21 for 512px/1024px is reasonable visible blur
            k_size = random.choice([15, 17, 19, 21])
            hair_blur = cv2.GaussianBlur(hair_raw, (k_size, k_size), 0)
            
            # 2. Protect Face Boundary (Forehead)
            # Dilate face region to create a "Protection Zone"
            # We want the hair near the face to be SHARP (or eroded), not blurry.
            dilate_k = random.choice([15, 21, 25])
            face_zone = cv2.dilate(face_raw, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)
            
            # Smooth the transition of the protection zone
            face_zone_soft = cv2.GaussianBlur(face_zone, (15, 15), 0)
            
            # 3. Combine
            # Near Face (Face Zone=1): Use Sharp Hair (hair_raw)
            # Far from Face (Face Zone=0): Use Blurred Hair (hair_blur)
            # Formula: Sharp * Zone + Blur * (1 - Zone)
            
            final_mask_np = hair_raw * face_zone_soft + hair_blur * (1.0 - face_zone_soft)
            
            # Optional: Erode the sharp part slightly? 
            # User said "Sharp or Very Subtle Blur" for forehead. 
            # Current logic keeps it Sharp (hair_raw).
            # To strictly follow "Erode", we can erode hair_raw before using it in the mix.
            # let's erode slightly (1-3px) to prevent leaking.
            kernel_erode = np.ones((3,3), np.uint8)
            hair_eroded = cv2.erode(hair_raw, kernel_erode, iterations=1)
            
            # Update formula to use Eroded Sharp mask near face
            final_mask_np = hair_eroded * face_zone_soft + hair_blur * (1.0 - face_zone_soft)
            
        else:
            # Sharp Mode
            final_mask_np = hair_raw
            
        # Convert Mask to Tensor [1, H, W]
        geom_mask_tensor = torch.from_numpy(final_mask_np).unsqueeze(0).float()
        geom_mask_tensor = torch.clamp(geom_mask_tensor, 0.0, 1.0)
        
        # --- Masked Bald Image Creation (Identity Input) ---
        # ALWAYS use Sharp Mask for cutout to ensure clean removal
        # Use hair_raw (Sharp) 
        
        hair_sharp_tensor = torch.from_numpy(hair_raw).unsqueeze(0).float()
        
        # Masked Bald = Bald * (1 - Mask) + Black * Mask
        # Black is -1.0 in [-1, 1] space
        masked_bald = bald_pixel_values * (1.0 - hair_sharp_tensor) + (-1.0) * hair_sharp_tensor
        
        return {
            "orig_pixel_values": orig_pixel_values,
            "bald_pixel_values": bald_pixel_values,
            "masked_bald_pixel_values": masked_bald,
            "hair_mask": geom_mask_tensor, # This goes to Geometry Net & Loss (Soft/Sharp mixed)
            "prompt": sample["prompt"],
            "orig_path": str(sample["orig"]),
            "bald_path": str(sample["bald"]),
            "mask_path": str(sample["mask"]),
        }
