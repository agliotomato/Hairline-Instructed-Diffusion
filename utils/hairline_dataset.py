import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HairlineDataset(Dataset):
    """
    Simple directory-based dataset that pairs bald face renders with their forehead masks and (optional) prompts.

    We assume that `bald_dir` and `mask_dir` contain files that can be matched by filename stem, e.g.
    `bald_dir/0001.png` <-> `mask_dir/0001.png`. Prompts are loaded from an optional metadata file.
    """

    def __init__(
        self,
        bald_dir: str,
        mask_dir: str,
        metadata_path: Optional[str] = None,
        metadata_text_key: str = "prompt",
        resolution: int = 512,
    ) -> None:
        super().__init__()
        bald_dir = Path(bald_dir)
        mask_dir = Path(mask_dir)
        if not bald_dir.exists():
            raise FileNotFoundError(f"Could not find bald_dir: {bald_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Could not find mask_dir: {mask_dir}")

        bald_files = sorted([p for p in bald_dir.iterdir() if p.is_file()])
        mask_map = {p.stem: p for p in mask_dir.iterdir() if p.is_file()}

        metadata = self._load_metadata(metadata_path, metadata_text_key)

        self.items: List[Dict[str, Path]] = []
        for bald_path in bald_files:
            mask_path = mask_map.get(bald_path.stem)
            if mask_path is None:
                continue
            prompt = metadata.get(bald_path.stem, "")
            self.items.append({"image": bald_path, "mask": mask_path, "prompt": prompt})

        if not self.items:
            raise RuntimeError(
                f"No paired samples found between {bald_dir} and {mask_dir}. "
                "Ensure file names (without extension) match between both folders."
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
        image = Image.open(sample["image"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")
        pixel_values = self.image_transform(image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)
        return {
            "pixel_values": pixel_values,
            "hair_mask": mask_tensor,
            "prompt": sample["prompt"],
            "image_path": str(sample["image"]),
            "mask_path": str(sample["mask"]),
        }
