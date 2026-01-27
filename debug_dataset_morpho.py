
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from utils.hairline_dataset_v2 import HairlineDatasetV2

def test_morpho_aug():
    print("Testing Randomized Morphological Augmentation...")
    
    # Setup dummy paths (assuming user has these dirs standard)
    orig_dir = "test_data/original_images_v2" # Need a valid path
    bald_dir = "test_data/bald_images_v2"
    mask_dir = "test_data/segmantic_masks"
    
    # Check if we need to mock or use real data.
    # Let's try to assume test_data exists based on previous file list.
    # List: test_data exists.
    # Let's check subdirs first?
    pass

def main():
    # Configure paths based on known structure
    # We saw 'test_data' and 'data' in file list.
    # Let's use 'data/myset/images' etc if they exist, or create dummy if needed.
    # Actually, let's use the paths from a known working script or just ask user? 
    # Better: Inspect what directories are actually available deeply.
    # But wait, I can just use 'data/original_images', 'data/bald_images', 'data/only_forehead_line' 
    # as mentioned in 'whatidid.md'.
    
    orig_dir = "data/original_images"
    bald_dir = "data/bald_images"
    mask_dir = "data/segmantic_masks"
    
    if not os.path.exists(orig_dir):
        print(f"Path not found: {orig_dir}. Attempting to use test_data if structured.")
        return

    # Force augmentation
    dataset = HairlineDatasetV2(
        orig_dir=orig_dir,
        bald_dir=bald_dir,
        mask_dir=mask_dir,
        resolution=512,
        aug_prob=1.0, # Force aug
        aug_blur_max=10.0,
        aug_morph_max=3
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Sample 5 times from index 0 to see variations
    output_img = Image.new("L", (512 * 5, 512))
    
    idx = 0
    # Find a valid index
    if len(dataset) == 0:
        print("Empty dataset")
        return

    print("Generating 5 augmented variations of the same sample...")
    for i in range(5):
        sample = dataset[idx]
        mask_tensor = sample["hair_mask"] # [1, H, W]
        
        # Convert SDF [-1, 1] to [0, 1] for visualization
        mask_vis = (mask_tensor + 1.0) / 2.0
        mask_pil = transforms.ToPILImage()(mask_vis)
        output_img.paste(mask_pil, (512 * i, 0))
        
        # Check stats (Should be near -1 or 1, and 0 at boundary)
        print(f"Sample {i}: Min={mask_tensor.min():.4f}, Max={mask_tensor.max():.4f}, Mean={mask_tensor.mean():.4f}")
        
    output_path = "debug_aug_samples.png"
    output_img.save(output_path)
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()
