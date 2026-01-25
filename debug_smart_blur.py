
import os
import numpy as np
from PIL import Image, ImageFilter
import argparse

def test_smart_blur(mask_path, blur_radius=5.0):
    print(f"Testing smart_blur on {mask_path}")
    
    if not os.path.exists(mask_path):
        print(f"Error: {mask_path} not found")
        return

    # Load Raw Mask
    raw_mask = Image.open(mask_path).convert("L")
    raw_np = np.array(raw_mask)
    
    unique_values = np.unique(raw_np)
    print(f"Unique pixel values in mask: {unique_values}")
    
    # Analyze assumption
    # Hair > 200?
    hair_mask = (raw_np > 200).astype(np.uint8) 
    print(f"Hair pixels count: {np.sum(hair_mask)}")
    
    # Face: 50 < x < 200?
    face_mask = ((raw_np > 50) & (raw_np < 200)).astype(np.uint8) 
    print(f"Face pixels count: {np.sum(face_mask)}")

    if np.sum(face_mask) == 0:
        print("WARNING: No Face area detected! smart_blur relies on Face area for protection.")
        # Try to infer if it's a binary mask (0, 255 only)
        if len(unique_values) <= 2:
             print("It seems to be a binary mask. Smart Blur protection zone cannot be created from Face label.")
    
    # Run Smart Blur Logic
    mask = Image.fromarray((hair_mask * 255).astype(np.uint8))
    
    # 1. Heavy Blur
    heavy_radius = blur_radius * 4.0
    mask_heavy = mask.filter(ImageFilter.GaussianBlur(heavy_radius))
    mask_heavy.save("debug_smart_blur_heavy.png")
    
    # 2. Light Blur
    mask_light = mask.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # 3. Protection Zone
    face_pil = Image.fromarray((face_mask * 255).astype(np.uint8))
    # Dilation logic
    protection_zone = face_pil.filter(ImageFilter.MaxFilter(41)) 
    protection_zone = protection_zone.filter(ImageFilter.GaussianBlur(15))
    protection_zone.save("debug_smart_blur_protection.png")
    
    # 4. Composite
    final_mask = Image.composite(mask_light, mask_heavy, protection_zone)
    final_mask.save("debug_smart_blur_final.png")
    print("Saved debug images: debug_smart_blur_heavy.png, debug_smart_blur_protection.png, debug_smart_blur_final.png")

if __name__ == "__main__":
    # Test on a sample mask
    test_smart_blur("test_data/segmantic_masks/test4.png")
