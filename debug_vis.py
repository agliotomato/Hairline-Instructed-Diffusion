import os
from PIL import Image, ImageOps
import numpy as np

def debug_vis():
    # Paths
    img_path = "data/bald_images/test1.png"
    mask_path = "data/semantic_masks/test1.png"
    
    # Load Image
    img = Image.open(img_path)
    print(f"Original Image Mode: {img.mode}")
    
    if img.mode == 'RGBA':
        # Create a white background comparison
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        com = Image.alpha_composite(bg, img).convert('RGB')
        com.save("debug_input_white_bg.png")
        print("Saved debug_input_white_bg.png (Composited on White)")
        
        # Default convert
        conv = img.convert("RGB")
        conv.save("debug_input_convert_rgb.png")
        print("Saved debug_input_convert_rgb.png (Direct Convert)")
    else:
        img.save("debug_input_original.png")
        print("Saved debug_input_original.png")

    # Load Mask
    mask = Image.open(mask_path)
    print(f"Mask Mode: {mask.mode}")
    # Save mask visualization
    mask.convert("L").save("debug_mask_vis.png")
    
    # Analyze Corners
    img_rgb = img.convert("RGB")
    data = np.array(img_rgb)
    print(f"Top-Left Pixel: {data[0,0]}")
    print(f"Bottom-Right Pixel: {data[-1,-1]}")

if __name__ == "__main__":
    debug_vis()
