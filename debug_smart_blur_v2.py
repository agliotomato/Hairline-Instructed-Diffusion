
import os
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import binary_erosion

def test_smart_blur_v2(mask_path, blur_radius=5.0):
    print(f"Testing Smart Blur V2 on {mask_path}")
    
    # 1. Load Data
    raw_mask = Image.open(mask_path).convert("L")
    raw_np = np.array(raw_mask)
    
    # Definitions
    # Hair: > 200 (White)
    hair_mask = (raw_np > 200)
    # Face: 50~200 (Gray)
    face_mask = (raw_np > 50) & (raw_np < 200)
    
    # 2. Prepare Base Blurs
    mask_pil = Image.fromarray((hair_mask * 255).astype(np.uint8))
    
    # Heavy Blur (Volume)
    heavy_radius = blur_radius * 4.0
    mask_heavy = mask_pil.filter(ImageFilter.GaussianBlur(heavy_radius))
    
    # Light Blur (Detail)
    mask_light = mask_pil.filter(ImageFilter.GaussianBlur(blur_radius))
    
    # 3. Create "Core" Mask (Erosion-based)
    # Idea: Only apply heavy blur to the "Deep Core" of the hair.
    # Thin parts (sideburns) or Edges will be excluded from Core.
    
    # Erosion Iterations: Determines how "deep" the core is.
    # 1 iteration ~ 1 pixel? No, binary_erosion uses structure.
    # We want to erode by roughly `heavy_radius` pixels.
    # heavy_radius is 20.0 (5.0 * 4).
    # So we erode by ~20 pixels.
    
    erosion_size = int(heavy_radius) 
    # Structure for erosion (Disk shape for isotropy)
    y, x = np.ogrid[-erosion_size:erosion_size+1, -erosion_size:erosion_size+1]
    struct = x*x + y*y <= erosion_size*erosion_size
    
    core_mask_np = binary_erosion(hair_mask, structure=struct)
    core_pil = Image.fromarray((core_mask_np * 255).astype(np.uint8))
    
    # Smooth the Core Mask transition
    core_mask_smooth = core_pil.filter(ImageFilter.GaussianBlur(blur_radius * 2))
    
    # 4. Create Face Protection Zone (Original Logic)
    # Still useful for bangs that might be "thick" (Core) but touch the face?
    # If Core logic is robust (Erosion), boundaries are always "Edge".
    # So Core is always > 20px away from the edge.
    # Hair touching face is an edge. So it WILL be excluded from Core.
    # So Face Protection might be redundant IF Erosion Size >= Protection Dilation.
    # But let's keep it as an override for safety if desired, or rely on Core.
    
    # Let's inspect "Core Mask" vs "Hair Mask"
    # Core Mask should be smaller. Area between Hair and Core is "Edge Zone" (Light Blur).
    # Sideburns (if thinner than 2*20=40px) will satisfy Core=False -> Edge Zone -> Light Blur.
    
    # 5. Composite
    # Logic: Core -> Heavy, Edge -> Light
    final_mask = Image.composite(mask_heavy, mask_light, core_mask_smooth)
    
    # Save Debugs
    mask_heavy.save("debug_v2_heavy.png")
    mask_light.save("debug_v2_light.png")
    core_mask_smooth.save("debug_v2_core_alpha.png")
    final_mask.save("debug_v2_final.png")
    
    print("Saved: debug_v2_heavy.png, debug_v2_light.png, debug_v2_core_alpha.png, debug_v2_final.png")

if __name__ == "__main__":
    if not os.path.exists("test_data/segmantic_masks/test4.png"):
        print("Test file not found")
    else:
        test_smart_blur_v2("test_data/segmantic_masks/test4.png")
