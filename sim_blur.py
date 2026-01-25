
import numpy as np
from PIL import Image, ImageFilter

def simulate_thin_line_blur():
    # Create a 100x100 canvas
    canvas = np.zeros((100, 100), dtype=np.uint8)
    
    # Draw a thin line (width 2px) simulating a sideburn
    canvas[20:80, 48:50] = 255
    
    img = Image.fromarray(canvas)
    
    # Apply Blur 5.0 (Current setting)
    blurred_5 = img.filter(ImageFilter.GaussianBlur(5.0))
    max_val_5 = np.max(np.array(blurred_5))
    
    # Apply Blur 2.0 (Proposed)
    blurred_2 = img.filter(ImageFilter.GaussianBlur(2.0))
    max_val_2 = np.max(np.array(blurred_2))
    
    print(f"Original Max: 255")
    print(f"Blur 5.0 Max Intensity: {max_val_5}")
    print(f"Blur 2.0 Max Intensity: {max_val_2}")

if __name__ == "__main__":
    simulate_thin_line_blur()
