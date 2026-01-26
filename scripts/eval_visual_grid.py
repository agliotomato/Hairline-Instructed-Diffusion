
import argparse
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def crop_center(img, crop_size=512):
    h, w = img.shape[:2]
    y = (h - crop_size) // 2
    x = (w - crop_size) // 2
    return img[y:y+crop_size, x:x+crop_size]

def find_hairline_region(mask_np):
    """
    Find the top boundary of the mask (Hairline)
    """
    # Sum across width to find top-most hair pixel
    # mask_np is 0-255
    # Find rows with hair
    rows = np.any(mask_np > 127, axis=1)
    if not np.any(rows):
        return None
    
    min_y = np.argmax(rows)
    # Center x?
    # Get indices of hair in that min_y row
    xs = np.where(mask_np[min_y + 10] > 127)[0] # Look slightly below top
    if len(xs) == 0:
        center_x = mask_np.shape[1] // 2
    else:
        center_x = int(np.mean(xs))
        
    return center_x, min_y

def find_tips_region(mask_np):
    """
    Find bottom/outer regions
    """
    rows = np.any(mask_np > 127, axis=1)
    if not np.any(rows):
        return None
        
    # Bottom most
    max_y = len(rows) - 1 - np.argmax(rows[::-1])
    
    # Get x
    xs = np.where(mask_np[max_y - 10] > 127)[0]
    if len(xs) == 0:
        center_x = mask_np.shape[1] // 2
    else:
        center_x = int(np.mean(xs))
        
    return center_x, max_y

def create_grid(orig_path, mask_path, result_path, output_path):
    # Load
    orig = cv2.imread(orig_path)
    if orig is None: print(f"Err: {orig_path}"); return
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    res = cv2.imread(result_path)
    
    # Resize to same
    H, W = 1024, 1024
    orig = cv2.resize(orig, (W, H))
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    res = cv2.resize(res, (W, H))
    
    # Mask to RGB
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Find Regions
    hairline_pt = find_hairline_region(mask)
    tips_pt = find_tips_region(mask)
    
    # Crops (256x256)
    crop_sz = 256
    
    def get_crop(img, pt):
        if pt is None: return np.zeros((crop_sz, crop_sz, 3), dtype=np.uint8)
        cx, cy = pt
        x1 = max(0, cx - crop_sz // 2)
        y1 = max(0, cy - crop_sz // 2)
        x2 = min(W, x1 + crop_sz)
        y2 = min(H, y1 + crop_sz)
        
        # Adjust if out of bounds
        if x2 - x1 < crop_sz: x1 = x2 - crop_sz
        if y2 - y1 < crop_sz: y1 = y2 - crop_sz
        
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop, (crop_sz, crop_sz)) # Ensure size

    hl_orig = get_crop(orig, hairline_pt)
    hl_res = get_crop(res, hairline_pt)
    
    tp_orig = get_crop(orig, tips_pt)
    tp_res = get_crop(res, tips_pt)
    
    # Layout
    # [ Orig | Mask | Result ]
    # [ HL_O | HL_R | TP_O | TP_R ] -- Wait, layout mismatch
    
    # Row 1: Full Views (Resize to 512 for compactness)
    view_sz = 512
    r1_orig = cv2.resize(orig, (view_sz, view_sz))
    r1_mask = cv2.resize(mask_rgb, (view_sz, view_sz))
    r1_res = cv2.resize(res, (view_sz, view_sz))
    
    row1 = np.hstack([r1_orig, r1_mask, r1_res])
    
    # Row 2: Zoom-ins
    # We have 4 crops. 4 * 256 = 1024. 
    # Row 1 width = 512 * 3 = 1536.
    # We need to stretch cues.
    
    # Let's clean up:
    # Col 1: Original (Full)
    # Col 2: Mask (Full)
    # Col 3: Result (Full)
    # Col 4: Zoom (Hairline Result)
    # Col 5: Zoom (Tips Result)
    
    # Better:
    # [ Orig ] [ Mask ] [ Result ] [ Zoom HL ] [ Zoom Tips ]
    
    idx_sz = 350
    final_h = idx_sz
    
    items = []
    for img, text in [
        (orig, "Original"),
        (mask_rgb, "Mask"),
        (res, "Result"),
        (hl_res, "Zoom: Hairline"),
        (tp_res, "Zoom: Tips")
    ]:
        resized = cv2.resize(img, (idx_sz, idx_sz))
        # Add text
        cv2.putText(resized, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        items.append(resized)
        
    grid = np.hstack(items)
    
    cv2.imwrite(output_path, grid)
    print(f"Grid saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--res", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    create_grid(args.orig, args.mask, args.res, args.out)
