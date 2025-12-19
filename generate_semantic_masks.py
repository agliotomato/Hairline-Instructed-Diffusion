import argparse
import sys
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import os
import numpy as np

# Assuming 'models' and 'utils' are in the current directory or python path
# If models/face_parsing/model.py exists in Stable-Hair/models, we might need to add it to sys.path
# Based on whatidid.md, it expects 'models.face_parsing.model'

# Using embedded BiSeNet logic
try:
    from utils.bisenet import BiSeNet
except ImportError:
    print("Error: Could not import BiSeNet from utils.bisenet")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Semantic Masks (Hair Specific) using BiSeNet")
    parser.add_argument("--images_dir", type=str, required=True, help="Directory or File Path containing input images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save hair masks")
    parser.add_argument("--checkpoint", type=str, default="models/face_segment16.pth", help="Path to BiSeNet checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_path = Path(args.images_dir) # Can be dir or file
    mask_dir = Path(args.output_dir)
    
    # Handle single file vs directory
    if input_path.is_file():
        image_paths = [input_path]
        # If output_dir looks like a file (ends in png/jpg), use its parent as dir
        if mask_dir.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
            mask_dir.parent.mkdir(parents=True, exist_ok=True)
        else:
            mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        image_paths = sorted([p for p in input_path.iterdir() 
                          if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
        mask_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Found {len(image_paths)} images to process.')
    if not image_paths:
        print('No images found to segment.')
        return

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize BiSeNet
    try:
        seg_model = BiSeNet(n_classes=16).to(device)
        if not os.path.exists(args.checkpoint):
            # Fallback to search in Stable-Hair directory
            alt_ckpt = os.path.join("Stable-Hair", args.checkpoint)
            if os.path.exists(alt_ckpt):
                args.checkpoint = alt_ckpt
            else:
                raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
                
        seg_model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
        seg_model.eval()
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    print("Starting generation...")
    with torch.no_grad():
        for idx, image_path in enumerate(image_paths, 1):
            image_raw = Image.open(image_path).convert('RGB')
            # Resize logic matches build_hairline_mask? Or simple resize.
            # Script in markdown used transform resize.
            
            image_seg_input = transform(image_raw).unsqueeze(0).to(device)
            
            # get_seg and get_seg_mask are likely custom utils from 'utils' package
            # If they are missing, we implement basic argmax logic
            try:
                from utils import get_seg
                image_seg_output, _ = get_seg(seg_model, image_seg_input, image_seg_input.shape[2:], sigmoid=True)
                # BiSeNet Output is usually [B, 16, H, W] logits
                # If get_seg is missing, we use direct model output
            except ImportError:
                 logits = seg_model(image_seg_input)[0]
                 image_seg_output = logits

            # Extract Hair Class (10)
            # Typically BiSeNet for CelebAMask-HQ: 10 is Hair
            # Logic: valid_mask = argmax(logits) == 10
            
            pred = torch.argmax(image_seg_output, dim=1).squeeze(0).cpu().numpy() # [H, W]
            
            # Create 3-class mask
            # 0: Background
            # 127: Face (Labels 1-9, 11-13 etc)
            # 255: Hair (Label 10)
            
            mask = np.zeros_like(pred, dtype=np.uint8)
            
            # Define Face Labels (CelebAMask-HQ 19 classes usually, BiSeNet 16 classes)
            # 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye, 6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 
            # 10: nose, 11: mouth, 12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
            # Wait, user's BiSeNet is 16 classes usually.
            # Standard BiSeNet 19 labels often used:
            # 0:'bg', 1:'skin', 2:'l_brow', 3:'r_brow', 4:'l_eye', 5:'r_eye',
            # 6:'eye_g', 7:'l_ear', 8:'r_ear', 9:'ear_r', 10:'nose', 11:'mouth',
            # 12:'u_lip', 13:'l_lip', 14:'neck', 15:'neck_l', 16:'cloth', 17:'hair', 18:'hat'
            
            # But whatidid.md said "Hair is 10" in the original code snippet.
            # > "16개의 클래스 중 머리(hair) 클래스에 해당하는 픽셀만 1... Hair(10)"
            # Let's trust the previous code snippet in whatidid.md which said "mask_hair = (pred == 10)"
            
            # So assuming 10 is Hair.
            # Face parts are usually 1~9, 11~15 (excluding background 0 and hair 10)
            
            face_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15] 
            
            for label in face_labels:
                mask[pred == label] = 127
            
            mask[pred == 10] = 255 # Hair
            
            # Save using PIL 'L' mode to preserve exact 0, 127, 255 values
            Image.fromarray(mask, mode='L').save(mask_dir / image_path.name)
            
            if idx % 10 == 0 or idx == len(image_paths):
                print(f'Processed {idx}/{len(image_paths)}: {image_path.name}')

    print('Hair mask generation completed.')

if __name__ == "__main__":
    main()
