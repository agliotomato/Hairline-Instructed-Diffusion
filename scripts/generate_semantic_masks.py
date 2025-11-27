import torch
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
import torchvision.transforms as transforms

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.bisenet import BiSeNet

def preprocess_image(image_path, mean, std):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = image.astype(np.float32)
    image = image / 255.0
    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0).float()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/original_images')
    parser.add_argument('--output_dir', type=str, default='data/semantic_masks')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to 79999_iter.pth')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Initialize BiSeNet
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"Weights not found at {args.weights_path}. Please download them.")
        
    net.load_state_dict(torch.load(args.weights_path, map_location=device))
    net.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    
    image_paths = list(Path(args.input_dir).glob('*.png')) + list(Path(args.input_dir).glob('*.jpg'))
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Class Mapping
    # 0: Background (0, 16, 17, 18 - background, cloth, neck)
    # 1: Face (1-12, 15 - skin, features)
    # 2: Hair (13, 14 - hair, hat)
    
    # Original Labels:
    # 0: 'background', 1: 'skin', 2: 'nose', 3: 'eye_g', 4: 'l_eye', 5: 'r_eye',
    # 6: 'l_brow', 7: 'r_brow', 8: 'l_ear', 9: 'r_ear', 10: 'mouth', 11: 'u_lip',
    # 12: 'l_lip', 13: 'hair', 14: 'hat', 15: 'ear_r', 16: 'neck_l', 17: 'neck', 18: 'cloth'

    print(f"Processing {len(image_paths)} images...")
    
    with torch.no_grad():
        for img_path in tqdm(image_paths):
            tensor = preprocess_image(str(img_path), mean, std).to(device)
            out, _, _ = net(tensor)
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            
            # Map to 3 classes
            new_mask = np.zeros_like(parsing, dtype=np.uint8)
            
            # Face (1)
            face_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]
            for idx in face_indices:
                new_mask[parsing == idx] = 127 # 0.5 in float
                
            # Hair (2)
            hair_indices = [13, 14]
            for idx in hair_indices:
                new_mask[parsing == idx] = 255 # 1.0 in float
                
            # Background is already 0
            
            # Save
            save_path = os.path.join(args.output_dir, img_path.name)
            cv2.imwrite(save_path, new_mask)

    print("Done.")

if __name__ == '__main__':
    main()
