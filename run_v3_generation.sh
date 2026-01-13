#!/bin/bash

# V3 Generation for test1
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test2
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test2.jpg" \
  --mask_path "data/semantic_masks/test2.jpg" \
  --output_path "results/final_hybrid/test2_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test3
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test3.jpg" \
  --mask_path "data/semantic_masks/test3.jpg" \
  --output_path "results/final_hybrid/test3_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test4
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test4.jpg" \
  --mask_path "data/semantic_masks/test4.jpg" \
  --output_path "results/final_hybrid/test4_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test5
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test5.jpg" \
  --mask_path "data/semantic_masks/test5.jpg" \
  --output_path "results/final_hybrid/test5_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256
