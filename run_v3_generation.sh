#!/bin/bash

# V3 Generation for test1
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/01047.png" \
  --mask_path "data/semantic_masks/01047.png" \
  --output_path "results/final_hybrid/01047_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test2
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/01056.png" \
  --mask_path "data/semantic_masks/01056.png" \
  --output_path "results/final_hybrid/01056_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test3
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/01057.png" \
  --mask_path "data/semantic_masks/01057.png" \
  --output_path "results/final_hybrid/01057_v3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256