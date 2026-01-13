#!/bin/bash

python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k"


python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test2.jpg" \
  --mask_path "data/semantic_masks/test2.jpg" \
  --output_path "results/final_hybrid/test2.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k"


python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test3.jpg" \
  --mask_path "data/semantic_masks/test3.jpg" \
  --output_path "results/final_hybrid/test3.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k"


python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test4.jpg" \
  --mask_path "data/semantic_masks/test4.jpg" \
  --output_path "results/final_hybrid/test4.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k"


python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test5.jpg" \
  --mask_path "data/semantic_masks/test5.jpg" \
  --output_path "results/final_hybrid/test5.png" \
  --prompt "high quality, realistic hairstyle, detailed texture, 8k"



