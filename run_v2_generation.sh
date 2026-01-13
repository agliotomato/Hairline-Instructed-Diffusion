#!/bin/bash

# V2 Generation for 01047
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/01047.png" \
  --mask_path "data/semantic_masks/01047.png" \
  --output_path "results/final_hybrid/01047_result_v2.png" \
  --prompt "a photo of a person with short brown hair"

# V2 Generation for 01056
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/01056.png" \
  --mask_path "data/semantic_masks/01056.png" \
  --output_path "results/final_hybrid/01056_result_v2.png" \
  --prompt "a photo of a person with short brown hair"

# V2 Generation for 01057
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/01057.png" \
  --mask_path "data/semantic_masks/01057.png" \
  --output_path "results/final_hybrid/01057_result_v2.png" \
  --prompt "a photo of a person with short brown hair"
