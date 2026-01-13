#!/bin/bash

# V3 Generation for test1
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1_1.png" \
  --prompt "high quality realistic male hairstyle, low skin fade haircut, black hair, clean sides, textured top, dry matte finish, sharp hairline, natural lighting, high detail, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test2
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1_2.png" \
  --prompt "realistic korean male two block haircut, dark brown hair, soft volume, clean contour, slightly wavy textured fringe, natural shine, high detail, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test3
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1_3.png" \
  --prompt "realistic male textured crop haircut, short fringe, ash gray highlights on dark base, rough texture, slightly tousled top, modern barbershop style, cinematic lighting, ultra-detailed, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test2
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1_4.png" \
  --prompt "realistic male hairstyle, natural perm with soft waves, medium length fringe pushed slightly forward, subtle brown highlights, airy movement, fluffy texture, high detail, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256

# V3 Generation for test3
python test_sd3_sonic_hybrid_v2.py \
  --image_path "data/bald_images/test1.png" \
  --mask_path "data/semantic_masks/test1.png" \
  --output_path "results/final_hybrid/test1_5.png" \
  --prompt "realistic male slicked back hairstyle, blonde hair color, wet glossy texture, strong hold finish, clean sides, sharp edges, modern gentleman style, high fidelity detail, 8k" \
  --adapter_path "output/tiny_adapter_v3_checkpoints/tiny_adapter_v2_final.pth" \
  --hidden_channels 256