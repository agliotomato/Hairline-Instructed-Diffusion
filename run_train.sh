#!/bin/bash

# Mixed Training Strategy 
# 70% Sharp (Original Lanczos)
# 30% Smart Blur / Morph (Augmentation)
# Controlled by --aug_prob 0.3

# Ensure accelerate is configured or remove 'accelerate launch' if running directly (but script uses accelerate)
# If config is missing, run: accelerate config default

echo "Starting Mixed Training (70:30)..."

accelerate launch train_sd3_tiny_adapter_native.py \
  --orig_dir "data/original_images" \
  --bald_dir "data/bald_images" \
  --mask_dir "data/segmantic_masks" \
  --resolution 1024 \
  --train_batch_size 2 \
  --num_train_epochs 100 \
  --learning_rate 1e-4 \
  --aug_prob 0.3 \
  --aug_blur_max 8.0 \
  --aug_morph_max 2 \
  --output_dir "output/tiny_adapter_mixed_7030" \
  --mixed_precision "bf16"
