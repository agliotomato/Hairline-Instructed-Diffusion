#!/bin/bash
python train_tiny_adapter_sd3_v2.py \
  --data_root "data" \
  --resolution 1024 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_train_steps 500 \
  --learning_rate 1e-4 \
  --hidden_channels 128
