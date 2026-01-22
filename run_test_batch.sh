#!/bin/bash

# Output Directory
OUTPUT_DIR="results/native2"
mkdir -p "$OUTPUT_DIR"

# Path to trained adapter
ADAPTER_PATH="output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth"

# Loop for test1 to test6
for i in {1..6}
do
    echo "Processing test${i}..."
    
    python test_sd3_native_adapter_inference.py \
      --image_path "test_data/bald_images/test${i}.png" \
      --mask_path "test_data/segmantic_masks/test${i}.png" \
      --adapter_path "$ADAPTER_PATH" \
      --output_path "${OUTPUT_DIR}/test${i}_smart.png" \
      --prompt "high quality, realistic hairstyle, detailed hair texture"
      
    echo "Finished test${i}"
done

echo "All tests completed. Results saved to $OUTPUT_DIR"
