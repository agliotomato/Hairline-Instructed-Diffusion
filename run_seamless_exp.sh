
#!/bin/bash

# Define arrays to test
scales=(1.0 1.5 2.0)
dilations=(0 10 20)
blur_radii=(5.0 9.0)

image="test_data/bald_images/test4.png"
mask="test_data/segmantic_masks/test4.png"
adapter="output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth"
prompt="high quality realisttic hair style, black hair, high detail, 8k"

output_dir="results/seamless_exp"
mkdir -p $output_dir

echo "Starting Grid Search (Scale x Dilation x Blur)..."

for scale in "${scales[@]}"
do
    for dil in "${dilations[@]}"
    do
        for blur in "${blur_radii[@]}"
        do
           echo "Running with Scale: $scale, Dilation: $dil, Blur: $blur"
           python test_sd3_native_adapter_inference.py \
              --image_path "$image" \
              --mask_path "$mask" \
              --output_path "$output_dir/test4_scale_${scale}_dil_${dil}_blur_${blur}.png" \
              --prompt "$prompt" \
              --adapter_path "$adapter" \
              --adapter_scale $scale \
              --smart_blur \
              --soft_blending \
              --blur_radius $blur \
              --mask_dilation $dil
        done
    done
done

echo "Experiment Complete. Check $output_dir"
