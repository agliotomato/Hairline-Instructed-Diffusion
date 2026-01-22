python test_sd3_native_adapter_inference.py \
  --image_path "test_data/bald_images/test6.png" \
  --mask_path "test_data/segmantic_masks/test6.png" \
  --output_path "results/native2/test6_1.png" \
  --prompt "high quality realistic male hairstyle, low skin fade haircut, black hair, clean sides, textured top, dry matte finish, sharp hairline, high detail, 8k" \
  --adapter_path "output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth" \
  --smart_blur \
  --adapter_scale 1.5 \
  --mask_dilation 10

# V3 Generation for test2
python test_sd3_native_adapter_inference.py \
  --image_path "test_data/bald_images/test6.png" \
  --mask_path "test_data/segmantic_masks/test6.png" \
  --output_path "results/native2/test6_2.png" \
  --prompt "realistic korean male two block haircut, dark brown hair, soft volume, clean contour, slightly wavy textured fringe,high detail, 8k" \
  --adapter_path "output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth" \
  --smart_blur \
  --adapter_scale 1.5 \
  --mask_dilation 10

# V3 Generation for test3
python test_sd3_native_adapter_inference.py \
  --image_path "test_data/bald_images/test6.png" \
  --mask_path "test_data/segmantic_masks/test6.png" \
  --output_path "results/native2/test6_3.png" \
  --prompt "realistic male textured crop haircut, short fringe, ash gray highlights on dark base, rough texture, slightly tousled top, modern barbershop style,ultra-detailed, 8k" \
  --adapter_path "output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth" \
  --smart_blur \
  --adapter_scale 1.5 \
  --mask_dilation 10

# V3 Generation for test2
python test_sd3_native_adapter_inference.py \
  --image_path "test_data/bald_images/test6.png" \
  --mask_path "test_data/segmantic_masks/test6.png" \
  --output_path "results/native2/test6_4.png" \
  --prompt "realistic male hairstyle, natural perm with soft waves, medium length fringe pushed slightly forward, subtle brown highlights, airy movement, fluffy texture, high detail, 8k" \
  --adapter_path "output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth" \
  --smart_blur \
  --adapter_scale 1.5 \
  --mask_dilation 10

# V3 Generation for test3
python test_sd3_native_adapter_inference.py \
  --image_path "test_data/bald_images/test6.png" \
  --mask_path "test_data/segmantic_masks/test6.png" \
  --output_path "results/native2/test6_5.png" \
  --prompt "realistic male slicked back hairstyle, blonde hair color, wet glossy texture, strong hold finish, clean sides, sharp edges, modern gentleman style, high fidelity detail, 8k" \
  --adapter_path "output/tiny_adapter_native_checkpoints/tiny_adapter_native_final.pth" \
  --smart_blur \
  --adapter_scale 1.5 \
  --mask_dilation 10
