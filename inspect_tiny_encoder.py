from diffusers import ControlNetModel
import torch

# Instantiate ControlNet with 1 conditioning channel (as in V4)
# We use a config similar to SD1.5 (block_out_channels=[320, 640, 1280, 1280])
cnet = ControlNetModel(
    in_channels=4, # Latent input
    conditioning_channels=1, # Our Mask Input
    block_out_channels=(320, 640, 1280, 1280),
    down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
)

print("Tiny Encoder (controlnet_cond_embedding) Structure:")
print(cnet.controlnet_cond_embedding)

# Create dummy input: Batch=1, Channel=1, H=512, W=512
dummy_mask = torch.randn(1, 1, 512, 512)

# Pass through the embedding layer
output = cnet.controlnet_cond_embedding(dummy_mask)

print(f"\nInput Shape: {dummy_mask.shape}")
print(f"Output Shape: {output.shape}")

print(f"\nFinal Channel Count: {output.shape[1]}")
if output.shape[1] == 320:
    print("Verification Successful: Output channels are 320.")
else:
    print("Verification Failed: Output channels mismatch.")
