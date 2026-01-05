
import torch
import json
from diffusers import SD3Transformer2DModel, SD3ControlNetModel

print("Loading Transformer...")
transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch.float16)

print("Creating ControlNet from Transformer...")
controlnet = SD3ControlNetModel.from_transformer(transformer)

print("\n=== ControlNet Config ===")
# Convert FrozenDict to dict for printing
config_dict = dict(controlnet.config)
print(json.dumps(config_dict, indent=2))
