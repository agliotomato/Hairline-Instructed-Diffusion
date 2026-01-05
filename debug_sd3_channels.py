
import torch
from diffusers import SD3Transformer2DModel, SD3ControlNetModel

print("Loading Transformer...")
transformer = SD3Transformer2DModel.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="transformer", torch_dtype=torch.float16)

print("Creating ControlNet from Transformer...")
controlnet = SD3ControlNetModel.from_transformer(transformer)

print(f"ControlNet Config: {controlnet.config}")
print(f"ControlNet pos_embed_input weight shape: {controlnet.pos_embed_input.weight.shape}")

# Create dummy inputs
hidden_states = torch.randn(1, 16, 64, 64).to(torch.float16)
condition = torch.randn(1, 16, 64, 64).to(torch.float16)
timestep = torch.tensor([1]).long()

print("Attempting Forward Pass with 16ch condition...")
try:
    controlnet(hidden_states, controlnet_cond=condition, timestep=timestep)
    print("Success with 16ch!")
except Exception as e:
    print(f"Failed with 16ch: {e}")

print("Attempting Forward Pass with 17ch condition...")
condition_17 = torch.randn(1, 17, 64, 64).to(torch.float16)
try:
    controlnet(hidden_states, controlnet_cond=condition_17, timestep=timestep)
    print("Success with 17ch!")
except Exception as e:
    print(f"Failed with 17ch: {e}")
