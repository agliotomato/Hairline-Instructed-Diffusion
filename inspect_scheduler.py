
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-3.5-medium", subfolder="scheduler")
scheduler.set_timesteps(28)

print("Sigmas first 5:", scheduler.sigmas[:5])
print("Sigmas last 5:", scheduler.sigmas[-5:])
print("Timesteps:", scheduler.timesteps)

strength = 0.9
init_timestep_idx = int(len(scheduler.timesteps) * (1.0 - strength))
print(f"Strength {strength}: idx {init_timestep_idx}, sigma {scheduler.sigmas[init_timestep_idx]}")

strength = 0.7
init_timestep_idx = int(len(scheduler.timesteps) * (1.0 - strength))
print(f"Strength {strength}: idx {init_timestep_idx}, sigma {scheduler.sigmas[init_timestep_idx]}")
