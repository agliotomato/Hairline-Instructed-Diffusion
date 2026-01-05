
import torch
from diffusers import SD3ControlNetModel

def main():
    print("Checking SD3ControlNetModel output structure...")
    # Init small model
    model = SD3ControlNetModel(
        sample_size=32,
        patch_size=2,
        in_channels=4, # Small test
        num_layers=2,
        attention_head_dim=4,
        num_attention_heads=4,
        joint_attention_dim=32,
        caption_projection_dim=32,
        pooled_projection_dim=32,
        out_channels=4
    )
    
    # Dummy inputs
    hidden_states = torch.randn(1, 4, 32, 32)
    timestep = torch.tensor([1])
    encoder_hidden_states = torch.randn(1, 10, 32) # seq_len 10
    pooled_projections = torch.randn(1, 32)
    controlnet_cond = torch.randn(1, 4, 32, 32) # Matches in_channels
    
    # Forward
    output = model(
        hidden_states=hidden_states,
        controlnet_cond=controlnet_cond,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False
    )
    
    print(f"Output type: {type(output)}")
    print(f"Output length: {len(output)}")
    print(f"Output[0] type: {type(output[0])}")
    
    if isinstance(output, tuple):
        # usually (controlnet_block_samples,)
        samples = output[0]
        print(f"Internal structure of output[0] (should be list of tensors): {type(samples)}")
        if isinstance(samples, (list, tuple)):
            print(f"Number of blocks: {len(samples)}")
            print(f"Shape of first block: {samples[0].shape}")

if __name__ == "__main__":
    main()
