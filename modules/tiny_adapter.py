import torch
import torch.nn as nn

class TinyAdapter(nn.Module):
    """
    A lightweight adapter to project 1-channel binary masks into 
    SD3.5 VAE Latent Space (16 channels).
    
    Architecture:
    - Input: [B, 1, H, W] (Mask, resized to latent resolution, e.g. 128x128)
    - Block 1: Conv2d(1 -> 16) + SiLU
    - Block 2: Conv2d(16 -> 16) + SiLU
    - Block 3: Conv2d(16 -> 16) (Zero Convolution for stable training start)
    """
    def __init__(self, input_channels=1, output_channels=16):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, output_channels, kernel_size=1, padding=0) # 1x1 proj to target
        )
        
        # Zero-init the last layer to ensure it starts by adding nothing (ControlNet practice)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    # Test Dimension
    adapter = TinyAdapter()
    dummy_input = torch.randn(1, 1, 128, 128)
    output = adapter(dummy_input)
    print(f"TinyAdapter Output Shape: {output.shape}") # Should be [1, 16, 128, 128]
