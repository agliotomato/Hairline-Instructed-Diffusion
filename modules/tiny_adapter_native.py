import torch
import torch.nn as nn

class TinyAdapterNative(nn.Module):
    """
    TinyAdapter Native: SD 3.5 Optimized (1024px -> 128px)
    
    Architecture designed to bridge the gap between 1024x1024 Mask Input 
    and 128x128 SD 3.5 Latent Space without pre-resizing loss.
    
    Downsampling Factor: 8 (Stride 2^3)
    Input: [B, 1, 1024, 1024]
    Output: [B, 16, 128, 128]
    """
    def __init__(self, input_channels=1, base_channels=32, output_channels=16):
        super().__init__()
        
        self.net = nn.Sequential(
            # Stage 1: 1024 -> 512
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            
            # Stage 2: 512 -> 256
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            
            # Stage 3: 256 -> 128 (Target Resolution)
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            
            # Processing Stage: 128 -> 128 (Deepening)
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=32, num_channels=base_channels*4),
            nn.SiLU(),
            
            # Projection Stage: Channel mapping to Latent Space (128ch -> 16ch)
            # Uses Zero-Conv initialization for safe injection
            nn.Conv2d(base_channels*4, output_channels, kernel_size=1, stride=1, padding=0)
        )
        
        # Zero-init the last layer (ControlNet Standard)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x: [B, 1, 1024, 1024]
        return self.net(x)

if __name__ == "__main__":
    # Functional Test
    model = TinyAdapterNative(base_channels=32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Random Input matching SD3.5 Native Mask Resolution
    x = torch.randn(1, 1, 1024, 1024).to(device)
    y = model(x)
    
    print(f"Input Shape: {x.shape}")   # Expected: [1, 1, 1024, 1024]
    print(f"Output Shape: {y.shape}")  # Expected: [1, 16, 128, 128]
    
    assert y.shape == (1, 16, 128, 128), "Output shape mismatch!"
    print("TinyAdapterNative: Functional Test Passed ✅")
    
    # Calculate Params
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")
