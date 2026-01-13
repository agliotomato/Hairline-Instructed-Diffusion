import torch
import torch.nn as nn

class TinyAdapterV2(nn.Module):
    """
    TinyAdapter V2: "Medium" Capacity
    Implements an 'Expand-Squeeze' architecture for better feature extraction.
    
    Architecture:
    - Input: [B, 1, H, W]
    - Block 1 (Expansion): Conv2d(1 -> 128) + SiLU
        -> Expands the low-dim mask into a high-dim feature space to capture complex geometry.
    - Block 2 (Processing): Conv2d(128 -> 128) + SiLU
        -> Processes features in this rich space.
    - Block 3 (Compression): Conv2d(128 -> 16) (Zero Convolution)
        -> Compresses vital information back to SD3.5 Latent spec (16ch).
    """
    def __init__(self, input_channels=1, hidden_channels=128, output_channels=16):
        super().__init__()
        
        self.net = nn.Sequential(
            # Expansion Phase
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            
            # Processing Phase (Deepening)
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            
            # Compression Phase (Projection to Latent Space)
            nn.Conv2d(hidden_channels, output_channels, kernel_size=1, padding=0) 
        )
        
        # Zero-init the last layer (ControlNet Standard)
        # Ensures the model starts by doing "nothing", allowing safe fine-tuning.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        # x: [B, 1, H, W]
        # Logs for first run verification
        # print(f"[TinyAdapterV2] Input: {x.shape}")
        h = self.net(x)
        # print(f"[TinyAdapterV2] Output: {h.shape}")
        return h

if __name__ == "__main__":
    # Test
    model = TinyAdapterV2(hidden_channels=128)
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    # Total Params
    params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {params:,}")
