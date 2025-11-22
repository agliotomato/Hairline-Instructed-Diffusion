from __future__ import annotations

import torch
from torch import nn


class HairlineConditioningEmbeddings(nn.Module):
    """
    Projects hairline mask (and optional bald latent) into cross-attention tokens.

    mask_latent: (B, 1, H, W)
    bald_latent: (B, 4, H, W) or None
    returns: (B, num_tokens, hidden_size) where tokens = [bald?, mask]
    """

    def __init__(self, hidden_size: int, use_bald_token: bool = True) -> None:
        super().__init__()
        self.use_bald_token = use_bald_token
        self.mask_proj = nn.Linear(1, hidden_size)
        self.bald_proj = nn.Linear(4, hidden_size) if use_bald_token else None

    def forward(self, mask_latent: torch.Tensor, bald_latent: torch.Tensor | None) -> torch.Tensor:
        mask_vec = mask_latent.flatten(2).mean(dim=-1, keepdim=True)  # (B, 1, 1)
        mask_token = self.mask_proj(mask_vec.squeeze(-1)).unsqueeze(1)  # (B, 1, D)

        tokens = [mask_token]
        if self.use_bald_token and bald_latent is not None:
            bald_vec = bald_latent.flatten(2).mean(dim=-1)  # (B, 4)
            bald_token = self.bald_proj(bald_vec).unsqueeze(1)  # (B, 1, D)
            tokens.insert(0, bald_token)

        return torch.cat(tokens, dim=1)
