import torch
from torch import nn


def enable_hairline_conditioning(unet, mask_channels: int = 1):
    """
    Expands the first convolution of the UNet so it can ingest an additional mask channel.

    The original Stable Diffusion UNet expects 4 latent channels. We pad the `conv_in` layer with zeros for the new
    mask channel so the pre-trained weights are preserved on the original latent channels.
    """
    if mask_channels <= 0:
        return unet

    added_channels = getattr(unet.config, "hair_conditioning_channels", 0)
    base_in_channels = getattr(unet.config, "base_in_channels", None)

    if added_channels == mask_channels:
        return unet

    if base_in_channels is None:
        base_in_channels = unet.conv_in.in_channels
        unet.config.base_in_channels = base_in_channels

    if added_channels not in (0, mask_channels):
        raise ValueError(
            f"UNet already configured with {added_channels} conditioning channels. "
            f"Cannot switch to {mask_channels} without reloading base weights."
        )

    current_in = unet.conv_in.in_channels
    if added_channels:
        # Already expanded once and saved. We simply update the config metadata.
        unet.config.in_channels = current_in
        unet.config.hair_conditioning_channels = added_channels
        return unet

    expected_in = base_in_channels + mask_channels

    old_conv: nn.Conv2d = unet.conv_in
    new_conv = nn.Conv2d(
        expected_in,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        bias=old_conv.bias is not None,
    )

    with torch.no_grad():
        new_conv.weight[:, : current_in, ...] = old_conv.weight
        nn.init.zeros_(new_conv.weight[:, current_in:, ...])
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    unet.conv_in = new_conv
    unet.config.in_channels = expected_in
    unet.config.hair_conditioning_channels = mask_channels
    return unet
