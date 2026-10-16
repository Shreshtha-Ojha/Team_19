"""Small U-Net style enhancer for low-light image enhancement.

Kept deliberately small (3 downsampling levels, narrow channel widths) to be
trainable on CPU in a reasonable time. Input/output: 3-channel images,
values expected in [0, 1]; output passed through sigmoid so it stays in range.
"""
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """3-level U-Net. Base channel width `base_ch` controls size.

    base_ch=16 -> ~0.4M params; base_ch=32 -> ~1.6M params.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 16):
        super().__init__()
        self.enc1 = conv_block(in_ch, base_ch)
        self.enc2 = conv_block(base_ch, base_ch * 2)
        self.enc3 = conv_block(base_ch * 2, base_ch * 4)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = conv_block(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = conv_block(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = conv_block(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out_conv(d1))


if __name__ == "__main__":
    model = SmallUNet(base_ch=16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("output shape:", y.shape, "range:", y.min().item(), y.max().item())
