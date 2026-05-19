from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((skip, x), dim=1)
        return self.conv(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_channels: int, skip_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attention = self.psi(self.relu(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * attention


class AttentionUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.attention = AttentionGate(
            gate_channels=in_channels,
            skip_channels=skip_channels,
            inter_channels=max(out_channels // 2, 1),
        )
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        attended_skip = self.attention(x, skip)
        x = self.up(x)
        if x.shape[-2:] != attended_skip.shape[-2:]:
            x = F.interpolate(x, size=attended_skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat((attended_skip, x), dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels * factor for factor in (1, 2, 4, 8, 16)]
        self.inc = DoubleConv(in_channels, widths[0])
        self.down1 = Down(widths[0], widths[1])
        self.down2 = Down(widths[1], widths[2])
        self.down3 = Down(widths[2], widths[3])
        self.down4 = Down(widths[3], widths[4])
        self.up1 = Up(widths[4], widths[3], widths[3])
        self.up2 = Up(widths[3], widths[2], widths[2])
        self.up3 = Up(widths[2], widths[1], widths[1])
        self.up4 = Up(widths[1], widths[0], widths[0])
        self.outc = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        widths = [base_channels * factor for factor in (1, 2, 4, 8, 16)]
        self.inc = DoubleConv(in_channels, widths[0])
        self.down1 = Down(widths[0], widths[1])
        self.down2 = Down(widths[1], widths[2])
        self.down3 = Down(widths[2], widths[3])
        self.down4 = Down(widths[3], widths[4])
        self.up1 = AttentionUp(widths[4], widths[3], widths[3])
        self.up2 = AttentionUp(widths[3], widths[2], widths[2])
        self.up3 = AttentionUp(widths[2], widths[1], widths[1])
        self.up4 = AttentionUp(widths[1], widths[0], widths[0])
        self.outc = nn.Conv2d(widths[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def build_model(
    architecture: str,
    *,
    in_channels: int = 3,
    out_channels: int = 1,
    base_channels: int = 32,
) -> nn.Module:
    normalized = architecture.lower().replace("-", "_")
    if normalized in {"unet", "baseline", "u_net"}:
        return UNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    if normalized in {"attention_unet", "attention", "attention_u_net"}:
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
    raise ValueError(f"Unknown architecture: {architecture}")
