"""Lightweight restoration architectures used by the OCST-MAIE checkpoints."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def group_norm(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            group_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            group_norm(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class NAFBlockLite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = group_norm(channels)
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.dw = nn.Conv2d(channels * 2, channels * 2, 3, padding=1, groups=channels * 2)
        self.project = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(self.expand(self.norm(x)))
        left, right = y.chunk(2, dim=1)
        return x + self.project(left * right) * self.beta


class ChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class RestormerBlockLite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = group_norm(channels)
        self.attn = ChannelAttention(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        return x + self.ffn(self.norm(x))


class WindowMixBlockLite(nn.Module):
    def __init__(self, channels: int, window: int = 8):
        super().__init__()
        self.window = window
        self.norm = group_norm(channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        pooled = F.avg_pool2d(y, self.window, stride=self.window, ceil_mode=True)
        pooled = F.interpolate(pooled, size=x.shape[-2:], mode="nearest")
        return x + self.mix(y + pooled) * self.alpha


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, block_factory):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            group_norm(out_channels),
            nn.SiLU(inplace=True),
            block_factory(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, block_factory):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 1),
            block_factory(out_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class UNetRestorer(nn.Module):
    """Three-level U-Net using a NAF, Restormer, or SwinIR-lite block family."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32, family: str = "naf"):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 3
        block_factory = {
            "naf": NAFBlockLite,
            "restormer": RestormerBlockLite,
            "swinir": WindowMixBlockLite,
        }.get(family)
        if block_factory is None:
            raise ValueError(f"Unsupported model family: {family}")
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            ConvBlock(c1),
            block_factory(c1),
        )
        self.down1 = Down(c1, c2, block_factory)
        self.down2 = Down(c2, c3, block_factory)
        self.bottleneck = nn.Sequential(block_factory(c3), ConvBlock(c3), block_factory(c3))
        self.up1 = Up(c3, c2, c2, block_factory)
        self.up2 = Up(c2, c1, c1, block_factory)
        self.head = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        x = self.down2(skip2)
        x = self.bottleneck(x)
        x = self.up1(x, skip2)
        x = self.up2(x, skip1)
        return torch.sigmoid(self.head(x))


class NAFNetLite(UNetRestorer):
    def __init__(self, base_channels: int = 32):
        super().__init__(base_channels=base_channels, family="naf")


class DirectionalScan2d(nn.Module):
    """Four-direction recurrent scan used by the compact Mamba-like model."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.in_proj = nn.Conv2d(channels, channels * 2, 1)
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.decay_logits = nn.Parameter(torch.zeros(channels, 4))
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _scan(self, x: torch.Tensor, decay: torch.Tensor, dim: int, reverse: bool) -> torch.Tensor:
        if reverse:
            x = torch.flip(x, dims=(dim,))
        index = torch.arange(x.shape[dim], device=x.device, dtype=x.dtype)
        shape = (1, 1, 1, -1) if dim == 3 else (1, 1, -1, 1)
        powers = decay.view(1, -1, 1, 1).to(x.dtype) ** index.view(*shape)
        result = powers * torch.cumsum(x / powers.clamp_min(1e-4), dim=dim)
        return torch.flip(result, dims=(dim,)) if reverse else result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        content, gate = self.in_proj(self.norm(x)).chunk(2, dim=1)
        content = self.dwconv(content)
        decays = 0.90 + 0.095 * torch.sigmoid(self.decay_logits)
        scanned = sum(
            self._scan(content, decays[:, index], dim=dimension, reverse=reverse)
            for index, dimension, reverse in ((0, 3, False), (1, 3, True), (2, 2, False), (3, 2, True))
        ) * 0.25
        return residual + self.out_proj(scanned * torch.sigmoid(gate)) * self.scale


class MambaDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = nn.Sequential(group_norm(out_channels), nn.SiLU(inplace=True), ConvBlock(out_channels), DirectionalScan2d(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(x))


class MambaUp(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.fuse = nn.Sequential(nn.Conv2d(out_channels + skip_channels, out_channels, 1), ConvBlock(out_channels))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))


class MambaRestorationNet(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 3
        self.stem = nn.Sequential(nn.Conv2d(3, c1, 3, padding=1), ConvBlock(c1))
        self.down1, self.down2 = MambaDown(c1, c2), MambaDown(c2, c3)
        self.bottleneck = nn.Sequential(ConvBlock(c3), DirectionalScan2d(c3), ConvBlock(c3), DirectionalScan2d(c3))
        self.up1, self.up2 = MambaUp(c3, c2, c2), MambaUp(c2, c1, c1)
        self.head = nn.Conv2d(c1, 3, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.stem(x)
        skip2 = self.down1(skip1)
        x = self.up2(self.up1(self.bottleneck(self.down2(skip2)), skip2), skip1)
        return torch.sigmoid(self.head(x))


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(torch.arange(half, device=time.device, dtype=torch.float32) * (-math.log(10000.0) / max(1, half - 1)))
        arguments = time.float()[:, None] * frequencies[None, :]
        return torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=1)


class ResidualDiffusionRefiner(nn.Module):
    def __init__(self, in_channels: int = 10, base_channels: int = 32):
        super().__init__()
        self.time = SinusoidalTime(base_channels)
        self.time_proj = nn.Linear(base_channels, base_channels * 3)
        self.net = UNetRestorer(in_channels=in_channels, base_channels=base_channels, family="naf")

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        embedding = self.time_proj(self.time(time)).view(time.shape[0], -1, 1, 1)
        return torch.tanh(self.net(x) * 2.0 - 1.0 + embedding[:, :3] * 0.01)


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    kernel_x, kernel_y = kernel_x.repeat(x.shape[1], 1, 1, 1), kernel_y.repeat(x.shape[1], 1, 1, 1)
    grad_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
    grad_y = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)
