"""CaVE from CODiff, adapted for self-contained CR-VLA use.

The module layout and parameter names follow the MIT-licensed CODiff CaVE
implementation so its published state dictionaries can be loaded directly.
"""

from collections import OrderedDict
from collections.abc import Sequence

import torch
import torch.nn as nn


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("OrderedDict input is not supported")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            modules.extend(module.children())
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv(
    in_channels: int = 64,
    out_channels: int = 64,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    bias: bool = True,
    mode: str = "CBR",
    negative_slope: float = 0.2,
):
    layers = []
    for token in mode:
        if token == "C":
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        elif token == "T":
            layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            )
        elif token == "B":
            layers.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-4, affine=True))
        elif token == "I":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif token == "R":
            layers.append(nn.ReLU(inplace=True))
        elif token == "r":
            layers.append(nn.ReLU(inplace=False))
        elif token == "L":
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif token == "l":
            layers.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif token in {"2", "3", "4"}:
            layers.append(nn.PixelShuffle(upscale_factor=int(token)))
        elif token == "U":
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif token == "u":
            layers.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif token == "v":
            layers.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif token == "M":
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif token == "A":
            layers.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError(f"undefined convolution mode token: {token}")
    return sequential(*layers)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = True,
        mode: str = "CRC",
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("ResBlock requires matching input and output widths")
        if mode[0] in {"R", "L"}:
            mode = mode[0].lower() + mode[1:]
        self.res = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            mode,
            negative_slope,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.res(x)


class QFAttention(ResBlock):
    def forward(self, x: torch.Tensor, gamma, beta) -> torch.Tensor:
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x + gamma * self.res(x) + beta


def _upsample_pixelshuffle(in_channels, out_channels, bias=True, mode="2R"):
    scale = int(mode[0])
    return conv(in_channels, out_channels * scale**2, bias=bias, mode="C" + mode)


def _upsample_upconv(in_channels, out_channels, bias=True, mode="2R"):
    prefix = {"2": "UC", "3": "uC", "4": "vC"}[mode[0]]
    return conv(in_channels, out_channels, bias=bias, mode=mode.replace(mode[0], prefix))


def _upsample_convtranspose(in_channels, out_channels, bias=True, mode="2R"):
    scale = int(mode[0])
    return conv(
        in_channels,
        out_channels,
        kernel_size=scale,
        stride=scale,
        padding=0,
        bias=bias,
        mode=mode.replace(mode[0], "T"),
    )


def _downsample_strideconv(in_channels, out_channels, bias=True, mode="2R"):
    scale = int(mode[0])
    return conv(
        in_channels,
        out_channels,
        kernel_size=scale,
        stride=scale,
        padding=0,
        bias=bias,
        mode=mode.replace(mode[0], "C"),
    )


def _downsample_pool(in_channels, out_channels, bias, mode, pool_token):
    scale = int(mode[0])
    replaced = mode.replace(mode[0], pool_token + "C")
    pool = conv(kernel_size=scale, stride=scale, mode=replaced[0])
    tail = conv(in_channels, out_channels, bias=bias, mode=replaced[1:])
    return sequential(pool, tail)


class CaVE(nn.Module):
    """Compression-aware visual embedder used by CODiff."""

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nc: Sequence[int] = (128, 256, 512, 1024),
        nb: int = 4,
        act_mode: str = "BR",
        downsample_mode: str = "strideconv",
        upsample_mode: str = "convtranspose",
    ) -> None:
        super().__init__()
        if len(nc) != 4:
            raise ValueError("CaVE requires four channel widths")
        self.m_head = conv(in_nc, nc[0], bias=True, mode="C")
        self.nb = nb
        self.nc = list(nc)

        if downsample_mode == "strideconv":
            downsample = _downsample_strideconv
        elif downsample_mode == "avgpool":

            def downsample(i, o, bias, mode):
                return _downsample_pool(i, o, bias, mode, "A")

        elif downsample_mode == "maxpool":

            def downsample(i, o, bias, mode):
                return _downsample_pool(i, o, bias, mode, "M")

        else:
            raise NotImplementedError(f"unsupported downsample mode: {downsample_mode}")

        self.m_down1 = sequential(
            *[ResBlock(nc[0], nc[0], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample(nc[0], nc[1], bias=True, mode="2"),
        )
        self.m_down2 = sequential(
            *[ResBlock(nc[1], nc[1], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample(nc[1], nc[2], bias=True, mode="4"),
        )
        self.m_down3 = sequential(
            *[ResBlock(nc[2], nc[2], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            downsample(nc[2], nc[3], bias=True, mode="4"),
        )
        self.m_body_encoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )
        self.m_body_decoder = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )

        upsample = {
            "upconv": _upsample_upconv,
            "pixelshuffle": _upsample_pixelshuffle,
            "convtranspose": _upsample_convtranspose,
        }.get(upsample_mode)
        if upsample is None:
            raise NotImplementedError(f"unsupported upsample mode: {upsample_mode}")
        self.m_up3 = nn.ModuleList(
            [upsample(nc[3], nc[2], bias=True, mode="4")]
            + [QFAttention(nc[2], nc[2], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )
        self.m_up2 = nn.ModuleList(
            [upsample(nc[2], nc[1], bias=True, mode="4")]
            + [QFAttention(nc[1], nc[1], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )
        self.m_up1 = nn.ModuleList(
            [upsample(nc[1], nc[0], bias=True, mode="2")]
            + [QFAttention(nc[0], nc[0], bias=True, mode="C" + act_mode + "C") for _ in range(nb)]
        )
        self.m_tail = conv(nc[0], out_nc, bias=True, mode="C")
        self.qf_pred = sequential(
            *[ResBlock(nc[3], nc[3], bias=True, mode="C" + act_mode + "C") for _ in range(nb)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(nc[3], nc[3]),
            nn.ReLU(),
            nn.Linear(nc[3], nc[3]),
            nn.ReLU(),
            nn.Linear(nc[3], 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _pad(x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_bottom = (-height) % 8
        pad_right = (-width) % 8
        return nn.ReplicationPad2d((0, pad_right, 0, pad_bottom))(x)

    def encode_map(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        x = self.m_head(x)
        x = self.m_down1(x)
        x = self.m_down2(x)
        x = self.m_down3(x)
        return self.m_body_encoder(x)

    def get_visual_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_map(x).flatten(-2, -1).permute(0, 2, 1).contiguous()

    def pred_qf(self, x: torch.Tensor) -> torch.Tensor:
        return self.qf_pred(self.encode_map(x))

    def forward(self, x: torch.Tensor):
        height, width = x.shape[-2:]
        padded = self._pad(x)
        x1 = self.m_head(padded)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        encoded = self.m_body_encoder(x4)
        qf = self.qf_pred(encoded)
        decoded = self.m_body_decoder(encoded) + x4

        decoded = self.m_up3[0](decoded)
        for index in range(self.nb):
            decoded = self.m_up3[index + 1](decoded, 1.0, 1.0)
        decoded = decoded + x3
        decoded = self.m_up2[0](decoded)
        for index in range(self.nb):
            decoded = self.m_up2[index + 1](decoded, 1.0, 1.0)
        decoded = decoded + x2
        decoded = self.m_up1[0](decoded)
        for index in range(self.nb):
            decoded = self.m_up1[index + 1](decoded, 1.0, 1.0)
        decoded = self.m_tail(decoded + x1)
        return decoded[..., :height, :width], qf
