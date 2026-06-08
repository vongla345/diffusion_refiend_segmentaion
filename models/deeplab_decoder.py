"""DeepLabV3 decoder (ASPP + progressive upsample) for UNI patch grids."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling used in notebook version."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int = 256,
        rates: Tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.atrous = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                for r in rates
            ]
        )
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        n_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * n_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        h, w = x.shape[2:]
        feats = [self.conv1x1(x)]
        feats += [m(x) for m in self.atrous]
        gap = self.gap(x)
        feats.append(
            F.interpolate(gap, size=(h, w), mode="bilinear", align_corners=False)
        )
        return self.project(torch.cat(feats, dim=1))


class DeepLabV3Decoder(nn.Module):
    """ASPP on patch grid then 4× transposed upsample + 1×1 logits."""

    def __init__(
        self,
        in_dim: int = 1024,
        aspp_out: int = 256,
        rates: Tuple[int, ...] = (6, 12, 18),
    ):
        super().__init__()
        self.aspp = ASPP(in_dim, aspp_out, rates)
        self.up_refine = nn.Sequential(
            nn.ConvTranspose2d(aspp_out, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = self.aspp(x)
        x = self.up_refine(x)
        return self.head(x)
