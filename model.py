"""Paper-aligned implementation of SBCNet.

The implementation follows the module definitions in the manuscript:
PVTv2-B3 encoder -> CFE -> semantic-guided boundary prior modeling -> RBD.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.Encoder import Encoder


def build_norm(channels: int) -> nn.GroupNorm:
    """Group normalization used throughout the non-backbone network."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def upsample_to(x: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class ConvGNAct(nn.Module):
    """Convolution followed by GroupNorm and an optional SiLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        *,
        groups: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            build_norm(out_channels),
            nn.SiLU(inplace=True) if activation else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScharrEdge(nn.Module):
    """Fixed depth-wise Scharr operator used by the CFE spatial branch."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        scharr_x = torch.tensor(
            [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]]
        ).view(1, 1, 3, 3)
        scharr_y = torch.tensor(
            [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", scharr_x.repeat(channels, 1, 1, 1))
        self.register_buffer("kernel_y", scharr_y.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.channels)
        gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.channels)
        return torch.sqrt(gx.square() + gy.square() + 1e-6)


class CFEBlock(nn.Module):
    """Camouflage feature enhancement at one encoder level.

    Implements Eqs. (1)--(3): spatial edge enhancement, frequency texture
    filtering, and adaptive dual-domain fusion.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scharr = ScharrEdge(channels)

        self.spatial_in = ConvGNAct(channels, channels, 1)
        self.spatial_out = ConvGNAct(channels, channels, 1)

        # A complex 1x1 convolution is represented by applying a real-valued
        # 1x1 convolution to concatenated real and imaginary components.
        self.frequency_transform = ConvGNAct(2 * channels, 2 * channels, 1)
        self.frequency_gate = nn.Sequential(
            ConvGNAct(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.frequency_out = ConvGNAct(channels, channels, 3)

        self.dual_projection = ConvGNAct(channels, channels, 1)
        self.fusion_gate = nn.Sequential(
            ConvGNAct(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.output = ConvGNAct(channels, channels, 3)

    def _spatial_branch(self, x: torch.Tensor) -> torch.Tensor:
        edge = self.scharr(x)
        return self.spatial_out(self.spatial_in(x) + edge)

    def _frequency_branch(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        input_dtype = x.dtype
        device_type = "cuda" if x.is_cuda else "cpu"

        # FFT kernels are executed in FP32 for numerical stability.
        with torch.amp.autocast(device_type=device_type, enabled=False):
            spectrum = torch.fft.rfft2(x.float(), norm="ortho")
            spectrum_ri = torch.cat((spectrum.real, spectrum.imag), dim=1)
            spectrum_ri = self.frequency_transform(spectrum_ri)
            real, imag = spectrum_ri.chunk(2, dim=1)
            reconstructed = torch.fft.irfft2(
                torch.complex(real.contiguous(), imag.contiguous()),
                s=(height, width),
                norm="ortho",
            )
            frequency = self.frequency_out(
                self.frequency_gate(reconstructed) * reconstructed
            )
        return frequency.to(input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dual = self._spatial_branch(x) + self._frequency_branch(x)
        alpha = self.fusion_gate(dual)
        return self.output(alpha * self.dual_projection(dual) + x)


class CFE(nn.Module):
    """Projects PVTv2-B3 features to 128 channels and enhances all levels."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        if len(encoder_channels) != 4:
            raise ValueError("encoder_channels must contain [C4, C3, C2, C1].")

        c4, c3, c2, c1 = encoder_channels
        self.projections = nn.ModuleList(
            [
                ConvGNAct(c1, out_channels, 1),
                ConvGNAct(c2, out_channels, 1),
                ConvGNAct(c3, out_channels, 1),
                ConvGNAct(c4, out_channels, 1),
            ]
        )
        self.blocks = nn.ModuleList([CFEBlock(out_channels) for _ in range(4)])

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        features = [proj(x) for proj, x in zip(self.projections, (x1, x2, x3, x4))]
        enhanced = [block(x) for block, x in zip(self.blocks, features)]
        f1, f2, f3, f4 = features
        e1, e2, e3, e4 = enhanced
        return e1, e2, e3, e4, f3, f4


class SGP(nn.Module):
    """Semantic global prior construction defined in Eq. (4)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.high_projection = ConvGNAct(channels, channels, 1)
        self.mid_projection = ConvGNAct(channels, channels, 1)
        self.fusion = ConvGNAct(channels, channels, 3)

    def forward(self, f3: torch.Tensor, f4: torch.Tensor) -> torch.Tensor:
        high = upsample_to(self.high_projection(f4), f3.shape[-2:])
        return self.fusion(high + self.mid_projection(f3))


class SBP(nn.Module):
    """Semantic-guided boundary selection defined in Eq. (5)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.e1_projection = ConvGNAct(channels, channels, 1)
        self.e2_projection = ConvGNAct(channels, channels, 1)
        self.activation = ConvGNAct(channels, channels, 3)
        self.semantic_projection = ConvGNAct(channels, channels, 1)
        self.boundary = ConvGNAct(channels, channels, 3)
        self.boundary_head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        semantic_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_size = e1.shape[-2:]
        e2 = upsample_to(self.e2_projection(e2), target_size)
        activation = torch.sigmoid(self.activation(e2 + self.e1_projection(e1)))

        semantic_prior = upsample_to(semantic_prior, target_size)
        boundary_prior = self.boundary(
            activation * self.semantic_projection(semantic_prior)
        )
        return boundary_prior, self.boundary_head(boundary_prior)


class DynamicKernelAttention(nn.Module):
    """Stage-specific dynamic multi-kernel structure modeling used by CSF."""

    class Branch(nn.Module):
        def __init__(self, channels: int, kernel_size: int) -> None:
            super().__init__()
            padding = kernel_size // 2
            self.horizontal = nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                groups=channels,
                bias=False,
            )
            self.vertical = nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                groups=channels,
                bias=False,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.vertical(self.horizontal(x))

    def __init__(self, channels: int, kernels: Sequence[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [self.Branch(channels, kernel_size) for kernel_size in kernels]
        )
        self.attention = nn.Conv2d(channels, len(kernels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(
            self.attention(F.adaptive_avg_pool2d(x, output_size=1)), dim=1
        )
        output = torch.zeros_like(x)
        for index, branch in enumerate(self.branches):
            output = output + branch(x) * weights[:, index : index + 1]
        return output


class CSF(nn.Module):
    """Cross-stage fusion unit defined in Eq. (6)."""

    def __init__(self, channels: int, kernels: Sequence[int]) -> None:
        super().__init__()
        self.pre_fusion = nn.Sequential(
            ConvGNAct(3 * channels, channels, 1),
            ConvGNAct(channels, channels, 3),
        )
        self.structure = DynamicKernelAttention(channels, kernels)
        self.structure_scale = nn.Parameter(torch.tensor(0.1))
        self.output_projection = nn.Sequential(
            ConvGNAct(channels, channels, 3),
            ConvGNAct(channels, channels, 3, activation=False),
        )

    def forward(
        self,
        local_feature: torch.Tensor,
        deep_feature: torch.Tensor,
        boundary_prior: torch.Tensor,
    ) -> torch.Tensor:
        target_size = local_feature.shape[-2:]
        deep_feature = upsample_to(deep_feature, target_size)
        boundary_prior = upsample_to(boundary_prior, target_size)

        x = self.pre_fusion(
            torch.cat((local_feature, deep_feature, boundary_prior), dim=1)
        )
        x_hat = x + self.structure_scale * self.structure(x)
        return x_hat + self.output_projection(x)


class CPG(nn.Module):
    """Coarse prediction guidance defined in Eqs. (7)--(8)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.prediction_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.uncertainty_boost = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.refine = ConvGNAct(channels, channels, 3, activation=False)
        self.refine_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        feature: torch.Tensor,
        previous_prediction: torch.Tensor,
    ) -> torch.Tensor:
        previous_prediction = upsample_to(previous_prediction, feature.shape[-2:])
        probability = torch.sigmoid(previous_prediction)
        uncertainty = 1.0 - torch.abs(2.0 * probability - 1.0)
        guidance = self.prediction_gate(probability) * self.uncertainty_boost(
            uncertainty
        )
        residual = self.refine(feature * guidance)
        return feature + self.refine_scale * residual


class RBDStage(nn.Module):
    def __init__(self, channels: int, kernels: Sequence[int]) -> None:
        super().__init__()
        self.csf = CSF(channels, kernels)
        self.cpg = CPG(channels)
        self.prediction = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        local_feature: torch.Tensor,
        deep_feature: torch.Tensor,
        boundary_prior: torch.Tensor,
        previous_prediction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feature = self.csf(local_feature, deep_feature, boundary_prior)
        feature = self.cpg(feature, previous_prediction)
        return feature, self.prediction(feature)


class RBD(nn.Module):
    """Three-stage region--boundary coupled decoder."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.stage3 = RBDStage(channels, kernels=(9, 15, 31))
        self.stage2 = RBDStage(channels, kernels=(7, 11, 15))
        self.stage1 = RBDStage(channels, kernels=(5, 9, 13))

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
        e3: torch.Tensor,
        e4: torch.Tensor,
        boundary_prior: torch.Tensor,
        p4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d3, p3 = self.stage3(e3, e4, boundary_prior, p4)
        d2, p2 = self.stage2(e2, d3, boundary_prior, p3)
        _, p1 = self.stage1(e1, d2, boundary_prior, p2)
        return p3, p2, p1


class SBCNet(nn.Module):
    """Semantic--Boundary Coupled Network for camouflaged object detection."""

    def __init__(self, config, pretrained: bool = True) -> None:
        super().__init__()
        channels = int(getattr(config, "inter_channel", 128))
        if channels != 128:
            raise ValueError("The paper uses inter_channel=128.")

        self.encoder = Encoder(config, pretrained)
        self.cfe = CFE(config.lateral_channels, out_channels=channels)
        self.sgp = SGP(channels)
        self.sbp = SBP(channels)
        self.coarse_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.rbd = RBD(channels)

    def forward(self, x: torch.Tensor):
        x1, x2, x3, x4 = self.encoder(x)
        e1, e2, e3, e4, f3, f4 = self.cfe(x1, x2, x3, x4)

        semantic_prior = self.sgp(f3, f4)
        boundary_prior, boundary_prediction = self.sbp(e1, e2, semantic_prior)

        p4 = self.coarse_head(e4)
        p3, p2, p1 = self.rbd(e1, e2, e3, e4, boundary_prior, p4)

        output_size = x.shape[-2:]
        p1 = upsample_to(p1, output_size)
        if not self.training:
            return torch.sigmoid(p1)

        boundary_prediction = upsample_to(boundary_prediction, output_size)
        region_predictions = [
            upsample_to(prediction, output_size)
            for prediction in (p4, p3, p2, p1)
        ]
        return boundary_prediction, region_predictions


Network = SBCNet
