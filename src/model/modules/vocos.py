import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.transforms import InverseSpectrogram


# Global Response Normalization
class GRN(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, channel), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, 1, channel), requires_grad=True)

    def forward(self, x):
        # [B, T, C]
        gx = x.norm(dim=1, keepdim=True)
        nx = gx / gx.mean(dim=-1, keepdim=True)
        return self.gamma * (nx * x) + self.bias + x


class ConvNeXtLayer(nn.Module):
    def __init__(self, channel, h_channel):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channel, channel, kernel_size=7, padding=3, groups=channel
        )
        self.norm = nn.LayerNorm(channel)
        self.pw1 = nn.Linear(channel, h_channel)
        self.pw2 = nn.Linear(h_channel, channel)
        self.grn = GRN(h_channel)

    def forward(self, x):
        res = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.grn(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = res + x
        return x


class Vocos(nn.Module):
    def __init__(
        self, in_channel, channel, h_channel, out_channel, num_layers, istft_config
    ):
        super().__init__()
        self.pad = nn.ReflectionPad1d([1, 0])
        self.in_conv = nn.Conv1d(in_channel, channel, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(channel)
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channel, h_channel) for _ in range(num_layers)]
        )
        self.norm_last = nn.LayerNorm(channel)
        self.out_conv = nn.Conv1d(channel, out_channel, 1)
        self.istft = InverseSpectrogram(**istft_config)

    def forward(self, x):
        x = self.pad(x)
        x = self.in_conv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_last(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_conv(x)
        mag, phase = x.chunk(2, dim=1)
        mag = mag.exp().clamp_max(max=1e2)
        s = mag * (phase.cos() + 1j * phase.sin())
        o = self.istft(s).unsqueeze(1)
        return o
