import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveletTransform(nn.Module):
    def __init__(self, channels):
        super(WaveletTransform, self).__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) / 2.0
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) / 2.0
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) / 2.0
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        weights = filters.repeat(channels, 1, 1, 1)
        self.register_buffer('weights', weights)
        self.groups = channels
    def forward(self, x):
        out = F.conv2d(x, self.weights, stride=2, padding=0, groups=self.groups)
        c = x.shape[1]
        ll = out[:, 0*c:1*c, :, :]
        lh = out[:, 1*c:2*c, :, :]
        hl = out[:, 2*c:3*c, :, :]
        hh = out[:, 3*c:4*c, :, :]
        return ll, lh, hl, hh

class MultiLevelWaveletTransform(nn.Module):
    def __init__(self, in_channels, levels=3):
        super(MultiLevelWaveletTransform, self).__init__()
        self.levels = levels
        self.dwt = WaveletTransform(in_channels)
    def forward(self, x):
        coeffs = []
        current_input = x
        for level in range(self.levels):
            ll, lh, hl, hh = self.dwt(current_input)
            if level < 2:
                coeffs.append(torch.cat([lh, hl, hh], dim=1))
            else:
                coeffs.append(torch.cat([ll, lh, hl, hh], dim=1))
            current_input = ll
        return coeffs[0], coeffs[1], coeffs[2]

class WATLayer(nn.Module):
    def __init__(self, wat_channels, level):
        super(WATLayer, self).__init__()
        k = 16 * (2**level)
        out_channels = 32 * (2**level)
        self.conv_prod = nn.Sequential(
            nn.Conv2d(wat_channels, k, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, out_channels, 3, padding=1)
        )
        self.conv_sum = nn.Sequential(
            nn.Conv2d(wat_channels, k, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(k, out_channels, 3, padding=1)
        )
        self.final_relu = nn.ReLU(inplace=True)
    def forward(self, x, wat):
        watp_prod = self.conv_prod(wat)
        watp_sum = self.conv_sum(wat)
        x = x * watp_prod
        x = x + watp_sum
        return self.final_relu(x) 