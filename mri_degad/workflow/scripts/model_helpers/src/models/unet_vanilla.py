import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet, self).__init__()
        k = [32, 64, 128, 256]
        self.enc1 = self._make_block(in_channels, k[0])
        self.enc2 = self._make_block(k[0], k[1])
        self.enc3 = self._make_block(k[1], k[2])
        self.bridge = self._make_block(k[2], k[3])
        self.dec1 = self._make_block(k[3]+k[2], k[2])
        self.dec2 = self._make_block(k[2]+k[1], k[1])
        self.dec3 = self._make_block(k[1]+k[0], k[0])
        self.final = nn.Sequential(nn.Conv2d(k[0], 1, 3, padding=1), nn.Sigmoid())
        self.pool = nn.AvgPool2d(2, 2)

    def _make_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pool = self.pool(enc1)
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool(enc2)
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool(enc3)
        bridge = self.bridge(enc3_pool)
        dec1 = F.interpolate(bridge, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc3], dim=1)
        dec1 = self.dec1(dec1)
        dec2 = F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec3 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc1], dim=1)
        dec3 = self.dec3(dec3)
        return self.final(dec3)

class UNetVanilla(nn.Module):
    def __init__(self, input_channels=1):
        super(UNetVanilla, self).__init__()
        self.unet = UNet(in_channels=input_channels)

    def forward(self, x):
        # x: (B, C, H, W)
        return self.unet(x)

if __name__ == "__main__":
    import torch.optim as optim

    print("Testing UNetVanilla (no SAP)...")
    model_vanilla = UNetVanilla(input_channels=1)
    x_single = torch.randn(1, 1, 256, 256)
    output_vanilla = model_vanilla(x_single)
    print(f"Input shape: {x_single.shape}")
    print(f"Output shape: {output_vanilla.shape}")
    num_params_vanilla = sum(p.numel() for p in model_vanilla.parameters() if p.requires_grad)
    print(f"Number of trainable parameters (Vanilla): {num_params_vanilla:,}")
    criterion_vanilla = nn.MSELoss()
    optimizer_vanilla = optim.Adam(model_vanilla.parameters(), lr=1e-3)
    scheduler_vanilla = optim.lr_scheduler.StepLR(optimizer_vanilla, step_size=10, gamma=0.1)
    print("Model, loss function, optimizer, and scheduler are set up and ready for training!") 