import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils.cbam import CBAM
from src.models.utils.wat import MultiLevelWaveletTransform, WATLayer

class WATUNet(nn.Module):
    def __init__(self, in_channels):
        super(WATUNet, self).__init__()
        k = [32, 64, 128, 256]
        self.wavelet_transform = MultiLevelWaveletTransform(in_channels=in_channels, levels=3)
        self.enc1 = self._make_block(in_channels, k[0]); self.cbam1 = CBAM(k[0])
        self.enc2 = self._make_block(k[0], k[1]); self.wat1 = WATLayer(in_channels*3, 1); self.cbam2 = CBAM(k[1])
        self.enc3 = self._make_block(k[1], k[2]); self.wat2 = WATLayer(in_channels*3, 2); self.cbam3 = CBAM(k[2])
        self.bridge = self._make_block(k[2], k[3]); self.wat3 = WATLayer(in_channels*4, 3); self.cbam4 = CBAM(k[3])
        self.dec1 = self._make_block(k[3]+k[2], k[2]); self.cbam_dec1 = CBAM(k[2])
        self.dec2 = self._make_block(k[2]+k[1], k[1]); self.cbam_dec2 = CBAM(k[1])
        self.dec3 = self._make_block(k[1]+k[0], k[0]); self.cbam_dec3 = CBAM(k[0])
        self.final = nn.Sequential(nn.Conv2d(k[0], 1, 3, padding=1), nn.Sigmoid())
        self.pool = nn.AvgPool2d(2, 2)

    def _make_block(self, in_c, out_c):
        return nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(inplace=True),
                             nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.LeakyReLU(inplace=True))

    def forward(self, x):
        wat1, wat2, wat3 = self.wavelet_transform(x)
        enc1_cbam = self.cbam1(self.enc1(x))
        enc2_cbam = self.cbam2(self.wat1(self.enc2(self.pool(enc1_cbam)), wat1))
        enc3_cbam = self.cbam3(self.wat2(self.enc3(self.pool(enc2_cbam)), wat2))
        bridge_cbam = self.cbam4(self.wat3(self.bridge(self.pool(enc3_cbam)), wat3))
        dec1_cbam = self.cbam_dec1(self.dec1(torch.cat([F.interpolate(bridge_cbam, scale_factor=2), enc3_cbam], 1)))
        dec2_cbam = self.cbam_dec2(self.dec2(torch.cat([F.interpolate(dec1_cbam, scale_factor=2), enc2_cbam], 1)))
        dec3_cbam = self.cbam_dec3(self.dec3(torch.cat([F.interpolate(dec2_cbam, scale_factor=2), enc1_cbam], 1)))
        return self.final(dec3_cbam)

class MultiInputWATUNet(nn.Module):
    def __init__(self, input_channels=1):
        super(MultiInputWATUNet, self).__init__()
        self.initial_conv = nn.Sequential(nn.Conv2d(input_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.unet = WATUNet(in_channels=64 * 3)

    def forward(self, x1, x2, x3):
        combined = torch.cat([self.initial_conv(x) for x in [x1, x2, x3]], dim=1)
        return self.unet(combined)

if __name__ == "__main__":
    import torch.optim as optim

    model = MultiInputWATUNet()
    inputs = [torch.randn(1, 1, 256, 256) for _ in range(3)]
    output = model(*inputs)
    print(f"Input shapes: {[x.shape for x in inputs]}")
    print(f"Output shape: {output.shape}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # --- PyTorch "compile" equivalent ---
    # Choose a loss function
    criterion = nn.MSELoss()  # or nn.MSELoss(), nn.CrossEntropyLoss(), etc.

    # Choose an optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # (Optional) Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Model, loss function, optimizer, and scheduler are set up and ready for training!") 