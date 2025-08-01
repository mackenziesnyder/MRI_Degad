import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.utils.cbam import CBAM

class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()
        self.k1 = 32
        self.k2 = 64
        self.k3 = 128
        self.k4 = 256

        # Encoder (Contracting Path)
        # Block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, self.k1, 3, padding=1),
            nn.BatchNorm2d(self.k1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k1, self.k1, 3, padding=1),
            nn.BatchNorm2d(self.k1),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam1 = CBAM(self.k1)

        # Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(self.k1, self.k2, 3, padding=1),
            nn.BatchNorm2d(self.k2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k2, self.k2, 3, padding=1),
            nn.BatchNorm2d(self.k2),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam2 = CBAM(self.k2)

        # Block 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(self.k2, self.k3, 3, padding=1),
            nn.BatchNorm2d(self.k3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k3, self.k3, 3, padding=1),
            nn.BatchNorm2d(self.k3),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam3 = CBAM(self.k3)

        # Block 4 (Bridge)
        self.bridge = nn.Sequential(
            nn.Conv2d(self.k3, self.k4, 3, padding=1),
            nn.BatchNorm2d(self.k4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k4, self.k4, 3, padding=1),
            nn.BatchNorm2d(self.k4),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam4 = CBAM(self.k4)

        # Decoder (Expansive Path)
        # Block 1
        self.dec1 = nn.Sequential(
            nn.Conv2d(self.k4 + self.k3, self.k3, 3, padding=1),
            nn.BatchNorm2d(self.k3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k3, self.k3, 3, padding=1),
            nn.BatchNorm2d(self.k3),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam_dec1 = CBAM(self.k3)

        # Block 2
        self.dec2 = nn.Sequential(
            nn.Conv2d(self.k3 + self.k2, self.k2, 3, padding=1),
            nn.BatchNorm2d(self.k2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k2, self.k2, 3, padding=1),
            nn.BatchNorm2d(self.k2),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam_dec2 = CBAM(self.k2)

        # Block 3
        self.dec3 = nn.Sequential(
            nn.Conv2d(self.k2 + self.k1, self.k1, 3, padding=1),
            nn.BatchNorm2d(self.k1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.k1, self.k1, 3, padding=1),
            nn.BatchNorm2d(self.k1),
            nn.LeakyReLU(inplace=True)
        )
        self.cbam_dec3 = CBAM(self.k1)

        # Final output
        self.final = nn.Sequential(
            nn.Conv2d(self.k1, 1, 3, padding=1),
            nn.Sigmoid()
        )

        # Pooling
        self.pool = nn.AvgPool2d(2, 2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc1_cbam = self.cbam1(enc1)
        enc1_pool = self.pool(enc1_cbam)

        enc2 = self.enc2(enc1_pool)
        enc2_cbam = self.cbam2(enc2)
        enc2_pool = self.pool(enc2_cbam)

        enc3 = self.enc3(enc2_pool)
        enc3_cbam = self.cbam3(enc3)
        enc3_pool = self.pool(enc3_cbam)

        # Bridge
        bridge = self.bridge(enc3_pool)
        bridge_cbam = self.cbam4(bridge)

        # Decoder
        dec1 = F.interpolate(bridge_cbam, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc3_cbam], dim=1)
        dec1 = self.dec1(dec1)
        dec1_cbam = self.cbam_dec1(dec1)

        dec2 = F.interpolate(dec1_cbam, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc2_cbam], dim=1)
        dec2 = self.dec2(dec2)
        dec2_cbam = self.cbam_dec2(dec2)

        dec3 = F.interpolate(dec2_cbam, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc1_cbam], dim=1)
        dec3 = self.dec3(dec3)
        dec3_cbam = self.cbam_dec3(dec3)

        out = self.final(dec3_cbam)
        return out

class StackedUNets(nn.Module):
    def __init__(self, input_channels=1):
        super(StackedUNets, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # First UNet takes concatenated input features
        self.unet1 = UNet(in_channels=64 * 3)  # 3 inputs * 64 channels each
        
        # Second UNet takes concatenated features from first UNet output and input
        self.unet2 = UNet(in_channels=64 * 3 + 1)  # (3 inputs * 64 channels) + 1 channel from first UNet

    def forward(self, x1, x2, x3):
        # Initial convolutions for each input
        feat1 = self.initial_conv(x1)
        feat2 = self.initial_conv(x2)
        feat3 = self.initial_conv(x3)
        
        # Concatenate features
        combined_features = torch.cat([feat1, feat2, feat3], dim=1)
        
        # First UNet
        pred1 = self.unet1(combined_features)
        
        # Concatenate first prediction with input features
        combined_features_2 = torch.cat([combined_features, pred1], dim=1)
        
        # Second UNet
        pred2 = self.unet2(combined_features_2)
        
        return pred2


if __name__ == "__main__":
    import torch.optim as optim

    model = StackedUNets()
    inputs = [torch.randn(1, 1, 256, 256) for _ in range(3)]
    output = model(*inputs)
    print(f"Input shapes: {[x.shape for x in inputs]}")
    print(f"Output shape: {output.shape}")
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # PyTorch 'compile' equivalent
    # Define loss function
    criterion = nn.MSELoss()  # or another appropriate loss

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # (Optional) Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print("Model, loss function, optimizer, and scheduler are set up and ready for training!") 