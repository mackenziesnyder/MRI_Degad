import torch
import torch.nn as nn
import torch.nn.functional as F
from model_helpers.cbam import CBAM
from model_helpers.wat import MultiLevelWaveletTransform, WATLayer


class WATUNet(nn.Module):
    def __init__(self, in_channels):
        super(WATUNet, self).__init__()
        self.k1, self.k2, self.k3, self.k4 = 32, 64, 128, 256
        self.wavelet_transform = MultiLevelWaveletTransform(
            in_channels=in_channels, levels=3
        )

        # Encoder
        self.enc1 = self._make_enc_block(in_channels, self.k1)
        self.cbam1 = CBAM(self.k1)
        self.enc2 = self._make_enc_block(self.k1, self.k2)
        self.wat_layer1 = WATLayer(wat_channels=in_channels * 3, level=1)
        self.cbam2 = CBAM(self.k2)
        self.enc3 = self._make_enc_block(self.k2, self.k3)
        self.wat_layer2 = WATLayer(wat_channels=in_channels * 3, level=2)
        self.cbam3 = CBAM(self.k3)

        # Bridge
        self.bridge = self._make_enc_block(self.k3, self.k4)
        self.wat_layer3 = WATLayer(wat_channels=in_channels * 4, level=3)
        self.cbam4 = CBAM(self.k4)

        # Decoder
        self.dec1 = self._make_dec_block(self.k4 + self.k3, self.k3)
        self.cbam_dec1 = CBAM(self.k3)
        self.dec2 = self._make_dec_block(self.k3 + self.k2, self.k2)
        self.cbam_dec2 = CBAM(self.k2)
        self.dec3 = self._make_dec_block(self.k2 + self.k1, self.k1)
        self.cbam_dec3 = CBAM(self.k1)

        self.final = nn.Sequential(nn.Conv2d(self.k1, 1, 3, padding=1), nn.Sigmoid())
        self.pool = nn.AvgPool2d(2, 2)

    def _make_enc_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def _make_dec_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        wat1, wat2, wat3 = self.wavelet_transform(x)

        # Encoder
        enc1_cbam = self.cbam1(self.enc1(x))
        enc2_cbam = self.cbam2(self.wat_layer1(self.enc2(self.pool(enc1_cbam)), wat1))
        enc3_cbam = self.cbam3(self.wat_layer2(self.enc3(self.pool(enc2_cbam)), wat2))

        # Bridge
        bridge_cbam = self.cbam4(
            self.wat_layer3(self.bridge(self.pool(enc3_cbam)), wat3)
        )

        # Decoder
        up1 = F.interpolate(
            bridge_cbam, scale_factor=2, mode="bilinear", align_corners=True
        )
        dec1_cbam = self.cbam_dec1(self.dec1(torch.cat([up1, enc3_cbam], dim=1)))

        up2 = F.interpolate(
            dec1_cbam, scale_factor=2, mode="bilinear", align_corners=True
        )
        dec2_cbam = self.cbam_dec2(self.dec2(torch.cat([up2, enc2_cbam], dim=1)))

        up3 = F.interpolate(
            dec2_cbam, scale_factor=2, mode="bilinear", align_corners=True
        )
        dec3_cbam = self.cbam_dec3(self.dec3(torch.cat([up3, enc1_cbam], dim=1)))

        return self.final(dec3_cbam)


class WATStackedUNets(nn.Module):
    def __init__(self, input_channels=1):
        super(WATStackedUNets, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.unet1 = WATUNet(in_channels=64 * 3)
        self.unet2 = WATUNet(in_channels=64 * 3 + 1)

    def forward(self, x1, x2, x3):
        feat1 = self.initial_conv(x1)
        feat2 = self.initial_conv(x2)
        feat3 = self.initial_conv(x3)

        combined1 = torch.cat([feat1, feat2, feat3], dim=1)
        pred1 = self.unet1(combined1)

        combined2 = torch.cat([combined1, pred1], dim=1)
        pred2 = self.unet2(combined2)

        return pred2
