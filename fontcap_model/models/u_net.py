import torch
import torch.nn as nn



# TODO: parameterise this
class UNet(nn.Module):
    """Implementation of a U-net"""

    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(True)
            )

        def conv_transpose_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
                nn.ReLU(True)
            )

        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(128, 256)

        self.up2 = conv_transpose_block(256, 128)
        self.dec2 = conv_block(256, 128)
        self.up1 = conv_transpose_block(128, 64)
        self.dec1 = conv_block(128, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        encoded1 = self.enc1(x)  # 64x32x32
        encoded2 = self.enc2(self.pool1(encoded1))  # 128x16x16

        bottleneck = self.bottleneck(self.pool2(encoded2))  # 256x8x8

        up2 = self.up2(bottleneck)  # 128x16x16
        decoded2 = self.dec2(torch.cat([up2, encoded2], dim=1))  # 128x16x16
        up1 = self.up1(decoded2)  # 64x32x32
        decoded1 = self.dec1(torch.cat([up1, encoded1], dim=1))  # 64x32x32

        return torch.sigmoid(self.out(decoded1))
