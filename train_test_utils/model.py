# Model for RadarHD

# Adapted from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch.nn as nn
import torch

from train_test_utils.unet_parts import *


class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.up5 = Up_nocat(64, 64, bilinear)
        self.up6 = Up_nocat(64, 64, bilinear)
        self.up7 = Up_nocat(64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.up5(x)
        x = self.up6(x)
        x = self.up7(x)
        conv_out = self.outc(x)
        logits = self.final_sigmoid(conv_out)

        return logits