import torch
import torch.nn as nn
import torch.nn.functional as F    
from models.common import DoubleConv, Down, Up, OutConv

################################################################################
######################## Diff UNet Model Definition ############################
################################################################################ 
class DiffUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(DiffUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, before, after):
        x11 = self.inc(before)
        x12 = self.inc(after)
        diff1 = torch.abs(x11 - x12)

        x21 = self.down1(x11)
        x22 = self.down1(x12)
        diff2 = torch.abs(x21 - x22)

        x31 = self.down2(x21)
        x32 = self.down2(x22)
        diff3 = torch.abs(x31 - x32)

        x41 = self.down3(x31)
        x42 = self.down3(x32)
        diff4 = torch.abs(x41 - x42)

        x5 = self.down4(diff4)
        
        x = self.up1(x5, diff4)
        x = self.up2(x, diff3)
        x = self.up3(x, diff2)
        x = self.up4(x, diff1)

        logits = self.outc(x)
        return logits