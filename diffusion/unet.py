from enum import UNIQUE
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super.__init__()

        self.conv0 = nn.Conv2d(in_channels, in_channels, 3)
        self.mpool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = nn.ReLU(x)
        x = self.mpool(x)
        x = nn.ReLU(x)
        x = self.conv1(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super.__init__()

        down_channels = [64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=0)

        self.down_blocks = nn.ModuleList([UNetBlock(down_channels[i], down_channels[i+1]) for i in range(len(down_channels)-1)])

        self.up_blocks = nn.ModuleList([UNetBlock(up_channels[i], up_channels[i+1]) for i in range(len(down_channels)-1)])
        self.conv1 = nn.Conv2d(up_channels[-1], out_channels, 3, padding=0)


    def forward(self, x):
        x = self.conv0(x)

        for layer in self.down_blocks:
            x = layer(x)

        for layer in self.up_blocks:
            x = layer(x)

        x = self.conv1(x)
        return x
        
