import torch
import torch.nn as nn
import torchvision

def crop(x, residual):
    _, _, H, W = x.shape 
    return torchvision.transforms.CenterCrop([H, W])(residual)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=True):
        super().__init__()

        self.conv0 = nn.Conv2d(in_channels, in_channels, 3)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.relu = nn.ReLU()

        if up:
            self.transform = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        else:
            self.transform = nn.MaxPool2d(2)

    def forward(self, x, residual=None):
        x = self.conv0(x)
        x = self.relu(x)
        print('block conv0+relu', x.shape)
        x = self.transform(x)
        saved_residual=x
        print('block transform', x.shape)

        if residual is not None:
            cropped = crop(x, residual)
            print('before crop', x.shape)
            x = torch.cat([x, cropped], dim=1)
            print('after crop', x.shape)

        x = self.conv1(x)
        x = self.relu(x)
        print('block conv1+relu', x.shape)
        return x, saved_residual


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        down_channels = [64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=0)
        self.down_blocks = nn.ModuleList([UNetBlock(down_channels[i], down_channels[i+1], up=False) for i in range(len(down_channels)-1)])
        self.up_blocks = nn.ModuleList([UNetBlock(up_channels[i], up_channels[i+1], up=True) for i in range(len(down_channels)-1)])
        self.conv1 = nn.Conv2d(up_channels[-1], up_channels[-1], 3, padding=0)
        self.output_conv = nn.Conv2d(up_channels[-1], out_channels, 1)


    def forward(self, x):
        print('initial dimension', x.shape)
        x = self.conv0(x)
        print('after conv0', x.shape)
        
        residuals = []
        for layer in self.down_blocks:
            (x, residual) = layer(x)
            residuals.append(residual)
            # print('appended', x.shape)

        # for r in residuals:
        #     print(r.shape)

        # we don't need the last residual connection
        print('residual len', len(residuals))

        print('after down blocks', x.shape)

        for layer in self.up_blocks:
            residual = residuals.pop()
            # print('popped', residual.shape)
            x, _ = layer(x, residual=residual)

        print('shape after up_blocks', x.shape)
        x = self.conv1(x)
        print('shape after conv1', x.shape)
        x = self.output_conv(x)
        print('shape after output conv', x.shape)
        return x
        
