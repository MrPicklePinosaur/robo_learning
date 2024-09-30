from position_embedding import PositionEmbedding
import torch
import torch.nn as nn
import torchvision
import logging

logger = logging.getLogger(__name__)

def crop(x, residual):
    _, _, H, W = x.shape 
    return torchvision.transforms.CenterCrop([H, W])(residual)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, up=True):
        super().__init__()

        self.time_dim = time_dim
        self.time_transform = nn.Linear(self.time_dim, in_channels)

        self.conv0 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

        if up:
            self.transform = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        else:
            self.transform = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x, t, residual=None):
        # TODO when should time embedding be applied?

        # transform time embedding
        t = self.time_transform(t)
        t = t[(...,) + (None,)*2] # add h and width dimentions
        logger.debug('time transform %s %s', t.shape, x.shape)

        x = x+t
        x = self.conv0(x)
        x = self.relu(x)
        logger.debug('block conv0+relu %s', x.shape)
        x = self.transform(x)
        saved_residual=x
        logger.debug('block transform %s', x.shape)

        if residual is not None:
            cropped = crop(x, residual)
            logger.debug('before crop %s', x.shape)
            x = torch.cat([x, cropped], dim=1)
            logger.debug('after crop %s', x.shape)

        x = self.conv1(x)
        x = self.relu(x)
        logger.debug('block conv1+relu %s', x.shape)
        return x, saved_residual


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, time_dim=32):
        super().__init__()

        down_channels = [64, 128, 256, 512, 1024]
        up_channels = [1024, 512, 256, 128, 64]

        self.time_dim = time_dim

        # TODO move the max time steps to hyperparam
        self.time_embed = PositionEmbedding(1000, self.time_dim)
        self.time_embed_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.time_dim, self.time_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(in_channels, down_channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([UNetBlock(down_channels[i], down_channels[i+1], self.time_dim, up=False) for i in range(len(down_channels)-1)])
        self.up_blocks = nn.ModuleList([UNetBlock(up_channels[i], up_channels[i+1], self.time_dim, up=True) for i in range(len(down_channels)-1)])
        self.conv1 = nn.Conv2d(up_channels[-1], up_channels[-1], 3, padding=1)
        self.output_conv = nn.Conv2d(up_channels[-1], out_channels, kernel_size=1)


    def forward(self, x, timestep):

        # time embedding
        t = self.time_embed(x, timestep)
        logger.debug('time embedding %s', t.shape)
        t = self.time_embed_linear(t)
        logger.debug('time embedding %s', t.shape)

        logger.debug('initial dimension %s', x.shape)
        x = self.conv0(x.float())
        logger.debug('after conv0 %s', x.shape)
        
        residuals = []
        for layer in self.down_blocks:
            (x, residual) = layer(x, t)
            residuals.append(residual)
            # logger.debug('appended', x.shape)

        # for r in residuals:
        #     logger.debug(r.shape)

        # we don't need the last residual connection
        logger.debug('residual len %s', len(residuals))

        logger.debug('after down blocks%s ', x.shape)

        for layer in self.up_blocks:
            residual = residuals.pop()
            # logger.debug('popped', residual.shape)
            x, _ = layer(x, t, residual=residual)

        logger.debug('shape after up_blocks %s', x.shape)
        x = self.conv1(x)
        logger.debug('shape after conv1 %s', x.shape)
        x = self.output_conv(x)
        logger.debug('shape after output conv %s', x.shape)
        return x
        
