import torch
import torch.nn as nn
import torchvision
import math

# inspired by https://towardsdatascience.com/diffusion-model-from-scratch-in-pytorch-ddpm-9d9760528946
class PositionEmbedding(nn.Module):
    def __init__(self, time_steps, dim):
        super().__init__()
        self.dim = dim
        self.time_steps = time_steps
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(-1 * math.log(10000) * torch.arange(0, self.dim, step=2).float()/(self.dim))
        self.embedding = torch.zeros(time_steps, self.dim, requires_grad=False)
        self.embedding[: , 0::2] = torch.sin(position * div)
        self.embedding[: , 1::2] = torch.cos(position * div)

    def forward(self, x, t):
        # print('dimensions', x.shape, t.shape)
        embeds = self.embedding[t.squeeze(-1)].to(x.device)
        return embeds[:, :, None, None]

