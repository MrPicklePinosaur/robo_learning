import torch
import torch.nn as nn
import torchvision
import math

class PositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim = self.dim // 2
        embedding = torch.exp(-1 * math.log(10000) * torch.arange(half_dim, device=time.device)/(half_dim - 1))
        embedding = time[:, None] * embedding[None, :]
        embedding = torch.stack((embedding.sin(), embedding.cos()), dim=1)
        embedding = embedding.flatten()
        return embedding

