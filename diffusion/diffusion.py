import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AnimeFaces

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def forward_pass(x):
    noise = torch.randn_like(x)

def normalize_dataset(norm_loader, w, h):

    pixel_sum = np.zeros(3) # assuming 3 channels for pixel value
    pixel_count = len(norm_loader) * w * h
    for (features, _, _) in norm_loader:
        pixel_sum += features.sum(axis=(0, 2, 3)).numpy()
    mean = pixel_sum/pixel_count

    sum_square_err = np.zeros(3)
    for (features, _, _) in norm_loader:
        sum_square_err += ((features.numpy() - mean.reshape(1, 3, 1, 1)) ** 2).sum(axis=(0, 2, 3))
    std = np.sqrt(sum_square_err/pixel_count)

    return mean, std

# Hyperparameters
BATCH_SIZE=64

# Setup
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Load dataset

# Determine mean and variance of pixel values
transformations = [
    transforms.Resize(64),
    transforms.ToTensor(),
]
preprocess = transforms.Compose(transformations)

dataset = AnimeFaces(img_dir='data', preprocess=preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
norm_mean, norm_std = normalize_dataset(dataloader, 64, 64)

# Apply the normalization transformations
transformations.append(transforms.Normalize(mean=norm_mean, std=norm_std))
preprocess = transforms.Compose(transformations)
dataset = AnimeFaces(img_dir='data', preprocess=preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
