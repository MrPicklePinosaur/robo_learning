import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import AnimeFaces
from unet import UNet

# TODO try other noise schedules
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# Hyperparameters
IMG_SIZE=64
BATCH_SIZE=64
T=300


'''
x_t = sqrt(overline(alpha_t)) x_0 + sqrt(1 - overline(alpha_t)) epsilon
A = sqrt(overline(alpha_t))
B = sqrt(1 - overline(alpha_t))
'''

betas = linear_beta_schedule(T)
alphas = 1 - betas
alphas_overline = torch.cumprod(alphas, axis=0)
A = torch.sqrt(alphas_overline)
B = torch.sqrt(1-alphas_overline)


# TODO make this work for a batch
# returns noisy mean, variance
def forward_pass(x, t):
    noise = torch.randn_like(x)
    return A[t] * x + B[t] * noise, noise


# converts from a tensor to viewable image
# removes the normalization that is done
reverse_transformations = transforms.Compose([
    transforms.Lambda(lambda t: (t+1)/2),
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
    transforms.Lambda(lambda t: t*255.),
    transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    transforms.ToPILImage()
])

def normalize_dataset(norm_loader, w, h):

    pixel_sum = np.zeros(3) # assuming 3 channels for pixel value
    pixel_count = len(norm_loader) * w * h
    for img in norm_loader:
        pixel_sum += img.sum(axis=(0, 2, 3)).numpy()
    mean = pixel_sum/pixel_count

    sum_square_err = np.zeros(3)
    for img in norm_loader:
        sum_square_err += ((img.numpy() - mean.reshape(1, 3, 1, 1)) ** 2).sum(axis=(0, 2, 3))
    std = np.sqrt(sum_square_err/pixel_count)

    return mean, std

# Setup
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# Load dataset

# Determine mean and variance of pixel values
transformations = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                 # scale to [0, 1]
    transforms.Lambda(lambda t: (t*2) - 1) # scale to [-1, 1]
]
preprocess = transforms.Compose(transformations)

dataset = AnimeFaces(img_dir='data', preprocess=preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
norm_mean, norm_std = normalize_dataset(dataloader, IMG_SIZE, IMG_SIZE)

# Apply the normalization transformations
# transformations.append(transforms.Normalize(mean=norm_mean, std=norm_std))
preprocess = transforms.Compose(transformations)
dataset = AnimeFaces(img_dir='data', preprocess=preprocess)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Apply forward diffusion
'''
image = next(iter(dataloader))[0] # grab some image
print(image.shape)
num_images = 10 # number of images to display in graph
step_size = int(T/num_images)

plt.figure()

fig_index = 1
for t in range(0, T, step_size):
    image, noise = forward_pass(image, t)
    plt.subplot(1, num_images+1, fig_index)
    plt.imshow(reverse_transformations(image))
    fig_index += 1

plt.show()
'''


x = torch.randn(1, 3, 572, 572)
model = UNet(in_channels=3, out_channels=3)
model(x)
