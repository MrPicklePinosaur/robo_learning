import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import logging
from datetime import datetime

from dataset import AnimeFaces
from unet import UNet

# TODO try other noise schedules
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# Hyperparameters
IMG_SIZE=64
BATCH_SIZE=256
T=300
EPOCHS=100
LEARING_RATE=1e-6

# For use when saving model
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

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

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    # print(vals.shape, t.shape)
    out = vals.gather(-1, t.squeeze(1).cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# x is a batch of images, t is the timestep of each image to sample at
def sample_noisy_image(x, t):
    noise = torch.randn_like(x)
    sqrt_alphas_overline_t = get_index_from_list(torch.sqrt(alphas_overline), t, x.shape)
    sqrt_one_minus_alphas_overline_t = get_index_from_list(torch.sqrt(1.-alphas_overline), t, x.shape)
    # print(A.shape, A[t].view(BATCH_SIZE,1,1,1).shape, x.shape)
    # return A[t].view(BATCH_SIZE,1,1,1) * x + B[t].view(BATCH_SIZE,1,1,1) * noise, noise
    return sqrt_alphas_overline_t.to(device) * x.to(device) + sqrt_one_minus_alphas_overline_t.to(device) * noise.to(device), noise.to(device)

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

def inference(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)

    model = UNet(in_channels=3, out_channels=3, time_dim=BATCH_SIZE)
    model.load_state_dict(checkpoint['weights'])

    image_count = 10 # number of images to show in result
    fig_index = 1
    sample_indices = [int(T/image_count) * (i+1) - 1 for i in range(image_count)]
    print(sample_indices)

    plt.figure()

    with torch.no_grad():
        # generate an image
        x_t = torch.randn((1, 3, IMG_SIZE, IMG_SIZE))
        for t in range(T-1, 1, -1):
            z = 0
            if t > 1:
                z  = torch.randn((1, 3, IMG_SIZE, IMG_SIZE))
            
            t = torch.tensor([[t]])
            predict = model(x_t, t)
            # print('predict', predict.shape)
            sigma_t = torch.sqrt(betas[t].view(1,1,1,1)) # TODO get the variance
            # print('alphas', t.shape, alphas[t].view(1,1,1,1).shape)
            x_t = 1/torch.sqrt(alphas[t].view(1,1,1,1)) * (x_t - (1-alphas[t].view(1,1,1,1))/torch.sqrt(1-alphas_overline[t].view(1,1,1,1)) * predict + sigma_t * z)
            # print('x_t', x_t.shape)

            # TODO this is pretty stupid
            if t in sample_indices:
                print(t)
                plt.subplot(1, image_count+1, fig_index)
                # TODO make reverse_transformations work with batch
                plt.imshow(reverse_transformations(x_t.squeeze(0)))
                fig_index += 1

    # show image
    plt.show()

if __name__ == "__main__":
    # Setup
    device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # setup basic logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Determine mean and variance of pixel values
    transformations = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),                 # scale to [0, 1]
        transforms.Lambda(lambda t: (t*2) - 1) # scale to [-1, 1]
    ]
    preprocess = transforms.Compose(transformations)

    dataset = AnimeFaces(img_dir='data', preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    RECOMPUTE_NORMS=False
    # computed previously
    norm_mean, norm_std = [25.66350674, 12.87127401, 11.44491336], [202.07254913, 101.43154071, 90.1956577]
    if RECOMPUTE_NORMS:
        norm_mean, norm_std = normalize_dataset(dataloader, IMG_SIZE, IMG_SIZE)

    print('norms computed', norm_mean, norm_std)

    # Apply the normalization transformations
    # transformations.append(transforms.Normalize(mean=norm_mean, std=norm_std))
    preprocess = transforms.Compose(transformations)
    dataset = AnimeFaces(img_dir='data', preprocess=preprocess)
    scaled_len = len(dataset)//BATCH_SIZE * BATCH_SIZE
    # dataset = Subset(dataset, range(scaled_len)) # truncate to multiple of batch size
    dataset = Subset(dataset, range(BATCH_SIZE*4)) # truncate to multiple of batch size
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

    model = UNet(in_channels=3, out_channels=3, time_dim=BATCH_SIZE)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('Total parameters:', total_params)
    # x = torch.randn(1, 3, 572, 572)
    # timestep = torch.zeros((1, 1))
    # model(x, timestep)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARING_RATE)
    criterion = nn.MSELoss()

    # Training process
    for epoch in range(EPOCHS):

        total_loss = 0
        for i, batch in enumerate(dataloader):
            model.train()

            optimizer.zero_grad()

            # sample random timestep
            t = torch.randint(0, T, (BATCH_SIZE, 1), device=device)

            # print('batch shape', batch.shape)
            e, noise = sample_noisy_image(batch, t)
            #e = torch.randint(0, 1, (BATCH_SIZE, 3, 256, 256), device=device).float()

            outputs = model(e, t)
            loss = criterion(outputs, noise)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f'finished iteration {i+1}/{len(dataloader)}')

        # TODO log some progress and visualize
        print(f'Completed Epoch {epoch}, Loss: {total_loss/len(dataloader)}')

        # save model
        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'norm_mean': norm_mean,
            'norm_std': norm_std,
        }
        torch.save(checkpoint, f'checkpoints/model_{timestamp}.pth')