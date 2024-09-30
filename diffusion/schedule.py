import torch

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

# TODO cosine schedule still doesn't look correct
def cosine_schedule(T, s=0.008):
    t = torch.linspace(0, T, T+1)
    alphas_overline = torch.cos((t/T + s)/(1+s) * torch.pi/2) ** 2
    alphas_overline = alphas_overline/alphas_overline[0]
    betas = 1 - (alphas_overline[1:] / alphas_overline[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class Sampler:
    def __init__(self, schedule):
        self.betas = schedule
        self.alphas = 1 - self.betas
        self.alphas_overline = torch.cumprod(self.alphas, axis=0)
        self.A = torch.sqrt(self.alphas_overline)
        self.B = torch.sqrt(1.-self.alphas_overline)

    def _extract(self, tensor, t, x_shape):
        batch_size = t.shape[0]
        sampled_tensor = tensor.gather(-1, t.cpu())
        sampled_tensor = torch.reshape(sampled_tensor, (batch_size,) + (1,) * (len(x_shape) - 1))
        return sampled_tensor.to(t.device)
    
    def sample(self, x, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        A_t = self._extract(self.A, t, x.shape)
        B_t = self._extract(self.B, t, x.shape)
        # TODO double check variance
        return A_t * x + B_t * noise, noise
