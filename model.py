import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Sinusoidal Time Embeddings
# -----------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.up = up
        if up:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.upsample = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.downsample = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t):
        h = self.relu(self.bnorm1(self.conv1(x)))
        t_emb = self.relu(self.time_mlp(t))[(...,) + (None,) * 2]
        h = h + t_emb
        h = self.relu(self.bnorm2(self.conv2(h)))

        if self.up:
            return self.upsample(h)
        else:
            return self.downsample(h)

# -----------------------------
# Conditional UNet
# -----------------------------
class ConditionalUNet(nn.Module):
    def __init__(self, image_channels=3, time_dim=256, cond_dim=32, base_ch=64):
        super().__init__()

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Conditional embedding
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial conv
        self.conv0 = nn.Conv2d(image_channels, base_ch, 3, padding=1)

        # Downsampling blocks
        self.downs = nn.ModuleList([
            ResidualBlock(base_ch, base_ch*2, time_dim),
            ResidualBlock(base_ch*2, base_ch*4, time_dim),
            ResidualBlock(base_ch*4, base_ch*4, time_dim)
        ])

        # Upsampling blocks
        # Channels must match concatenated skip connections
        self.ups = nn.ModuleList([
            ResidualBlock(base_ch*4 + base_ch*4, base_ch*2, time_dim, up=True),
            ResidualBlock(base_ch*2 + base_ch*2, base_ch, time_dim, up=True),
            ResidualBlock(base_ch + base_ch, base_ch, time_dim, up=True)
        ])

        # Output conv
        self.output = nn.Conv2d(base_ch, image_channels, 1)

    def forward(self, x, timestep, conditioning=None):
        # Embed time
        t = self.time_mlp(timestep)

        # Add conditional embedding
        if conditioning is not None:
            t = t + self.cond_proj(conditioning)

        x = self.conv0(x)
        residuals = []

        # Down path
        for down in self.downs:
            residuals.append(x)
            x = down(x, t)

        # Up path
        for up, res in zip(self.ups, reversed(residuals)):
            x = F.interpolate(x, size=res.shape[-2:], mode='nearest')  # match H/W
            x = torch.cat([x, res], dim=1)
            x = up(x, t)

        return self.output(x)

class Diffusion:
    """
    DDPM / Conditional DDPM diffusion process.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.timesteps = timesteps
        self.device = device
        self.img_size = img_size

        # Linear beta schedule
        self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        """
        Apply forward diffusion q(x_t | x_0) to input images x at timestep t
        Returns: noised images and the actual noise
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise

    @torch.no_grad()
    def sample(self, model, n, conditioning=None, clip_denoised=True):
        """
        Sample images from the learned model p(x_{t-1} | x_t)
        model: ConditionalUNet
        n: batch size
        conditioning: optional metadata tensor [n, cond_dim]
        """
        model.eval()
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, dtype=torch.long, device=self.device)

            # Predict noise
            predicted_noise = model(x, t, conditioning=conditioning)

            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

            if clip_denoised:
                x = x.clamp(-1., 1.)  # Optional: keep pixel values in range

        model.train()
        return x

from torch.utils.data import DataLoader
from wsi_tile_dataset import TissueH5Dataset
from torchvision import transforms

# Load dataset (Ovary only)
transform = transforms.Compose([transforms.ToTensor()])
dataset = TissueH5Dataset(csv_path="gtex_features.csv", transform=transform, tissue_filter=["Ovary"])
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

# Initialize model and diffusion
cond_dim = dataset[0]["metadata"].shape[0]
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConditionalUNet(image_channels=3, time_dim=256, cond_dim=cond_dim).to(device)
diffusion = Diffusion(timesteps=1000, img_size=256, device=device)

# Example: one batch training loop
for batch in loader:
    x = batch["image"].to(device)          # [B, 3, 256, 256]
    cond = batch["metadata"].to(device)    # [B, cond_dim]
    t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device)

    # Forward diffusion
    x_noisy, noise = diffusion.noise_images(x, t)

    # Predict noise
    predicted_noise = model(x_noisy, t, conditioning=cond)

    # Loss (MSE between true noise and predicted)
    loss = nn.MSELoss()(predicted_noise, noise)
    loss.backward()
    break