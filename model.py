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
# Improved Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.GroupNorm(8, out_ch)  # GroupNorm is more stable than BatchNorm
        self.bnorm2 = nn.GroupNorm(8, out_ch)
        self.relu = nn.SiLU()  # SiLU works better for diffusion models
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = x
        h = self.relu(self.bnorm1(self.conv1(h)))
        
        # Time conditioning with scale and shift
        t_emb = self.time_mlp(t)
        t_emb = t_emb[:, :, None, None]
        scale, shift = t_emb.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.relu(self.bnorm2(self.conv2(h)))
        h = self.dropout(h)
        
        return h + self.residual_conv(x)

# -----------------------------
# Attention Block (optional but recommended)
# -----------------------------
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = q.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        k = k.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        v = v.view(b, self.num_heads, c // self.num_heads, h * w).transpose(2, 3)
        
        # Scaled dot-product attention
        scale = (c // self.num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(2, 3).reshape(b, c, h, w)
        out = self.proj(out)
        
        return out + x

# -----------------------------
# Down/Up Blocks with proper structure
# -----------------------------
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)
        
    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        skip = x
        x = self.downsample(x)
        return x, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, time_emb_dim)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()
        
    def forward(self, x, skip, t):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t)
        x = self.res2(x, t)
        x = self.attn(x)
        return x

# -----------------------------
# Fixed Conditional UNet
# -----------------------------
class ConditionalUNet(nn.Module):
    def __init__(self, image_channels=3, time_dim=256, cond_dim=32, base_ch=64):
        super().__init__()
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4)
        )
        
        # Conditional embedding with better projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4)
        )
        
        # Initial conv
        self.conv0 = nn.Conv2d(image_channels, base_ch, 3, padding=1)
        
        # Down path with attention at lower resolutions
        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim * 4, use_attn=False)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim * 4, use_attn=True)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, time_dim * 4, use_attn=True)
        
        # Middle block
        self.mid1 = ResidualBlock(base_ch * 8, base_ch * 8, time_dim * 4)
        self.mid_attn = AttentionBlock(base_ch * 8)
        self.mid2 = ResidualBlock(base_ch * 8, base_ch * 8, time_dim * 4)
        
        # Up path
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, time_dim * 4, use_attn=True)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, time_dim * 4, use_attn=True)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch, time_dim * 4, use_attn=False)
        
        # Output layers
        self.final_res = ResidualBlock(base_ch, base_ch, time_dim * 4)
        self.output = nn.Conv2d(base_ch, image_channels, 1)
        
    def forward(self, x, timestep, conditioning=None):
        # Embed time
        t = self.time_mlp(timestep)
        
        # Add conditional embedding if provided
        if conditioning is not None:
            cond_emb = self.cond_proj(conditioning)
            t = t + cond_emb
        
        # Initial conv
        x = self.conv0(x)
        
        # Down path with skip connections
        x, skip1 = self.down1(x, t)
        x, skip2 = self.down2(x, t)
        x, skip3 = self.down3(x, t)
        
        # Middle
        x = self.mid1(x, t)
        x = self.mid_attn(x)
        x = self.mid2(x, t)
        
        # Up path
        x = self.up3(x, skip3, t)
        x = self.up2(x, skip2, t)
        x = self.up1(x, skip1, t)
        
        # Output
        x = self.final_res(x, t)
        return self.output(x)

# -----------------------------
# Simple UNet (non-conditional)
# -----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, image_channels=3, time_dim=256, base_ch=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4)
        )
        
        # Same architecture as ConditionalUNet but without conditioning
        self.conv0 = nn.Conv2d(image_channels, base_ch, 3, padding=1)
        
        self.down1 = DownBlock(base_ch, base_ch * 2, time_dim * 4, use_attn=False)
        self.down2 = DownBlock(base_ch * 2, base_ch * 4, time_dim * 4, use_attn=True)
        self.down3 = DownBlock(base_ch * 4, base_ch * 8, time_dim * 4, use_attn=True)
        
        self.mid1 = ResidualBlock(base_ch * 8, base_ch * 8, time_dim * 4)
        self.mid_attn = AttentionBlock(base_ch * 8)
        self.mid2 = ResidualBlock(base_ch * 8, base_ch * 8, time_dim * 4)
        
        self.up3 = UpBlock(base_ch * 8, base_ch * 8, base_ch * 4, time_dim * 4, use_attn=True)
        self.up2 = UpBlock(base_ch * 4, base_ch * 4, base_ch * 2, time_dim * 4, use_attn=True)
        self.up1 = UpBlock(base_ch * 2, base_ch * 2, base_ch, time_dim * 4, use_attn=False)
        
        self.final_res = ResidualBlock(base_ch, base_ch, time_dim * 4)
        self.output = nn.Conv2d(base_ch, image_channels, 1)
        
    def forward(self, x, timestep):
        t = self.time_mlp(timestep)
        
        x = self.conv0(x)
        
        x, skip1 = self.down1(x, t)
        x, skip2 = self.down2(x, t)
        x, skip3 = self.down3(x, t)
        
        x = self.mid1(x, t)
        x = self.mid_attn(x)
        x = self.mid2(x, t)
        
        x = self.up3(x, skip3, t)
        x = self.up2(x, skip2, t)
        x = self.up1(x, skip1, t)
        
        x = self.final_res(x, t)
        return self.output(x)

# -----------------------------
# Improved Diffusion Class
# -----------------------------
class Diffusion:
    """
    Improved DDPM with variance scheduling options
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, 
                 img_size=256, device="cuda", schedule="linear"):
        self.timesteps = timesteps
        self.device = device
        self.img_size = img_size
        
        # Beta schedule
        if schedule == "linear":
            self.beta = torch.linspace(beta_start, beta_end, timesteps).to(device)
        elif schedule == "cosine":
            # Cosine schedule from improved DDPM paper
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0.0001, 0.9999).to(device)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1], (1, 0), value=1.0)
        
        # Pre-calculate for sampling
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (1. - self.alpha_hat_prev) / (1. - self.alpha_hat)
        
    def noise_images(self, x, t):
        """Apply forward diffusion"""
        sqrt_alpha_hat = self.sqrt_alpha_hat[t][:, None, None, None]
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t][:, None, None, None]
        noise = torch.randn_like(x)
        x_noisy = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return x_noisy, noise
    
    @torch.no_grad()
    def sample(self, model, n, conditioning=None, cfg_scale=3.0):
        """
        Sample with optional classifier-free guidance
        cfg_scale: guidance scale (1.0 = no guidance)
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)
        
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, dtype=torch.long, device=self.device)
            
            # Predict noise
            if isinstance(model, ConditionalUNet) and conditioning is not None:
                # Classifier-free guidance
                if cfg_scale > 1.0:
                    # Unconditional prediction
                    uncond = torch.zeros_like(conditioning)
                    noise_uncond = model(x, t, conditioning=uncond)
                    # Conditional prediction
                    noise_cond = model(x, t, conditioning=conditioning)
                    # Guided prediction
                    predicted_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
                else:
                    predicted_noise = model(x, t, conditioning=conditioning)
            else:
                predicted_noise = model(x, t)
            
            # Denoise step
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            # Mean of the distribution
            mean = self.sqrt_recip_alpha[t][:, None, None, None] * \
                   (x - beta / self.sqrt_one_minus_alpha_hat[t][:, None, None, None] * predicted_noise)
            
            if i > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(self.posterior_variance[t])[:, None, None, None]
                x = mean + variance * noise
            else:
                x = mean
        
        model.train()
        return x