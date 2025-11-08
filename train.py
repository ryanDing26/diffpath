import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Import your modules
from wsi_tile_dataset import TissueH5Dataset
from model import Diffusion, ConditionalUNet, SimpleUNet

# ================== TRAINING ==================

class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def train_diffusion(
    model,
    dataloader,
    diffusion,
    epochs=100,
    lr=1e-4,
    device='cuda',
    save_every=10,
    model_name='diffusion_model',
    use_ema=True,
    conditional=False,
    cfg_prob=0.1  # Probability of dropping conditioning for CFG training
):
    """
    Improved training loop with EMA and CFG support
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()
    model = model.to(device)
    
    # EMA for stable generation
    ema = EMA(model) if use_ema else None
    
    # Create output directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        losses = []
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            batch_size = images.shape[0]
            
            # Normalize images to [-1, 1]
            images = images * 2.0 - 1.0
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Add noise
            x_t, noise = diffusion.noise_images(images, t)
            
            # Predict noise with optional conditioning
            if conditional and 'metadata' in batch:
                metadata = batch['metadata'].to(device)
                
                # Classifier-free guidance training: randomly drop conditioning
                if np.random.random() < cfg_prob:
                    # Use zero conditioning for CFG training
                    metadata = torch.zeros_like(metadata)
                
                predicted_noise = model(x_t, t, conditioning=metadata)
            else:
                predicted_noise = model(x_t, t)
            
            # Calculate loss
            loss = mse(noise, predicted_noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Update EMA
            if ema is not None:
                ema.update()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses[-100:]), 'lr': scheduler.get_last_lr()[0]})
        
        # Step scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1} - Loss: {np.mean(losses):.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': np.mean(losses),
            }
            if ema is not None:
                checkpoint['ema_shadow'] = ema.shadow
            
            torch.save(checkpoint, f'checkpoints/{model_name}_epoch_{epoch+1}.pt')
            
            # Generate samples
            model.eval()
            if ema is not None:
                ema.apply_shadow()
            
            with torch.no_grad():
                # Generate with different conditioning if available
                if conditional and len(dataloader.dataset) > 0:
                    # Get some sample metadata for generation
                    sample_batch = next(iter(dataloader))
                    sample_metadata = sample_batch['metadata'][:4].to(device)
                    samples = diffusion.sample(model, 4, conditioning=sample_metadata)
                else:
                    samples = diffusion.sample(model, 4)
                
                # Denormalize from [-1, 1] to [0, 1]
                samples = (samples.clamp(-1, 1) + 1) / 2
                save_images(samples, f'samples/samples_epoch_{epoch+1}.png')
            
            if ema is not None:
                ema.restore()
            model.train()
    
    return model

def save_images(tensors, path, nrow=2):
    """Save tensor images as grid"""
    from torchvision.utils import make_grid
    
    grid = make_grid(tensors, nrow=nrow, padding=2, normalize=False)
    img = T.ToPILImage()(grid.cpu())
    img.save(path)
    print(f"Saved samples to {path}")

def visualize_dataset(dataloader, num_samples=8):
    """Visualize some samples from the dataset"""
    batch = next(iter(dataloader))
    images = batch['image'][:num_samples]
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, img in enumerate(images):
        if i >= 8:
            break
        img_np = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.close()
    print("Saved dataset samples to dataset_samples.png")

def main():
    parser = argparse.ArgumentParser(description='DiffPath - Tissue Aging Diffusion Model')
    
    # Mode
    parser.add_argument('--mode', choices=['train', 'sample', 'test'], default='train',
                       help='Mode to run: train, sample, or test')
    
    # Data
    parser.add_argument('--csv_path', type=str, default='gtex_features.csv',
                       help='Path to CSV with metadata')
    parser.add_argument('--tissue', type=str, nargs='+', default=['Ovary'],
                       help='Filter for specific tissue types')
    
    # Model
    parser.add_argument('--conditional', action='store_true',
                       help='Use conditional model with metadata')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base channel size for UNet')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='cosine',
                       help='Noise schedule type')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA for model weights')
    
    # Generation
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                       help='Classifier-free guidance scale')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize diffusion
    diffusion = Diffusion(
        timesteps=args.timesteps, 
        device=device,
        schedule=args.schedule
    )
    
    if args.mode == 'train':
        # Setup data
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            # Note: normalization to [-1, 1] happens in training loop
        ])
        
        # Load dataset
        try:
            dataset = TissueH5Dataset(
                csv_path=args.csv_path,
                transform=transform,
                tissue_filter=args.tissue,
                cache_h5=False  # Set to True if you have enough RAM
            )
            print(f"Loaded dataset with {len(dataset)} tiles")
            
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True
            )
            
            # Visualize some samples
            visualize_dataset(dataloader)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        # Initialize model
        if args.conditional:
            # Get conditioning dimension from dataset
            sample = dataset[0]
            cond_dim = sample['metadata'].shape[0]
            print(f"Using conditional model with {cond_dim}-dim metadata")
            model = ConditionalUNet(
                cond_dim=cond_dim,
                base_ch=args.base_channels
            )
        else:
            print("Using unconditional model")
            model = SimpleUNet(base_ch=args.base_channels)
        
        # Load checkpoint if provided
        start_epoch = 0
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Train
        model = train_diffusion(
            model, 
            dataloader, 
            diffusion,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_every=args.save_every,
            model_name='diffpath',
            use_ema=args.use_ema,
            conditional=args.conditional
        )
        
        # Save final model
        torch.save(model.state_dict(), 'checkpoints/diffpath_final.pt')
        print("Training completed!")
        
    elif args.mode == 'sample':
        if not args.checkpoint:
            print("Please provide a checkpoint path with --checkpoint")
            return
        
        # Load model
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        if args.conditional:
            # Need to determine cond_dim - load dataset briefly
            dataset = TissueH5Dataset(
                csv_path=args.csv_path,
                tissue_filter=args.tissue
            )
            cond_dim = dataset[0]['metadata'].shape[0]
            model = ConditionalUNet(cond_dim=cond_dim, base_ch=args.base_channels)
        else:
            model = SimpleUNet(base_ch=args.base_channels)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Generate samples
        os.makedirs('generated', exist_ok=True)
        
        with torch.no_grad():
            if args.conditional:
                # Generate with different age conditions
                # You'll need to construct appropriate metadata vectors here
                print("Generating conditional samples...")
                # This is a placeholder - you need to create proper conditioning
                conditioning = torch.randn(args.num_samples, cond_dim).to(device)
                samples = diffusion.sample(
                    model, 
                    args.num_samples,
                    conditioning=conditioning,
                    cfg_scale=args.cfg_scale
                )
            else:
                print("Generating unconditional samples...")
                samples = diffusion.sample(model, args.num_samples)
            
            # Denormalize and save
            samples = (samples.clamp(-1, 1) + 1) / 2
            
            for i, sample in enumerate(samples):
                img = T.ToPILImage()(sample.cpu())
                img.save(f'generated/sample_{i:04d}.png')
            
            # Save grid
            save_images(samples[:16], 'generated/sample_grid.png', nrow=4)
        
        print(f"Generated {args.num_samples} samples in 'generated/' directory")
        
    elif args.mode == 'test':
        # Quick functionality test
        print("Running test mode...")
        
        # Test with dummy data
        model = SimpleUNet(base_ch=32).to(device)
        dummy_images = torch.randn(2, 3, 256, 256).to(device)
        t = torch.randint(0, diffusion.timesteps, (2,), device=device).long()
        
        # Test forward pass
        with torch.no_grad():
            output = model(dummy_images, t)
            print(f"Model output shape: {output.shape}")
        
        # Test sampling
        samples = diffusion.sample(model, 2)
        print(f"Generated samples shape: {samples.shape}")
        
        print("Test completed successfully!")

if __name__ == '__main__':
    main()