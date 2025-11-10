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

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

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
    save_every=1,
    model_name='diffusion_model',
    use_ema=True,
    conditional=False,
    cfg_prob=0.1,
    mixed_precision=False,
    accumulation_steps=1,
    early_stop_patience=0
):
    """
    Memory-optimized training loop with:
    - Mixed precision (FP16)
    - Gradient accumulation
    - EMA
    - Gradient checkpointing (configured in model)
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse = nn.MSELoss()
    model = model.to(device)
    best_loss = float('inf')
    patience_counter = 0
    # Mixed precision scaler
    scaler = GradScaler() if mixed_precision else None
    
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
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            batch_size = images.shape[0]
            
            # Normalize images to [-1, 1]
            images = images * 2.0 - 1.0
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
            
            # Add noise
            x_t, noise = diffusion.noise_images(images, t)
            
            # Forward pass with optional mixed precision
            if mixed_precision:
                with autocast():
                    # Predict noise with optional conditioning
                    if conditional and 'metadata' in batch:
                        metadata = batch['metadata'].to(device)
                        
                        # Classifier-free guidance training
                        if np.random.random() < cfg_prob:
                            metadata = torch.zeros_like(metadata)
                        
                        predicted_noise = model(x_t, t, conditioning=metadata)
                    else:
                        predicted_noise = model(x_t, t)
                    
                    # Calculate loss
                    loss = mse(noise, predicted_noise)
                    loss = loss / accumulation_steps
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
            else:
                # Standard forward pass
                if conditional and 'metadata' in batch:
                    metadata = batch['metadata'].to(device)
                    if np.random.random() < cfg_prob:
                        metadata = torch.zeros_like(metadata)
                    predicted_noise = model(x_t, t, conditioning=metadata)
                else:
                    predicted_noise = model(x_t, t)
                
                loss = mse(noise, predicted_noise)
                loss = loss / accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if mixed_precision:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    # Step optimizer with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                optimizer.zero_grad()
                
                # Update EMA
                if ema is not None:
                    ema.update()
            
            losses.append(loss.item() * accumulation_steps)
            pbar.set_postfix({
                'loss': np.mean(losses[-100:]),
                'lr': scheduler.get_last_lr()[0],
                'mem': f"{torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
            })
        
        # Step scheduler
        scheduler.step()
        
        print(f"Epoch {epoch+1} - Loss: {np.mean(losses):.6f}")

        # Early stopping check
        epoch_loss = np.mean(losses)
        if early_stop_patience > 0:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stop_patience} epochs)")
                    break

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
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
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
        if 'age_bracket' in batch:
            axes[i].set_title(f"Age: {batch['age_bracket'][i]}", fontsize=8)
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png')
    plt.close()
    print("Saved dataset samples to dataset_samples.png")

def main():
    parser = argparse.ArgumentParser(description='DiffPath - Memory Optimized Training')
    
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
    parser.add_argument('--use_checkpoint', action='store_true',
                       help='Use gradient checkpointing (saves memory)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (effective batch = batch_size * accumulation_steps)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--schedule', choices=['linear', 'cosine'], default='cosine',
                       help='Noise schedule type')
    parser.add_argument('--save_every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--use_ema', action='store_true', default=True,
                       help='Use EMA for model weights')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16) - saves memory')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                   help='Stop if loss does not improve for N epochs (0=disabled)')
    
    # Generation
    parser.add_argument('--num_samples', type=int, default=16,
                       help='Number of samples to generate')
    parser.add_argument('--cfg_scale', type=float, default=3.0,
                       help='Classifier-free guidance scale')
    
    # Memory optimization
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
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
        ])
        
        # Load dataset
        try:
            dataset = TissueH5Dataset(
                csv_path=args.csv_path,
                transform=transform,
                tissue_filter=args.tissue,
                cache_h5=False  # Don't cache to save memory
            )
            print(f"Loaded dataset with {len(dataset)} tiles")
            
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=True if args.num_workers > 0 else False
            )
            
            # Visualize some samples
            visualize_dataset(dataloader)
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Initialize model
        if args.conditional:
            sample = dataset[0]
            cond_dim = sample['metadata'].shape[0]
            print(f"Using conditional model with {cond_dim}-dim metadata")
            model = ConditionalUNet(
                cond_dim=cond_dim,
                base_ch=args.base_channels,
                use_checkpoint=args.use_checkpoint
            )
        else:
            print("Using unconditional model")
            model = SimpleUNet(
                base_ch=args.base_channels,
                use_checkpoint=args.use_checkpoint
            )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Load checkpoint if provided
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        
        # Train
        print(f"\nTraining configuration:")
        print(f"  Base channels: {args.base_channels}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Accumulation steps: {args.accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
        print(f"  Gradient checkpointing: {args.use_checkpoint}")
        print(f"  Mixed precision: {args.mixed_precision}")
        print(f"  EMA: {args.use_ema}")
        
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
            conditional=args.conditional,
            mixed_precision=args.mixed_precision,
            accumulation_steps=args.accumulation_steps,
            early_stop_patience=args.early_stop_patience
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
            dataset = TissueH5Dataset(
                csv_path=args.csv_path,
                tissue_filter=args.tissue
            )
            cond_dim = dataset[0]['metadata'].shape[0]
            model = ConditionalUNet(
                cond_dim=cond_dim,
                base_ch=args.base_channels,
                use_checkpoint=args.use_checkpoint
            )
        else:
            model = SimpleUNet(
                base_ch=args.base_channels,
                use_checkpoint=args.use_checkpoint
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Generate samples
        os.makedirs('generated', exist_ok=True)
        
        with torch.no_grad():
            if args.conditional:
                print("Generating conditional samples...")
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
        # Quick functionality test with minimal memory
        print("Running test mode with minimal memory...")
        
        # Use very small model for testing
        print(f"Base channels: {args.base_channels}")
        print(f"Gradient checkpointing: {args.use_checkpoint}")
        print(f"Mixed precision: {args.mixed_precision}")
        
        model = SimpleUNet(
            base_ch=args.base_channels,
            use_checkpoint=args.use_checkpoint
        ).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Test with very small batch
        batch_size = 1
        dummy_images = torch.randn(batch_size, 3, 256, 256).to(device)
        t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device).long()
        
        # Test forward pass
        print("\nTesting forward pass...")
        with torch.no_grad():
            if args.mixed_precision:
                with autocast():
                    output = model(dummy_images, t)
            else:
                output = model(dummy_images, t)
            print(f"✓ Model output shape: {output.shape}")
        
        if torch.cuda.is_available():
            print(f"✓ Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Test sampling
        print("\nTesting sampling (2 samples)...")
        samples = diffusion.sample(model, 2)
        print(f"✓ Generated samples shape: {samples.shape}")
        
        print("\n✅ Test completed successfully!")

if __name__ == '__main__':
    main()