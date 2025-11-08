
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
from wsi_tile_dataset import TissueH5Dataset
from model import Diffusion, ConditionalUNet, SimpleUNet

# ================== TRAINING ==================

def train_diffusion(
    model,
    dataloader,
    diffusion,
    epochs=100,
    lr=1e-4,
    device='cuda',
    save_every=10,
    model_name='diffusion_model'
):
    """
    Simple training loop for diffusion model
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model = model.to(device)
    
    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        losses = []
        
        for batch in pbar:
            images = batch['image'].to(device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, diffusion.timesteps, (batch_size,)).long().to(device)
            
            # Add noise
            x_t, noise = diffusion.noise_images(images, t)
            
            # Predict noise
            if isinstance(model, ConditionalUNet):
                # Add conditioning if using conditional model
                # This is a placeholder - implement based on your metadata
                predicted_noise = model(x_t, t, conditioning=None)
            else:
                predicted_noise = model(x_t, t)
            
            # Calculate loss
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            pbar.set_postfix({'loss': np.mean(losses[-100:])})
        
        print(f"Epoch {epoch} - Loss: {np.mean(losses):.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': np.mean(losses),
            }, f'{model_name}_epoch_{epoch}.pt')
            
            # Generate samples
            model.eval()
            samples = diffusion.sample(model, 4)
            samples = (samples.clamp(-1, 1) + 1) / 2
            save_images(samples, f'samples_epoch_{epoch}.png')
            model.train()


def save_images(tensors, path, nrow=2):
    """Save tensor images as grid"""
    from torchvision.utils import make_grid
    import torchvision.transforms as T
    
    grid = make_grid(tensors, nrow=nrow)
    img = T.ToPILImage()(grid)
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description='DiffPath - Clean Tissue Diffusion')
    
    # Mode
    parser.add_argument('mode', choices=['train', 'sample'], 
                       help='Mode to run: train or sample')
    
    # Data
    parser.add_argument('--h5_dir', type=str, default='/shares/sinha/sinha_common/GTEx/h5-tiles-256/',
                       help='Directory with H5 files')
    parser.add_argument('--organ', choices=[''],
                       help='Filter for specific organ')
    
    # Model
    parser.add_argument('--conditional', action='store_true',
                       help='Use conditional model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    
    # Generation
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize diffusion
    diffusion = Diffusion(timesteps=args.timesteps, device=device)
    
    # Initialize model
    if args.conditional:
        model = ConditionalUNet()
    else:
        model = SimpleUNet()
    
    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {args.checkpoint}")
    
    if args.mode == 'train':
        # Setup data
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Get H5 files
        h5_files = list(Path(args.h5_dir).glob('*.h5'))
        dataset = TissueH5Dataset(h5_files, transform=transform, organ_filter=args.organ)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        # Train
        train_diffusion(
            model, dataloader, diffusion,
            epochs=args.epochs, lr=args.lr, device=device
        )
        
    elif args.mode == 'sample':
        # Generate samples
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            samples = diffusion.sample(model, args.num_samples)
            samples = (samples.clamp(-1, 1) + 1) / 2
            
            # Save individual samples
            for i, sample in enumerate(samples):
                img = T.ToPILImage()(sample)
                img.save(f'generated_{i}.png')
            
            # Save grid
            save_images(samples, 'generated_grid.png', nrow=4)
        
        print(f"Generated {args.num_samples} samples")
        
    elif args.mode == 'test':
        # Quick test with dummy data
        print("Running test mode...")
        
        # Create dummy data
        dummy_images = torch.randn(4, 3, 256, 256).to(device)
        t = torch.randint(0, diffusion.timesteps, (4,)).long().to(device)
        
        # Test forward pass
        model = model.to(device)
        with torch.no_grad():
            output = model(dummy_images, t)
            print(f"Model output shape: {output.shape}")
        
        # Test sampling
        samples = diffusion.sample(model, 2)
        print(f"Generated samples shape: {samples.shape}")
        
        print("Test completed successfully!")


if __name__ == '__main__':
    main()