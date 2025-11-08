import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
import os
from pathlib import Path

class TissueH5Dataset(Dataset):
    """
    Improved H5 tissue tile dataset with better error handling and metadata encoding.
    """
    
    def __init__(self, csv_path, transform=None, cache_h5=False, 
                 tissue_filter=None, verbose=True):
        """
        Args:
            csv_path (str): Path to CSV with metadata
            transform (callable): torchvision transform for images
            cache_h5 (bool): Keep H5 files open (uses more memory but faster)
            tissue_filter (list[str]): Only load tiles for these tissue types
            verbose (bool): Print loading information
        """
        self.transform = transform
        self.cache_h5 = cache_h5
        self.verbose = verbose
        
        # Load and process CSV
        self.df = pd.read_csv(csv_path)
        
        # Apply tissue filter if provided
        if tissue_filter is not None:
            tissue_filter_lower = [t.strip().lower() for t in tissue_filter]
            mask = self.df["Tissue"].str.strip().str.lower().isin(tissue_filter_lower)
            self.df = self.df[mask].reset_index(drop=True)
            if self.verbose:
                print(f"Filtered to tissues: {tissue_filter}")
                print(f"Samples after filtering: {len(self.df)}")
        
        # Validate and filter H5 files
        self.valid_rows = []
        self.tiles = []
        
        for idx, row in self.df.iterrows():
            h5_path = row["Tissue Sample ID"]
            
            # Check if file exists
            if not os.path.exists(h5_path):
                if self.verbose:
                    print(f"⚠️ Missing file: {h5_path}")
                continue
            
            try:
                with h5py.File(h5_path, "r") as f:
                    if "tiles" in f:
                        num_tiles = len(f["tiles"])
                        if num_tiles > 0:
                            self.valid_rows.append(idx)
                            
                            # Add tiles to global index
                            for t in range(num_tiles):
                                self.tiles.append((idx, t, h5_path))
                    else:
                        if self.verbose:
                            print(f"⚠️ No 'tiles' key in {h5_path}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error reading {h5_path}: {e}")
        
        if not self.tiles:
            raise ValueError("No valid tiles found! Check your data paths and CSV.")
        
        # Reset dataframe to only valid rows
        self.df = self.df.iloc[self.valid_rows].reset_index(drop=True)
        
        # Process metadata categories
        self._setup_metadata()
        
        # H5 cache
        self._h5_cache = {} if self.cache_h5 else None
        
        if self.verbose:
            print(f"✅ Dataset initialized:")
            print(f"   - Total tiles: {len(self.tiles)}")
            print(f"   - Valid slides: {len(self.valid_rows)}")
            print(f"   - Metadata dim: {self.metadata_dim}")
    
    def _setup_metadata(self):
        """Setup metadata encoding"""
        # Define metadata columns and handle missing values
        self.metadata_columns = {
            'tissue': 'Tissue',
            'sex': 'Sex', 
            'age': 'Age Bracket',
            'hardy': 'Hardy Scale'
        }
        
        self.encoders = {}
        self.metadata_dims = []
        
        for key, col in self.metadata_columns.items():
            if col in self.df.columns:
                # Handle missing values
                self.df[col] = self.df[col].fillna('Unknown')
                # Convert to categorical
                self.df[col] = self.df[col].astype('category')
                # Store categories for encoding
                self.encoders[key] = self.df[col].cat.categories
                self.metadata_dims.append(len(self.encoders[key]))
                if self.verbose:
                    print(f"   - {key}: {len(self.encoders[key])} categories")
        
        self.metadata_dim = sum(self.metadata_dims)
    
    def encode_metadata(self, row_idx):
        """Encode metadata for a given row as one-hot vector"""
        vectors = []
        df_row = self.df.iloc[row_idx]
        
        for key, col in self.metadata_columns.items():
            if col in self.df.columns:
                # Get category code
                value = df_row[col]
                if pd.isna(value):
                    code = 0  # Default to first category for missing values
                else:
                    code = self.df[col].cat.categories.get_loc(value)
                
                # One-hot encode
                one_hot = F.one_hot(
                    torch.tensor(code, dtype=torch.long),
                    num_classes=len(self.encoders[key])
                )
                vectors.append(one_hot)
        
        # Concatenate all metadata vectors
        if vectors:
            metadata = torch.cat(vectors, dim=0).float()
        else:
            metadata = torch.zeros(1).float()  # Fallback
        
        return metadata
    
    def __len__(self):
        return len(self.tiles)
    
    def __getitem__(self, idx):
        row_idx, tile_idx, h5_path = self.tiles[idx]
        
        # Load tile
        try:
            if self.cache_h5:
                if h5_path not in self._h5_cache:
                    self._h5_cache[h5_path] = h5py.File(h5_path, "r")
                f = self._h5_cache[h5_path]
                tile = f["tiles"][tile_idx][...]
            else:
                with h5py.File(h5_path, "r") as f:
                    tile = f["tiles"][tile_idx][...]
            
            # Convert to uint8 if needed
            if tile.dtype != np.uint8:
                if tile.max() <= 1.0:
                    tile = (tile * 255).astype(np.uint8)
                else:
                    tile = tile.astype(np.uint8)
            
            # Convert to PIL Image
            img = Image.fromarray(tile)
            
            # Apply transforms
            if self.transform:
                img = self.transform(img)
            else:
                # Default: convert to tensor
                img = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            
            # Encode metadata
            metadata = self.encode_metadata(row_idx)
            
            # Get additional info for analysis
            age_bracket = self.df.iloc[row_idx]['Age Bracket']
            tissue_type = self.df.iloc[row_idx]['Tissue']
            
            return {
                'image': img,
                'metadata': metadata,
                'age_bracket': age_bracket,
                'tissue_type': tissue_type,
                'slide_id': Path(h5_path).stem
            }
            
        except Exception as e:
            print(f"Error loading tile {idx} from {h5_path}: {e}")
            # Return a dummy sample
            dummy_img = torch.zeros(3, 256, 256)
            dummy_metadata = torch.zeros(self.metadata_dim)
            return {
                'image': dummy_img,
                'metadata': dummy_metadata,
                'age_bracket': 'Unknown',
                'tissue_type': 'Unknown',
                'slide_id': 'error'
            }
    
    def close_cache(self):
        """Close all cached H5 files"""
        if self._h5_cache:
            for f in self._h5_cache.values():
                f.close()
            self._h5_cache = {}
    
    def get_age_distribution(self):
        """Get distribution of age brackets in dataset"""
        return self.df['Age Bracket'].value_counts()
    
    def get_metadata_for_age(self, age_bracket, tissue_type=None):
        """
        Get a metadata vector for specific age bracket and tissue type.
        Useful for conditional generation.
        """
        # Find a matching row
        mask = self.df['Age Bracket'] == age_bracket
        if tissue_type:
            mask &= self.df['Tissue'] == tissue_type
        
        matching_rows = self.df[mask]
        if len(matching_rows) == 0:
            print(f"No samples found for age={age_bracket}, tissue={tissue_type}")
            return None
        
        # Use first matching row
        row_idx = matching_rows.index[0]
        return self.encode_metadata(row_idx)


# Testing code
# if __name__ == "__main__":
#     from torchvision import transforms
#     from torch.utils.data import DataLoader
#     import matplotlib.pyplot as plt
    
#     # Define transforms
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     # Create dataset
#     print("Loading dataset...")
#     dataset = TissueH5Dataset(
#         csv_path="gtex_features.csv",
#         transform=transform,
#         cache_h5=False,
#         tissue_filter=["Ovary"],
#         verbose=True
#     )
    
#     print(f"\nDataset statistics:")
#     print(f"Total tiles: {len(dataset)}")
#     print(f"Metadata dimension: {dataset.metadata_dim}")
#     print(f"\nAge distribution:")
#     print(dataset.get_age_distribution())
    
#     # Test single sample
#     print("\n Testing single sample...")
#     sample = dataset[0]
#     print(f"Image shape: {sample['image'].shape}")
#     print(f"Metadata shape: {sample['metadata'].shape}")
#     print(f"Age bracket: {sample['age_bracket']}")
#     print(f"Tissue type: {sample['tissue_type']}")
    
#     # Test dataloader
#     print("\nTesting dataloader...")
#     loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
#     batch = next(iter(loader))
#     print(f"Batch image shape: {batch['image'].shape}")
#     print(f"Batch metadata shape: {batch['metadata'].shape}")
    
#     # Visualize some samples
#     fig, axes = plt.subplots(2, 4, figsize=(12, 6))
#     axes = axes.ravel()
    
#     for i in range(8):
#         sample = dataset[i]
#         img = sample['image'].permute(1, 2, 0).numpy()
#         axes[i].imshow(img)
#         axes[i].set_title(f"{sample['age_bracket']}", fontsize=8)
#         axes[i].axis('off')
    
#     plt.suptitle("Dataset Samples")
#     plt.tight_layout()
#     plt.savefig("dataset_preview.png")
#     print("\nSaved preview to dataset_preview.png")
    
#     # Clean up
#     dataset.close_cache()