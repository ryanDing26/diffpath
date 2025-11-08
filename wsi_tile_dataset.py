import h5py
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F

class TissueH5Dataset(Dataset):
    """
    H5 tissue tile dataset with categorical metadata concatenation.
    Lazy loading designed for very large datasets (30M+ tiles).
    Each sample = {'image': image_tensor, 'metadata': concatenated_vector}
    Supports filtering by tissue type.
    """

    def __init__(self, csv_path, transform=None, cache_h5=False, tissue_filter: list = None):
        """
        Args:
            csv_path (str): Path to CSV with metadata (paths + categorical fields)
            transform (callable, optional): torchvision transform to apply to each image
            cache_h5 (bool): whether to keep H5 files open in memory for repeated access
            tissue_filter (list[str], optional): Only load tiles for these tissue types
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.cache_h5 = cache_h5

        # --- Apply tissue filter if provided ---
        if tissue_filter is not None:
            tissue_filter_lower = [t.strip().lower() for t in tissue_filter]
            self.df = self.df[self.df["Tissue"].str.strip().str.lower().isin(tissue_filter_lower)]
            self.df = self.df.reset_index(drop=True)

        # --- Filter valid files ---
        self.valid_rows = []
        for i, h5_path in enumerate(self.df["Tissue Sample ID"].values):
            try:
                with h5py.File(h5_path, "r") as f:
                    if "tiles" in f:
                        num_tiles = len(f["tiles"])
                        self.valid_rows.append((i, h5_path, num_tiles))
            except Exception as e:
                print(f"⚠️ Skipping missing/unreadable file: {h5_path} ({e})")

        if not self.valid_rows:
            raise ValueError("No valid H5 files found after applying tissue filter.")

        # --- Metadata ---
        self.tissue = self.df["Tissue"].astype("category")
        self.gender = self.df["Sex"].astype("category")
        self.age = self.df["Age Bracket"].astype("category")
        self.hardy = self.df["Hardy Scale"].astype("category")

        self.num_tissue = len(self.tissue.cat.categories)
        self.num_gender = len(self.gender.cat.categories)
        self.num_age = len(self.age.cat.categories)
        self.num_hardy = len(self.hardy.cat.categories)

        # --- Build global tile index ---
        self.tiles = []  # each item = (csv_row_idx, tile_idx, h5_path)
        for row_idx, h5_path, num_tiles in self.valid_rows:
            for t in range(num_tiles):
                self.tiles.append((row_idx, t, h5_path))

        print(f"✅ Loaded {len(self.tiles)} total tiles from {len(self.valid_rows)} H5 files")

        # --- H5 cache dict ---
        self._h5_cache = {} if self.cache_h5 else None

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        row_idx, tile_idx, h5_path = self.tiles[idx]

        # --- Load tile lazily ---
        if self.cache_h5:
            if h5_path not in self._h5_cache:
                self._h5_cache[h5_path] = h5py.File(h5_path, "r")
            f = self._h5_cache[h5_path]
            tile = f["tiles"][tile_idx][...]
        else:
            with h5py.File(h5_path, "r") as f:
                tile = f["tiles"][tile_idx][...]

        if tile.max() <= 1:
            tile = (tile * 255).astype(np.uint8)

        img = Image.fromarray(tile)
        if self.transform:
            img = self.transform(img)

        # --- One-hot encode metadata ---
        t_tissue = F.one_hot(torch.tensor(self.tissue.cat.codes[row_idx], dtype=torch.long), num_classes=self.num_tissue)
        t_gender = F.one_hot(torch.tensor(self.gender.cat.codes[row_idx], dtype=torch.long), num_classes=self.num_gender)
        t_age = F.one_hot(torch.tensor(self.age.cat.codes[row_idx], dtype=torch.long), num_classes=self.num_age)
        t_hardy = F.one_hot(torch.tensor(self.hardy.cat.codes[row_idx], dtype=torch.long), num_classes=self.num_hardy)

        metadata = torch.cat([t_tissue, t_gender, t_age, t_hardy], dim=0).float()

        return {"image": img, "metadata": metadata}

    def close_cache(self):
        """Close all cached H5 files if cache_h5=True"""
        if self._h5_cache:
            for f in self._h5_cache.values():
                f.close()
            self._h5_cache = {}

    
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Only load ovaries
dataset = TissueH5Dataset(
    csv_path="gtex_features.csv",
    transform=transform,
    cache_h5=True,
    tissue_filter=["Ovary"]
)

print(f"Number of tiles: {len(dataset)}")
sample = dataset[0]
print("Image shape:", sample["image"].shape)
print("Metadata shape:", sample["metadata"].shape)
# use num_workers=0 if cache_h5=True, 4 if False
loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
for batch in loader:
    print(batch["image"].shape, batch["metadata"].shape)
    break

dataset.close_cache()
