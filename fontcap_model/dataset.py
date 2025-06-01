from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

CHARSET = "abcdefghijklmnopqrstuvwxyz"

class FontcapDataset(Dataset):
    """Dataset wrapper for scraped fonts"""
    def __init__(self, data_root: Path, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.pairs = self._collect_pairs()

    def _collect_pairs(self):
        pairs = []
        for font_dir in self.data_root.iterdir():
            if not font_dir.is_dir():
                continue

            for c in CHARSET:
                upper_path = font_dir / f"{ord(c.upper())}.png"
                lower_path = font_dir / f"{ord(c.lower())}.png"
                if upper_path.exists() and lower_path.exists():
                    pairs.append((lower_path, upper_path))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Pulls images from data directory. unsqueezes them into (1, 32, 32) tensors"""
        lower_path, upper_path = self.pairs[idx]
        lower_img = Image.open(lower_path).convert('L')
        upper_img = Image.open(upper_path).convert('L')
        # Normalize pixel value
        lower_arr = np.array(lower_img, dtype=np.float32) / 255.0
        upper_arr = np.array(upper_img, dtype=np.float32) / 255.0
        lower_tensor = torch.from_numpy(lower_arr).unsqueeze(0)
        upper_tensor = torch.from_numpy(upper_arr).unsqueeze(0)

        if self.transform:
            lower_tensor = self.transform(lower_tensor)
            upper_tensor = self.transform(upper_tensor)

        return lower_tensor, upper_tensor

def get_dataloaders(
    data_root: str | Path,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    if type(data_root) is str:
        data_root = Path(data_root)
    dataset = FontcapDataset(data_root) # type: ignore
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    torch.manual_seed(seed)
    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
