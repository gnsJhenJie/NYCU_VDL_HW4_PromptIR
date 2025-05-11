"""Custom dataset for HW4 (paired degraded ↔ clean).
Handles original naming scheme:
    degraded: rain-1.png / snow-123.png
    clean:    rain_clean-1.png / snow_clean-123.png
If your own filenames already match 1‑to‑1, they load seamlessly.
"""
from pathlib import Path
from typing import List, Tuple
import random
import re
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

__all__ = ["HW4PairDataset", "get_train_val_loaders"]

_AUG = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90, expand=False),
])

_TO_TENSOR = T.ToTensor()

# ---------- helper -----------------------------------------------------------

_DEF_PAT = re.compile(r"^(?P<prefix>[a-zA-Z]+)-(?P<idx>.+)$")


def _derive_clean_name(deg_file: Path) -> str:
    """Given Path('rain-7.png') → 'rain_clean-7.png'."""
    m = _DEF_PAT.match(deg_file.stem)
    if m:
        return f"{m.group('prefix')}_clean-{m.group('idx')}{deg_file.suffix}"
    # fallback: same stem
    return deg_file.name

# ---------- dataset ----------------------------------------------------------


class HW4PairDataset(Dataset):
    def __init__(self, root: str, split: str = "train", patch: int = 256):
        assert split in {"train", "val"}
        self.patch = patch
        root = Path(root)
        self.clean_dir = root / split / "clean"
        self.deg_dir = root / split / "degraded"
        assert self.clean_dir.exists() and self.deg_dir.exists(), (
            f"Directory structure not found: {self.clean_dir} / {self.deg_dir}")
        self.pairs: List[Tuple[Path, Path]] = []
        for p in self.deg_dir.glob("*.png"):
            # build possible clean filename(s)
            cand = self.clean_dir / _derive_clean_name(p)
            if cand.exists():
                self.pairs.append((p, cand))
            else:
                # fallback to identical stem (without _clean)
                alt = self.clean_dir / p.name
                if alt.exists():
                    self.pairs.append((p, alt))
        if not self.pairs:
            raise RuntimeError(
                "Dataset is empty — verify that clean & degraded images share matching stems.\n"
                "Expect e.g.  data/train/clean/rain_clean-1.png  and  data/train/degraded/rain-1.png")

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    def _crop(self, arr: np.ndarray) -> np.ndarray:
        h, w, _ = arr.shape
        if h < self.patch or w < self.patch:
            pad_h = max(0, self.patch - h)
            pad_w = max(0, self.patch - w)
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w = arr.shape[:2]
        y = random.randint(0, h - self.patch)
        x = random.randint(0, w - self.patch)
        return arr[y:y+self.patch, x:x+self.patch]

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg = np.array(Image.open(deg_path).convert("RGB"))
        clean = np.array(Image.open(clean_path).convert("RGB"))
        if self.patch > 0:
            deg = self._crop(deg)
            clean = self._crop(clean)
        if self.training:
            cat = np.concatenate([deg, clean], axis=2)
            cat = _AUG(Image.fromarray(cat))
            cat = np.array(cat)
            deg, clean = cat[:, :deg.shape[1], :], cat[:, deg.shape[1]:, :]
        deg = _TO_TENSOR(deg)
        clean = _TO_TENSOR(clean)
        prompt_id = 0 if "rain" in deg_path.stem else 1
        return {"degraded": deg, "clean": clean, "prompt_id": prompt_id}

    training: bool = True

# ---------- loaders ----------------------------------------------------------


def get_train_val_loaders(
        root: str, batch: int, num_workers: int = 8, patch: int = 256):
    full = HW4PairDataset(root, "train", patch)
    n_val = int(0.05 * len(full))
    n_train = len(full) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_set.dataset.training = True
    val_set.dataset.training = False
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True, pin_memory=True,
        num_workers=num_workers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch, shuffle=False, pin_memory=True,
        num_workers=num_workers)
    return train_loader, val_loader
