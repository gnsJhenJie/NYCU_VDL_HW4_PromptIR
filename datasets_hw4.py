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
from utils.image_utils import random_augmentation
from torch.utils.data import Subset


__all__ = ["HW4PairDataset", "get_train_val_loaders"]

# _AUG = T.Compose([
#     T.RandomHorizontalFlip(),
#     T.RandomVerticalFlip(),
#     T.RandomRotation(90, expand=False),
#     # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
# ])

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
        self.training = split == "train"
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

    def _crop(self, arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w, _ = arr1.shape
        if h < self.patch or w < self.patch:
            pad_h = max(0, self.patch - h)
            pad_w = max(0, self.patch - w)
            arr1 = np.pad(
                arr1, ((0, pad_h),
                       (0, pad_w),
                       (0, 0)),
                mode="reflect")
            h, w = arr1.shape[:2]
        y = random.randint(0, h - self.patch)
        x = random.randint(0, w - self.patch)
        return arr1[y:y+self.patch, x:x+self.patch], arr2[y:y+self.patch, x:x+self.patch]

    def _cutmix(self, a1, b1):
        """Apply simple CutMix/linear‑mix with another random sample."""
        j = random.randrange(len(self.pairs))
        a2_path, b2_path = self.pairs[j]
        a2 = np.array(Image.open(a2_path).convert("RGB"))
        b2 = np.array(Image.open(b2_path).convert("RGB"))

        if self.patch > 0:                       # 做同樣的 crop
            a2, b2 = self._crop(a2, b2)

        lam = 0.5
        a = (lam * a1.astype(np.float32) + (1 - lam) * a2.astype(np.float32))
        b = (lam * b1.astype(np.float32) + (1 - lam) * b2.astype(np.float32))
        a = np.clip(a, 0, 255).astype(np.uint8)
        b = np.clip(b, 0, 255).astype(np.uint8)
        return a, b

    def __getitem__(self, idx):
        deg_path, clean_path = self.pairs[idx]
        deg = np.array(Image.open(deg_path).convert("RGB"))
        clean = np.array(Image.open(clean_path).convert("RGB"))
        if self.patch > 0:
            deg, clean = self._crop(deg, clean)
        # CutMix (10%): temorarily disabled
        if self.training and random.random() < 0.1 and False:
            deg, clean = self._cutmix(deg, clean)
        if self.training:
            deg, clean = random_augmentation(deg, clean)
        deg = _TO_TENSOR(deg)
        clean = _TO_TENSOR(clean)
        prompt_id = 0 if "rain" in deg_path.stem else 1
        return {"degraded": deg, "clean": clean, "prompt_id": prompt_id}

    training: bool = True

# ---------- loaders ----------------------------------------------------------


# def get_train_val_loaders(
#         root: str, batch: int, num_workers: int = 8, patch: int = 256):
#     full = HW4PairDataset(root, "train", patch)
#     n_val = int(0.05 * len(full))
#     n_train = len(full) - n_val
#     train_set, val_set = torch.utils.data.random_split(
#         full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
#     train_set.dataset.training = True
#     val_set.dataset.training = False

#     train_loader = torch.utils.data.DataLoader(
#         train_set, batch_size=batch, shuffle=True, pin_memory=True,
#         num_workers=num_workers, drop_last=True)
#     val_loader = torch.utils.data.DataLoader(
#         val_set, batch_size=batch, shuffle=False, pin_memory=True,
#         num_workers=num_workers)
#     return train_loader, val_loader

def get_train_val_loaders(
        root: str, batch: int, num_workers: int = 8, patch: int = 256):
    # 1. 建原始 dataset，取出所有樣本數
    full = HW4PairDataset(root, "train", patch)
    N = len(full)
    n_val = int(0.05 * N)
    n_train = N - n_val

    # 2. 產生打亂的索引（固定 seed 可重現）
    indices = list(range(N))
    random.seed(0)
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # 3. 各自建一個 dataset 實例
    train_ds = HW4PairDataset(root, "train", patch)
    val_ds = HW4PairDataset(root, "train", patch)

    # 4. 分別設定 training flag
    train_ds.training = True
    val_ds.training = False

    # 5. 用相同的 idx 切成 Subset
    train_set = Subset(train_ds, train_idx)
    val_set = Subset(val_ds,   val_idx)

    # 6. 建 DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch, shuffle=True,
        pin_memory=True, num_workers=num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch, shuffle=False,
        pin_memory=True, num_workers=num_workers
    )

    return train_loader, val_loader


if __name__ == "__main__":

    import argparse

    import torch

    from torchvision.utils import make_grid, save_image

    parser = argparse.ArgumentParser(

        description="Visualize a batch of degraded↔clean image pairs.")

    parser.add_argument("--root", type=str, default="data",

                        help="Root directory of dataset (e.g., data)")

    parser.add_argument("--batch-size", type=int, default=4,

                        help="Batch size to load and visualize")

    parser.add_argument("--num-workers", type=int, default=0,

                        help="Number of workers for DataLoader")

    parser.add_argument("--patch", type=int, default=128,

                        help="Patch size for cropping")

    parser.add_argument("--output", type=str, default="batch.png",

                        help="Output filename for the grid image")

    args = parser.parse_args()

    train_loader, _ = get_train_val_loaders(

        args.root, args.batch_size, args.num_workers, args.patch)

    batch = next(iter(train_loader))

    deg = batch["degraded"]  # (B, C, H, W)

    clean = batch["clean"]    # (B, C, H, W)

    # Concatenate degraded and clean side by side for each sample

    paired = [torch.cat([deg[i], clean[i]], dim=2) for i in range(deg.size(0))]

    grid = make_grid(paired, nrow=args.batch_size,
                     normalize=True, scale_each=True)

    save_image(grid, args.output)

    print(f"Saved batch visualization to {args.output}")
