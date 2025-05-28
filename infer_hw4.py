# =============================================================
# file: infer_hw4.py  (v3.1)  —  export pred.npz and wrap into a zip with patch=64, TTA matching training aug, and robust tiling
# =============================================================
"""Inference + `.npz` exporter for leaderboard submission with patch size 64,
Test-Time Augmentation matching training augmentations, and improved tiling logic.

用法範例：
```bash
python infer_hw4.py \
       --ckpt ckpts/promptir-epoch=119-val_PSNR=34.76.ckpt \
       --zip_name myrun.zip \
       --tta
```
* 會生成 `pred.npz`（key=檔名，value shape=(3,H,W) uint8）
* 並把它壓縮成 `myrun.zip`，方便直接上傳 leaderboard。
"""
from __future__ import annotations
import argparse
import zipfile
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from lightning_module import LitPromptIR

_TO_TENSOR = ToTensor()

# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------


def _pad_to_multiple(x: torch.Tensor, m: int = 8):
    """Reflect-pad (B,C,H,W) 使 H,W 變為 m 的倍數，回傳 (padded, orig_h, orig_w)."""
    h, w = x.shape[-2:]
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    if pad_h or pad_w:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, h, w


def _tile_forward(model, x: torch.Tensor, tile: int, overlap: int):
    """Tiled forward pass with overlap to avoid OOM and reduce seams."""
    b, c, h, w = x.shape
    # if image smaller than tile, just forward
    if h <= tile and w <= tile and overlap == 0:
        return model(x)
    # compute stride and tile positions
    stride = max(tile - overlap, 1)
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)

    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile, x0 + tile
            patch = x[..., y0:y1, x0:x1]
            # pad if necessary
            ph, pw = patch.shape[-2:]
            if ph < tile or pw < tile:
                pad_h = tile - ph
                pad_w = tile - pw
                patch = torch.nn.functional.pad(
                    patch, (0, pad_w, 0, pad_h), mode="reflect")
            pred = model(patch)
            out[..., y0:y1, x0:x1] += pred[..., :y1 - y0, :x1 - x0]
            weight[..., y0:y1, x0:x1] += 1
    return out / weight

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Path to Lightning checkpoint (.ckpt)")
    ap.add_argument(
        "--data_root", default="data/test/degraded",
        help="Folder with degraded test PNGs")
    ap.add_argument("--tile", type=int, default=64,
                    help="Patch size to tile (0 = disable tiling)")
    ap.add_argument("--overlap", type=int, default=32,
                    help="Tile overlap (#pixels)")
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--zip_name", default="submission.zip",
                    help="Name of output zip containing pred.npz")
    ap.add_argument("--tta", action="store_true",
                    help="Enable Test-Time Augmentation")
    args = ap.parse_args()

    device = torch.device(args.device)
    model = LitPromptIR.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    if args.half:
        model.half()

    # 定義 forward 函數
    def forward_fn(input_x: torch.Tensor) -> torch.Tensor:
        if args.tile > 0:
            return _tile_forward(model, input_x, args.tile, args.overlap)
        return model(input_x)

    # 定義 TTA 轉換與逆轉換  (training 用的 8 種模式)
    tta_transforms: list[tuple[callable, callable]] = []
    for mode in range(8):
        if mode == 0:
            def tfm(x): return x
            def inv(y): return y
        elif mode == 1:
            def tfm(x): return torch.flip(x, dims=[-2])
            def inv(y): return torch.flip(y, dims=[-2])
        elif mode == 2:
            def tfm(x): return torch.rot90(x, k=1, dims=[-2, -1])
            def inv(y): return torch.rot90(y, k=-1, dims=[-2, -1])
        elif mode == 3:
            def tfm(x): return torch.flip(
                torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2])

            def inv(y): return torch.rot90(
                torch.flip(y, dims=[-2]),
                k=-1, dims=[-2, -1])
        elif mode == 4:
            def tfm(x): return torch.rot90(x, k=2, dims=[-2, -1])
            def inv(y): return torch.rot90(y, k=2, dims=[-2, -1])
        elif mode == 5:
            def tfm(x): return torch.flip(
                torch.rot90(x, k=2, dims=[-2, -1]), dims=[-2])

            def inv(y): return torch.rot90(
                torch.flip(y, dims=[-2]),
                k=2, dims=[-2, -1])
        elif mode == 6:
            def tfm(x): return torch.rot90(x, k=3, dims=[-2, -1])
            def inv(y): return torch.rot90(y, k=1, dims=[-2, -1])
        else:
            def tfm(x): return torch.flip(
                torch.rot90(x, k=3, dims=[-2, -1]), dims=[-2])

            def inv(y): return torch.rot90(
                torch.flip(y, dims=[-2]),
                k=1, dims=[-2, -1])
        tta_transforms.append((tfm, inv))

    preds: dict[str, np.ndarray] = {}
    root = Path(args.data_root)

    with torch.no_grad():
        for img_path in tqdm(sorted(root.glob("*.png"))):
            # 1) load image
            img = Image.open(img_path).convert("RGB")
            x = _TO_TENSOR(img).unsqueeze(0).to(device)
            if args.half:
                x = x.half()

            # 2) pad to multiple of 8
            x, H, W = _pad_to_multiple(x, 8)

            # 3) forward (+ optional TTA)
            if args.tta:
                outs: list[torch.Tensor] = []
                for tfm, inv in tta_transforms:
                    y_aug = forward_fn(tfm(x))
                    outs.append(inv(y_aug))
                y = torch.stack(outs, dim=0).mean(dim=0)
            else:
                y = forward_fn(x)

            # 4) remove padding
            y = y[..., :H, :W]

            # 5) convert to numpy (3,H,W) uint8
            arr = (y.clamp(0, 1).cpu().squeeze().numpy() * 255).astype(np.uint8)
            preds[img_path.name] = arr

    # 6) save pred.npz
    npz_path = Path("pred.npz")
    np.savez(npz_path, **preds)
    print(f"Saved {len(preds)} images to {npz_path}")

    # 7) wrap into zip
    zip_path = Path(args.zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(npz_path, arcname=npz_path.name)
    print(f"Created zip archive → {zip_path.resolve()}")


if __name__ == "__main__":
    main()
