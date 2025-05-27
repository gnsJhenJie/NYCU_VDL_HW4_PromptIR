# =============================================================
# file: infer_hw4.py  (v3.0)  —  export pred.npz and wrap into a zip with patch=64 and TTA matching training augmentations
# =============================================================
"""Inference + `.npz` exporter for leaderboard submission with patch size 64 and Test-Time Augmentation.
Transforms match training-time aug (8 modes).

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


def _tile_forward(model, x, tile: int, overlap: int):
    """Tiled forward pass to avoid OOM on large images."""
    _, _, h, w = x.shape
    stride = tile - overlap
    out = torch.zeros_like(x)
    weight = torch.zeros_like(x)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            y1 = min(y0 + tile, h)
            x1 = min(x0 + tile, w)
            patch = x[..., y0:y1, x0:x1]
            pad_h = tile - (y1 - y0)
            pad_w = tile - (x1 - x0)
            patch = torch.nn.functional.pad(
                patch, (0, pad_w, 0, pad_h), mode="reflect")
            pred = model(patch)[..., : y1 - y0, : x1 - x0]
            out[..., y0:y1, x0:x1] += pred
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
    ap.add_argument("--tile", type=int, default=128,
                    help="Patch size to tile (0 = disable tiling)")
    ap.add_argument("--overlap", type=int, default=0,
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
        if args.tile > 0 and max(input_x.shape[-2:]) > args.tile:
            return _tile_forward(model, input_x, args.tile, args.overlap)
        return model(input_x)

    # 定義 TTA 轉換與逆轉換  (training 用的 8 種模式)
    tta_transforms = []
    for mode in range(8):
        if mode == 0:
            def tfm(x): return x
            def inv(y): return y
        elif mode == 1:
            def tfm(x): return torch.flip(x, dims=[-2])  # flip up/down
            def inv(y): return torch.flip(y, dims=[-2])
        elif mode == 2:
            def tfm(x): return torch.rot90(
                x, k=1, dims=[-2, -1])  # rotate 90 CCW

            def inv(y): return torch.rot90(y, k=-1, dims=[-2, -1])
        elif mode == 3:
            def tfm(x): return torch.flip(
                torch.rot90(x, k=1, dims=[-2, -1]), dims=[-2])

            def inv(y): return torch.rot90(
                torch.flip(y, dims=[-2]),
                k=-1, dims=[-2, -1])
        elif mode == 4:
            def tfm(x): return torch.rot90(x, k=2, dims=[-2, -1])  # rotate 180
            def inv(y): return torch.rot90(y, k=2, dims=[-2, -1])
        elif mode == 5:
            def tfm(x): return torch.flip(
                torch.rot90(x, k=2, dims=[-2, -1]), dims=[-2])

            def inv(y): return torch.rot90(
                torch.flip(y, dims=[-2]),
                k=2, dims=[-2, -1])
        elif mode == 6:
            def tfm(x): return torch.rot90(x, k=3, dims=[-2, -1])  # rotate 270
            def inv(y): return torch.rot90(y, k=1, dims=[-2, -1])
        elif mode == 7:
            def tfm(x): return torch.flip(
                torch.rot90(x, k=3, dims=[-2, -1]), dims=[-2])

            def inv(y): return torch.rot90(
                torch.flip(y, dims=[-2]),
                k=1, dims=[-2, -1])
        tta_transforms.append((tfm, inv))

    preds: dict[str, np.ndarray] = {}
    root = Path(args.data_root)

    with torch.no_grad():
        for img_path in sorted(root.glob("*.png")):
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
