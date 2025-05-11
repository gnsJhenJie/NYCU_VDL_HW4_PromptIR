"""Compute max / min HW across your dataset (train or test)."""
from pathlib import Path, PurePath
import sys, numpy as np
from PIL import Image

def main(folder):
    hmin=wmin=1e9
    hmax=wmax=0
    for p in Path(folder).glob("*.png"):
        img = Image.open(p)
        h,w = img.size[1], img.size[0]
        hmin=min(hmin,h); wmin=min(wmin,w)
        hmax=max(hmax,h); wmax=max(wmax,w)
    print(f"Resolution summary  →  min: {hmin}x{wmin}   max: {hmax}x{wmax}")
if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: python dataset_stats.py data/train/clean")
        sys.exit(1)
    main(sys.argv[1])