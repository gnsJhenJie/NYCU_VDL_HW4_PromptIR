# =============================================================
# file: train_hw4.py (unchanged except for import path note)
# =============================================================
"""Launch training with all GPUs available (defaults to 1)."""
import argparse
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from datasets_hw4 import get_train_val_loaders
from lightning_module import LitPromptIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data", type=str)
    parser.add_argument("--batch",      default=8,  type=int)
    parser.add_argument("--patch",      default=256, type=int)
    parser.add_argument("--epochs",     default=120, type=int)
    parser.add_argument("--gpus",       default=1,   type=int)
    parser.add_argument("--accum", default=1, type=int,
                        help="gradient accumulation")
    parser.add_argument("--precision",  default=16,
                        type=int, choices=[16, 32])
    args = parser.parse_args()

    train_loader, val_loader = get_train_val_loaders(
        args.data_root, args.batch, num_workers=8, patch=args.patch)

    model = LitPromptIR()

    ckpt_cb = ModelCheckpoint(
        dirpath="ckpts",
        filename="promptir-{epoch:03d}-{val_PSNR:.2f}",
        save_top_k=5,
        mode="max",
        monitor="val/PSNR")

    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.gpus,
        strategy="ddp" if args.gpus > 1 else "auto",
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.accum,       # ← 等效大 batch
        callbacks=[ckpt_cb, lr_cb],
        logger=TensorBoardLogger("logs", name="promptir_hw4"),
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_loader, val_loader)
