import torch
import torch.nn as nn
import lightning.pytorch as pl
from net.model import PromptIR


class LitPromptIR(pl.LightningModule):
    def __init__(self, lr: float = 2e-4):
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        out = self(batch["degraded"])
        loss = self.loss_fn(out, batch["clean"])
        self.log("train/l1", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        out = self(batch["degraded"])
        loss = self.loss_fn(out, batch["clean"])
        psnr = 10 * torch.log10(1. / torch.mean((out - batch["clean"]) ** 2))
        self.log_dict({"val/l1": loss, "val/PSNR": psnr}, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=150, eta_min=1e-6)
        return {"optimizer": opt, "lr_scheduler": sched}
