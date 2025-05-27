import torch
import torch.nn as nn
import lightning.pytorch as pl
from net.model import PromptIR
from pytorch_msssim import ssim           # ← NEW
from utils.loss_utils import GANLoss
from utils.val_utils import compute_psnr_ssim
# -----------------------------------------------------------


class LitPromptIR(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):  # ↓ 預設 LR 改小
        super().__init__()
        self.save_hyperparameters()
        self.net = PromptIR(decoder=True)
        self.l1 = nn.L1Loss()
        self.hparams.lr = lr  # 儲存學習率
        # self.loss_fn = GANLoss(
        #     use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
        #     tensor=torch.FloatTensor)

    # -------------------------------------------------------

    def forward(self, x):
        return self.net(x)

    # ----------------------- TRAIN -------------------------
    def training_step(self, batch, _):
        pred = self(batch["degraded"])
        gt = batch["clean"]
        l1 = self.l1(pred, gt)
        ssim_ = 1.0 - ssim(pred, gt, data_range=1)
        loss = 0.9 * l1 + 0.1 * ssim_
        # loss = self.loss_fn(pred, gt)  # 使用 GANLoss
        self.log_dict(
            {"train/l1": l1, "train/ssim": ssim_, "train/loss": loss},
            prog_bar=True, on_step=True)
        # self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    # ---------------------- VAL ----------------------------
    def validation_step(self, batch, _):
        pred = self(batch["degraded"])
        gt = batch["clean"]
        l1 = self.l1(pred, gt)
        # psnr = 10 * torch.log10(1. / torch.mean((pred - gt) ** 2))
        psnr, ssim, _ = compute_psnr_ssim(pred, gt)
        self.log_dict({"val/l1": l1, "val/PSNR": psnr, "val/ssim": ssim},
                      prog_bar=True, on_epoch=True)

    # ----------------- OPT & LR SCHED ----------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )

        # CosineAnnealing 以 epoch 為單位更新
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=200, eta_min=1e-6
        )

        # -------- Lightning 2.x 建議寫法 --------
        scheduler_cfg = {
            "scheduler": sched,     # 必填
            "interval": "epoch",    # "step" 或 "epoch"
            "frequency": 1,         # 幾個 interval 呼叫一次
            "monitor": "val/PSNR",  # 只有需要基於 metric 動態調整才填
        }

        return {"optimizer": opt, "lr_scheduler": scheduler_cfg}
