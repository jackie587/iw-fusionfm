"""训练器：AMP + 梯度累积 + cosine LR + 实验归档（8GB 显存适配）。

实验归档遵守目录约定第 2 条：权重/配置快照/日志三位一体，
统一写入 checkpoints/experiments/<experiment_name>/。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.logger import get_logger


class Trainer:
    def __init__(self, model: nn.Module, criterion, train_loader: DataLoader,
                 val_loader: DataLoader | None, cfg: dict, work_dir: str | Path):
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(cfg["experiment_name"],
                                 self.work_dir / "train.log")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion

        tcfg = cfg["train"]
        self.epochs = tcfg["epochs"]
        self.accum = tcfg.get("grad_accum_steps", 1)
        self.amp = tcfg.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.amp)

        self.opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                     weight_decay=tcfg.get("weight_decay", 0.01))
        warmup = tcfg.get("warmup_epochs", 0)
        main_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=max(1, self.epochs - warmup))
        self.scheduler = main_sch
        self.warmup_epochs = warmup
        self.base_lr = tcfg["lr"]

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_dice = 0.0

    def _warmup_step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(1, self.warmup_epochs)
            for g in self.opt.param_groups:
                g["lr"] = lr

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        self._warmup_step(epoch)
        self.opt.zero_grad(set_to_none=True)
        agg: dict[str, float] = {}
        n = 0

        for step, batch in enumerate(self.train_loader):
            img = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            swot = batch.get("swot")
            if swot is not None:
                swot = swot.to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = (self.model(img, swot=swot) if swot is not None
                          else self.model(img))
                loss, detail = self.criterion(logits, mask)
                loss = loss / self.accum
            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum == 0 or step + 1 == len(self.train_loader):
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

            for k, v in detail.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1

        return {k: v / max(n, 1) for k, v in agg.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        if self.val_loader is None:
            return {}
        from evaluation.metrics import dice_score, iou_score  # 避免循环导入
        self.model.eval()
        dices, ious = [], []
        film_stats: list[dict] = []
        for batch in self.val_loader:
            img = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            swot = batch.get("swot")
            if swot is not None:
                swot = swot.to(self.device)
            with torch.amp.autocast("cuda", enabled=self.amp):
                logits = (self.model(img, swot=swot) if swot is not None
                          else self.model(img))
            pred = (torch.sigmoid(logits) > 0.5).float()
            dices.append(dice_score(pred, mask))
            ious.append(iou_score(pred, mask))
            if swot is not None and hasattr(self.model, "film_modulation_stats"):
                film_stats.append(self.model.film_modulation_stats(swot))
        out = {"val_dice": sum(dices) / len(dices),
               "val_iou": sum(ious) / len(ious)}
        if film_stats:
            # FiLM 调制强度：|γ-1| / |β| 在验证集上的均值（0 = 恒等退化）
            for k in film_stats[0]:
                out[f"film_{k}"] = sum(s[k] for s in film_stats) / len(film_stats)
        return out

    def save_ckpt(self, epoch: int, metrics: dict, name: str) -> None:
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "metrics": metrics,
        }, self.work_dir / name)

    def fit(self) -> None:
        # 配置快照（约定：配置与权重同目录归档）
        with open(self.work_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, ensure_ascii=False, indent=2, default=str)

        save_every = self.cfg.get("logging", {}).get("save_every_epoch", 5)
        for epoch in range(self.epochs):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate()
            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            msg = (f"epoch {epoch + 1}/{self.epochs} "
                   f"loss={train_m.get('total', 0):.4f} "
                   + " ".join(f"{k}={v:.4f}" for k, v in val_m.items())
                   + f" ({time.time() - t0:.0f}s)")
            self.logger.info(msg)

            if val_m.get("val_dice", 0) > self.best_dice:
                self.best_dice = val_m["val_dice"]
                self.save_ckpt(epoch, val_m, "best.pth")
            if (epoch + 1) % save_every == 0:
                self.save_ckpt(epoch, {**train_m, **val_m}, f"epoch{epoch + 1}.pth")

        self.save_ckpt(self.epochs - 1, val_m, "last.pth")
        self.logger.info("训练完成，best val_dice=%.4f，权重在 %s",
                         self.best_dice, self.work_dir)
