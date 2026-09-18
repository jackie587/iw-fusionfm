"""基线训练入口。

用法（在 code/ 目录下）：
    python training/scripts/train_baseline.py --config configs/train_baseline.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

from torch.utils.data import DataLoader

from models.segmentation.unet import build_unet
from models.segmentation.swin_unet import build_swin_unet
from training.datasets.tao_dataset import TaoIWDataset, train_val_split
from training.datasets.tile_dataset import TileDataset
from training.losses.composite import CompositeLoss
from training.trainers.trainer import Trainer
from utils.config import load_config, load_yaml, resolve_nested
from utils.seed import set_seed


def build_dataloaders(cfg: dict):
    data_cfg = load_yaml(PROJECT_ROOT / cfg["data"]["config"])
    dname = cfg["data"]["dataset"]
    if dname == "tao2022" or dname.startswith("tao2022_negmix"):
        root = Path(resolve_nested(data_cfg, "paths.l1_dataset", PROJECT_ROOT)) / dname
        full = TaoIWDataset(root)
        if dname.startswith("tao2022_negmix"):
            # 负样本裁块 id≥100000 只进训练集；划分只作用在 Tao id 上
            # （与原实验同种子同逻辑 → 验证集严格一致，可横向对比）
            import numpy as np
            tao_ids = np.array(sorted(i for i in full.images if i < 100000))
            crop_ids = sorted(i for i in full.images if i >= 100000)
            rng = np.random.RandomState(cfg["seed"])
            rng.shuffle(tao_ids)
            n_tr = int(len(tao_ids) * cfg["data"]["train_split"])
            tr_ids = tao_ids[:n_tr].tolist() + crop_ids
            va_ids = tao_ids[n_tr:].tolist()
        else:
            tr_ids, va_ids = train_val_split(full, cfg["data"]["train_split"],
                                             cfg["seed"])
        train_ds = TaoIWDataset(root, split_ids=tr_ids, augment=True,
                                domain_aug=cfg["data"].get("domain_aug", False))
        val_ds = TaoIWDataset(root, split_ids=va_ids)
    elif dname == "tiles":
        tile_dir = resolve_nested(data_cfg, "paths.sar_tiles", PROJECT_ROOT)
        train_ds = val_ds = TileDataset(tile_dir)
    elif dname == "l2pair":
        # L2 S1×SWOT 配对块：按天划分（20230830 验证，其余训练）
        from training.datasets.l2_dataset import L2PairDataset
        root = Path(resolve_nested(data_cfg, "paths.l2_dataset", PROJECT_ROOT)) / "tiles"
        train_ds = L2PairDataset(
            root, split="train",
            swot_dropout=cfg["data"].get("swot_dropout", 0.0),
            swot_zero=cfg["data"].get("swot_zero", False),
            augment=True)
        val_ds = L2PairDataset(
            root, split="val",
            swot_zero=cfg["data"].get("swot_zero", False))
    else:
        raise ValueError(f"未知数据集：{dname}")

    bs = cfg["train"]["batch_size"]
    nw = cfg["data"].get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True)
    return train_loader, val_loader


def build_model(cfg: dict):
    model_cfg = load_yaml(PROJECT_ROOT / cfg["model"]["config"])
    # 权重路径按项目根解析（配置里写的是相对项目根的路径）
    for key in ("imagenet_weights", "ssl4eo_weights",
                    "warm_start_unet", "warm_start_swin", "init_ckpt"):
        if model_cfg.get(key):
            p = Path(model_cfg[key])
            model_cfg[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)
    if model_cfg["name"] == "unet":
        model = build_unet(model_cfg)
    elif model_cfg["name"] == "swin_unet":
        model = build_swin_unet(model_cfg)
    elif model_cfg["name"] == "dual_fuse":
        from models.segmentation.dual_fuse import build_dual_fuse
        model = build_dual_fuse(model_cfg)
    elif model_cfg["name"] == "swin_film":
        from models.segmentation.swin_film import build_swin_film
        model = build_swin_film(model_cfg)
    else:
        raise ValueError(f"未知模型：{model_cfg['name']}")
    if model_cfg.get("init_ckpt"):
        # 微调：整模型权重热加载（覆盖 imagenet/ssl4eo 初始化）
        import torch
        ckpt = torch.load(model_cfg["init_ckpt"], map_location="cpu",
                          weights_only=True)
        sd = ckpt.get("model", ckpt)
        if model_cfg["name"] == "swin_film":
            # 融合模型比 v7 多出 swot_encoder/film.mlp，按形状匹配部分加载
            msd = model.state_dict()
            sub = {k: v for k, v in sd.items()
                   if k in msd and msd[k].shape == v.shape}
            model.load_state_dict(sub, strict=False)
            n_miss = len(msd) - len(sub)
            print(f"[init_ckpt] 部分热加载 ← {model_cfg['init_ckpt']}："
                  f"匹配 {len(sub)} 层，缺失（随机初始化）{n_miss} 层")
        else:
            model.load_state_dict(sd)
            print(f"[init_ckpt] 整模型加载 ← {model_cfg['init_ckpt']}（{len(sd)} 层）")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_baseline.yaml")
    args = ap.parse_args()

    cfg = load_config(PROJECT_ROOT / "code" / args.config
                      if not Path(args.config).is_absolute() else args.config)
    set_seed(cfg.get("seed", 42))

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    criterion = CompositeLoss(cfg["loss"])

    work_dir = PROJECT_ROOT / cfg["logging"]["ckpt_dir"] / cfg["experiment_name"]
    trainer = Trainer(model, criterion, train_loader, val_loader, cfg, work_dir)
    trainer.fit()


if __name__ == "__main__":
    main()
