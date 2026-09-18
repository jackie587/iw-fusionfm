"""训练上下文校验器（两阶段检测第二阶段：候选斑块 → 真内波/虚警 二分类）。

两套场景级防泄漏划分 × 两种输入消融，共 4 个实验：
  --split winter_holdout : train=round1-5, test=round6（冬）
  --split summer_holdout : train=round2-6, test=round1（夏）
  --mode center  : 仅 512 中心裁块单通道（基线）
  --mode context : 中心裁块 + 1536 上下文 双通道

uncertain 标签不参与训练与评估；坐标校验失败的样本剔除。
v9 改进（2026-09-07）：
- 训练集按面积过滤巨型连通域（area_px > 3e6，满幅风滚轴/条形码伪影属
  不可学尾部分布）；测试集不过滤（生产中巨块照样会出现，须诚实计入 FAR）；
- 边缘样本（512 窗超出场景、反射填充伪影，edge_flags.json 标记）
  训练与测试均剔除（该伪影是裁块产物，生产推理的 feather 拼接不存在）。

用法（在 code/ 目录下）：
    python training/train_verifier.py --split winter_holdout --mode context
    python training/train_verifier.py --split winter_holdout --mode center --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent

from models.verification.context_verifier import Verifier  # noqa: E402

SAMPLE_ROOT = PROJECT_ROOT / "data" / "datasets" / "negative_samples"
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "experiments" / "context_verifier"
MAX_TRAIN_AREA_PX = 3e6  # 训练集面积上限：巨型连通域（满幅风滚轴/条形码伪影）不可学

SPLITS = {
    "winter_holdout": {"train_rounds": [1, 2, 3, 4, 5], "test_rounds": [6]},
    "summer_holdout": {"train_rounds": [2, 3, 4, 5, 6], "test_rounds": [1]},
}


def load_index() -> list[dict]:
    """合并 manifest 与坐标缓存 meta，剔除 uncertain / 坐标校验失败样本。"""
    manifest = json.load(open(SAMPLE_ROOT / "manifest.json", encoding="utf-8"))
    meta = json.load(open(SAMPLE_ROOT / "context_cache" / "meta.json",
                          encoding="utf-8"))["samples"]
    status = {m["index"]: m["status"] for m in meta}
    flags_path = SAMPLE_ROOT / "edge_flags.json"
    edge = set(json.load(open(flags_path, encoding="utf-8"))["indices"]) \
        if flags_path.exists() else set()
    out = []
    for i, e in enumerate(manifest):
        if e["label"] == "uncertain":
            continue
        if status.get(i) != "ok":
            continue
        out.append({"index": i, "file": e["file"], "label": e["label"],
                    "scene": e["scene"], "round": e["round"],
                    "area_px": e.get("area_px", 0), "edge": i in edge})
    return out


class VerifierDataset(Dataset):
    """输入张量 [C,384,384]：center=[vv], context=[vv中心, vv上下文]。"""

    def __init__(self, entries: list[dict], mode: str, augment: bool = False):
        self.entries = entries
        self.mode = mode
        self.augment = augment
        self.ctx = np.load(SAMPLE_ROOT / "context_cache" / "context.npy",
                           mmap_mode="r")

    def __len__(self):
        return len(self.entries)

    def _load(self, e: dict) -> tuple[torch.Tensor, float]:
        d = np.load(SAMPLE_ROOT / e["file"])
        vv = torch.from_numpy(d["vv"].astype(np.float32))[None, None]   # [1,1,512,512]
        vv = F.interpolate(vv, size=384, mode="bilinear",
                           align_corners=False)[0]                       # [1,384,384]
        y = 1.0 if e["label"] == "positive" else 0.0
        if self.mode == "center":
            return vv, y
        ctx = torch.from_numpy(self.ctx[e["index"]].astype(np.float32))[None]
        return torch.cat([vv, ctx], 0), y

    def __getitem__(self, i: int):
        x, y = self._load(self.entries[i])
        if self.augment:
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-1])
            if torch.rand(()) < 0.5:
                x = torch.flip(x, dims=[-2])
            if torch.rand(()) < 0.5:
                x = x.transpose(-1, -2)  # 主对角镜像，等效 rot90 族
        return x, y


def scene_level_val_split(entries: list[dict], seed: int = 0,
                          val_frac: float = 0.2) -> tuple[list[dict], list[dict]]:
    """从训练集按场景切出验证集（模型选择用），保证验证集含正负两类。"""
    scenes = sorted({e["scene"] for e in entries})
    rng = np.random.RandomState(seed)
    for _ in range(100):
        val_scenes = set(rng.choice(scenes, size=max(1, int(len(scenes) * val_frac)),
                                    replace=False))
        val = [e for e in entries if e["scene"] in val_scenes]
        labels = {e["label"] for e in val}
        if labels == {"positive", "negative"}:
            tr = [e for e in entries if e["scene"] not in val_scenes]
            return tr, val
    raise RuntimeError("无法切出同时含正负类的场景级验证集")


def vv_stats(entries: list[dict]) -> tuple[float, float]:
    """训练集 vv 通道均值/标准差（对中心裁块统计）。"""
    s, sq, n = 0.0, 0.0, 0
    for e in entries:
        vv = np.load(SAMPLE_ROOT / e["file"])["vv"].astype(np.float32)
        s += float(vv.sum()); sq += float((vv ** 2).sum()); n += vv.size
    mean = s / n
    return mean, float(np.sqrt(sq / n - mean ** 2))


@torch.no_grad()
def predict_scores(model: Verifier, entries: list[dict], mode: str,
                   batch_size: int = 64) -> np.ndarray:
    model.eval()
    ds = VerifierDataset(entries, mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    scores = []
    dev = next(model.parameters()).device
    for x, _ in dl:
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            scores.append(torch.sigmoid(model(x.to(dev))).float().cpu())
    return torch.cat(scores).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=list(SPLITS), required=True)
    ap.add_argument("--mode", choices=["center", "context"], required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="小数据 1 epoch 冒烟")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.split}_{args.mode}"
    out_dir = CKPT_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = load_index()
    cfg = SPLITS[args.split]
    train_all = [e for e in entries if e["round"] in cfg["train_rounds"]
                 and not e["edge"] and e["area_px"] <= MAX_TRAIN_AREA_PX]
    test = [e for e in entries if e["round"] in cfg["test_rounds"]
            and not e["edge"]]
    n_drop = sum(1 for e in entries if e["round"] in cfg["train_rounds"]) \
        - len(train_all)
    print(f"[过滤] 训练集剔除边缘/巨型样本 {n_drop} 条；"
          f"train={len(train_all)} test={len(test)}")
    train, val = scene_level_val_split(train_all, seed=args.seed)
    if args.smoke:
        train, val, test, epochs = train[:64], val[:32], test[:32], 1
    else:
        epochs = args.epochs

    mean, std = vv_stats(train)
    model = Verifier(args.mode, vv_mean=mean, vv_std=std).to(dev)
    n_pos = sum(e["label"] == "positive" for e in train)
    n_neg = len(train) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.GradScaler("cuda", enabled=dev.type == "cuda")

    dl = DataLoader(VerifierDataset(train, args.mode, augment=True),
                    batch_size=args.batch_size, shuffle=True, num_workers=0,
                    drop_last=False)
    best_val_loss, history = float("inf"), []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, cnt = time.time(), 0.0, 0
        for x, y in dl:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(
                    model(x), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss.detach()) * len(y); cnt += len(y)
        sched.step()
        # 验证集模型选择（场景级、无增强）
        val_scores = predict_scores(model, val, args.mode)
        val_y = np.array([e["label"] == "positive" for e in val], np.float32)
        vl = F.binary_cross_entropy_with_logits(
            torch.from_numpy(val_scores).clamp(1e-6, 1 - 1e-6).logit(),
            torch.from_numpy(val_y)).item()
        history.append({"epoch": ep, "train_loss": tot / cnt, "val_loss": vl})
        tag = ""
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save({"model": model.state_dict(), "mode": args.mode,
                        "split": args.split, "vv_mean": mean, "vv_std": std,
                        "epoch": ep, "args": vars(args)},
                       out_dir / "best.pt")
            tag = " *"
        print(f"ep{ep:02d} train_loss={tot/cnt:.4f} val_loss={vl:.4f}"
              f" ({time.time()-t0:.0f}s){tag}", flush=True)

    # 用选中 checkpoint 对 train/val/test 全量打分，供 eval_verifier 汇总
    ckpt = torch.load(out_dir / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ckpt["model"])
    preds = {}
    for name, es in [("train", train), ("val", val), ("test", test)]:
        if args.smoke and name == "train":
            es = train_all[:0]  # 冒烟不存 train 预测
        scores = predict_scores(model, es, args.mode) if es else np.array([])
        preds[name] = [{"index": e["index"], "file": e["file"],
                        "label": e["label"], "scene": e["scene"],
                        "round": e["round"], "score": float(s)}
                       for e, s in zip(es, scores)]
    with open(out_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump({"split": args.split, "mode": args.mode,
                   "best_epoch": ckpt["epoch"], "history": history,
                   "n_train": len(train), "n_val": len(val), "n_test": len(test),
                   "n_train_pos": n_pos, "n_train_neg": n_neg,
                   "val_scenes": sorted({e["scene"] for e in val}),
                   "predictions": preds}, f, ensure_ascii=False, indent=1)
    print(f"→ {out_dir}/ (best.pt epoch{ckpt['epoch']} + predictions.json)")


if __name__ == "__main__":
    main()
