"""门控融合 DualFuse 前向/反向形状自检（临时脚本，验证后可删）。"""
import sys
sys.path.insert(0, str(__file__).rsplit("\\", 2)[0] if "\\" in __file__ else
                __file__.rsplit("/", 2)[0])

import torch

from models.segmentation.dual_fuse import build_dual_fuse
from utils.config import load_yaml

cfg = load_yaml("configs/model_dual_fuse_gated.yaml")
cfg["use_gradient_checkpointing"] = False
cfg["pretrained"] = None  # 自检不加载 ImageNet 权重（相对路径按项目根解析）
cfg["warm_start_unet"] = cfg["warm_start_swin"] = None  # 自检不加载热启动
m = build_dual_fuse(cfg)
x = torch.randn(1, 3, 512, 512)
out = m(x)
print("out:", tuple(out.shape),
      "params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
out.mean().backward()
print("backward ok")
print("gate bias init:", m.fuse4.gate.bias.mean().item(),
      "(sigmoid ≈ 0.27，CNN 初始注入弱)")

# v1 concat 配置仍能正常构建（旧检查点兼容性）
cfg1 = load_yaml("configs/model_dual_fuse.yaml")
cfg1["use_gradient_checkpointing"] = False
cfg1["pretrained"] = None
cfg1["warm_start_unet"] = cfg1["warm_start_swin"] = None
m1 = build_dual_fuse(cfg1)
print("concat v1 out:", tuple(m1(x).shape))
