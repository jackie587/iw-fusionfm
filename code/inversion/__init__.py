"""参数反演模块：事件数据库几何观测 → 内波物理参数（振幅 η0、相速度 c）。

当前实现：两层流体 KdV/eKdV 反演（physics.py）+ 南海北部季节性参数场景
（scenarios.py）+ 事件库批量反演（batch_invert.py）+ 图件（plot_inversion.py）。

设计为可升级接口：physics.py 全部为纯函数（环境参数显式传入），后续
DJL 完全非线性求解器 / PINN 反演只需提供相同的 (观测, 环境) → (η0, c)
签名即可替换 kdv_invert / ekdv_invert，上游 batch_invert 不必改动。
"""
