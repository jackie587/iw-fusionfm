"""两层流体 KdV / eKdV 内孤立波反演核心（纯函数，无 IO）。

符号约定（遵循 Helfrich & Melville 2006；Jia et al. 2019, RS 11:1706 式12-22）：
    eKdV:  η_t + (c0 + αη + α1η²)η_x + βη_xxx = 0
    下凹波（depression，南海北部典型）η < 0，对应 h1 < h2 时 α < 0、α1 < 0。
    振幅一律用 |η0|（米，正值）输出，极性单独标注。

系数（两层、刚盖近似、Boussinesq）：
    c0  = sqrt(g · Δρ/ρ · h1 h2 / (h1+h2))          # 线性长波速
    α   = (3/2) c0 (h1-h2) / (h1 h2)                # 二次非线性
    β   = c0 h1 h2 / 6                              # 频散
    α1  = 3c0/(h1h2)² · [7/8·(h1-h2)² − (h1³+h2³)/(h1+h2)]   # 三次非线性（恒负）

KdV 孤子解 η = η0·sech²(ξ/l)：
    l² = 12β/(αη0)，  c = c0 + αη0/3
    （注意：若写作 Δ=sqrt(4β/(αη0)) 的形式，Δ 与 sech² 剖面半宽 l 差 √3 倍；
     本模块采用与 c=c0+αη0/3 自洽的 l²=12β/(αη0)。）

观测口径（关键假设）：
    事件库 wavelength_m 是相邻波峰间距中位数（build_event_database.py 由
    波峰质心投影间距得到），并非单个孤子的剖面宽度。按通行做法把首孤子
    峰间距折算为特征半宽： l = λ / κ。
    κ 中心值取 π —— 依据 Jia et al. 2019 海南陆架个例：峰间距 440 m、
    曲线拟合半宽 l≈145 m，κ≈3.0。η0 ∝ κ²，κ 的不确定度可直接解析换算
    （κ=2 → η0×0.41；κ=2π → η0×4.0），批量结果中 κ 固定、敏感性单列。

eKdV 孤子解（Jia et al. 2019 式19-22）：
    η(ξ) = η0 / (b + (1-b)cosh²(γξ))
    γ² = η0(α + α1η0/2)/(12β)
    b  = −η0α1/(2α + α1η0)          （b<1；|η0|→|α/α1| 时 b→1 波形平顶、宽度→∞）
    c  = c0 + η0(α + α1η0/2)/3
半幅点：cosh²(γξ_h) = (2-b)/(1-b)。定义等效半宽 l_e ≡ ξ_h/arccosh(√2)，
KdV 极限（b→0）下 l_e = 1/γ = l，与 KdV 口径一致。
反演即解 l_e(η0) = l。l_e(η0) 在 η0→0 与 η0→a_lim 两端都 →∞，中间有
最小值 → 可能存在双解（方案文档所述"双解问题"）：取小振幅根（与 KdV
解连续的分支）作为 eKdV 修正，大振幅根记录在案；无解则只回退 KdV。
"""
from __future__ import annotations

import math

G = 9.81  # m/s²

KAPPA_DEFAULT = math.pi  # λ = κ·l 的中心口径（Jia et al. 2019 个例 κ≈3.0）
_HALF = math.acosh(math.sqrt(2.0))  # sech² 剖面半幅点位置系数 ≈ 0.8814


def two_layer_coeffs(h1: float, h2: float, drho: float,
                     g: float = G) -> dict:
    """两层流体 KdV/eKdV 系数。

    参数：h1 上层（混合层）厚度 m；h2 下层厚度 m；drho = Δρ/ρ 约化密度差。
    返回 dict(c0, alpha, beta, alpha1)，单位 SI。
    假设：h1>0, h2>0, drho>0；h1<h2 为下凹波 regime（α<0）。
    """
    if h1 <= 0 or h2 <= 0 or drho <= 0:
        raise ValueError(f"非物理参数 h1={h1}, h2={h2}, drho={drho}")
    c0 = math.sqrt(g * drho * h1 * h2 / (h1 + h2))
    alpha = 1.5 * c0 * (h1 - h2) / (h1 * h2)
    beta = c0 * h1 * h2 / 6.0
    alpha1 = (3.0 * c0 / (h1 * h2) ** 2
              * (7.0 / 8.0 * (h1 - h2) ** 2 - (h1 ** 3 + h2 ** 3) / (h1 + h2)))
    return {"c0": c0, "alpha": alpha, "beta": beta, "alpha1": alpha1}


def kdv_invert(lam: float, c0: float, alpha: float, beta: float,
               kappa: float = KAPPA_DEFAULT) -> dict:
    """由峰间距 λ 反演 KdV 孤子振幅与相速度（下凹波，α<0）。

    返回 dict(eta0_m=|η0|, half_width_m=l, c_ms=c, polarity="depression")。
    假设：κ 口径见模块 docstring；α≠0；结果η0 与 1/l² 成正比。
    """
    if alpha == 0:
        raise ValueError("alpha=0（h1≈h2），KdV 二次非线性消失，需 eKdV/DJL")
    l = lam / kappa
    a = abs(alpha)
    eta0 = 12.0 * beta / (a * l * l)
    c = c0 + a * eta0 / 3.0  # 下凹波 α<0、η0<0 → αη0/3 = +a·|η0|/3
    return {"eta0_m": eta0, "half_width_m": l, "c_ms": c,
            "polarity": "depression"}


def ekdv_profile(eta0: float, alpha: float, alpha1: float,
                 beta: float) -> dict:
    """eKdV 孤子波形参数（η0 为振幅大小，正值；下凹波）。

    返回 dict(gamma, b, c_ms, xi_half)，xi_half 为半幅点距离（m）。
    振幅上限 a_lim = |α/α1|（b→1 平顶极限）。
    """
    a_lim = abs(alpha / alpha1)
    if not 0 < eta0 < a_lim:
        raise ValueError(f"η0={eta0:.1f} 超出 eKdV 孤子存在域 (0, {a_lim:.1f})")
    s = alpha + 0.5 * alpha1 * eta0  # α<0, α1<0, 取 η0=|η0| 时用 −|α|−... 统一写幅值形式
    # 幅值形式：对下凹波令 a=|α|, a1=|α1|，方程系数用 −a、−a1，η 取 −η0，
    # γ² = η0(a − a1η0/2)/(12β)（要求 η0 < 2a/a1，b<1 已更严）
    a, a1 = abs(alpha), abs(alpha1)
    gamma2 = eta0 * (a - 0.5 * a1 * eta0) / (12.0 * beta)
    if gamma2 <= 0:
        raise ValueError("γ²≤0，超出孤子存在域")
    gamma = math.sqrt(gamma2)
    b = eta0 * a1 / (2.0 * a - a1 * eta0)  # = −η0α1/(2α+α1η0)，下凹波 0<b<1
    if b >= 1.0:
        raise ValueError("b≥1，平顶极限之外")
    xi_half = math.acosh(math.sqrt((2.0 - b) / (1.0 - b))) / gamma
    c = c0_from = None  # 占位，c 由 ekdv_speed 计算
    return {"gamma": gamma, "b": b, "xi_half": xi_half,
            "a_lim": a_lim}


def ekdv_speed(eta0: float, c0: float, alpha: float, alpha1: float) -> float:
    """eKdV 相速度：c = c0 + |η0|( |α| − |α1|η0/2 )/3（下凹波幅值形式）。"""
    return c0 + eta0 * (abs(alpha) - 0.5 * abs(alpha1) * eta0) / 3.0


def ekdv_invert(lam: float, c0: float, alpha: float, alpha1: float,
                beta: float, kappa: float = KAPPA_DEFAULT,
                n_scan: int = 600) -> dict:
    """由峰间距 λ 反演 eKdV 振幅（数值求根，处理双解）。

    返回 dict:
      eta0_m      小振幅根（与 KdV 连续分支；无解时为 None）
      eta0_alt_m  大振幅根（近平顶分支；无双解时为 None）
      c_ms        对应小振幅根的 eKdV 相速度
      flag        "ok" / "dual_root" / "no_root"（l 低于 eKdV 最小可达半宽，
                  说明观测间距超出 eKdV 波形范围，回退 KdV）
    """
    l = lam / kappa
    target = _HALF * l  # 目标半幅点距离
    a_lim = abs(alpha / alpha1)
    lo, hi = 1e-3 * a_lim, 0.999 * a_lim

    def f(eta):
        return ekdv_profile(eta, alpha, alpha1, beta)["xi_half"] - target

    # 对数等距扫描找符号变化
    xs = [lo * (hi / lo) ** (i / (n_scan - 1)) for i in range(n_scan)]
    roots = []
    f_prev = f(xs[0])
    for i in range(1, n_scan):
        f_cur = f(xs[i])
        if f_prev == 0.0:
            roots.append(xs[i - 1])
        elif f_prev * f_cur < 0:
            roots.append(_brentq(f, xs[i - 1], xs[i]))
        f_prev = f_cur
    roots.sort()
    if not roots:
        return {"eta0_m": None, "eta0_alt_m": None, "c_ms": None,
                "flag": "no_root"}
    eta0 = roots[0]
    return {"eta0_m": eta0,
            "eta0_alt_m": (roots[1] if len(roots) > 1 else None),
            "c_ms": ekdv_speed(eta0, c0, alpha, alpha1),
            "flag": ("dual_root" if len(roots) > 1 else "ok")}


def _brentq(f, a: float, b: float, tol: float = 1e-10,
            maxiter: int = 100) -> float:
    """Brent 求根（简化实现，避免 scipy 依赖路径问题；f(a)·f(b)<0）。"""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("brentq: 区间端点同号")
    c, fc = a, fa
    d = e = b - a
    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * 1e-15 * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if abs(xm) <= tol1 or fb == 0.0:
            return b
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm
        a, fa = b, fb
        b += d if abs(d) > tol1 else (tol1 if xm > 0 else -tol1)
        fb = f(b)
    return b
