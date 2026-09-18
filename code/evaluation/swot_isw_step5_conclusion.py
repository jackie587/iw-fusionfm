"""任务 5：SNR 预算汇总与"SWOT 独立内波检测可行性"结论。

汇总步骤 0/2/3/4 的结果，产出：
- fig_step5_snr_budget.png：预期信号幅度 vs 各产品噪声/背景水平的对比；
- conclusion.json：能检/不能检的分层结论 + 条件。

用法（code/ 目录下）：python evaluation/swot_isw_step5_conclusion.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
PROJECT_ROOT = CODE_ROOT.parent
OUT = PROJECT_ROOT / "results/swot_isw_detection"

# 预期 ISW 表面信号：下沉型 η_s ≈ |η|·Δρ/ρ；南海 η~30-100m, Δρ/ρ~2e-3
SIGNAL_CM = (2.0, 10.0)      # 预期幅度范围（cm），典型 ~5cm


def main() -> None:
    noise = json.load(open(OUT / "step2_noise_report.json", encoding="utf-8"))
    coloc = json.load(open(OUT / "step3_collocation_report.json",
                           encoding="utf-8"))
    det = json.load(open(OUT / "step4_detector_summary.json", encoding="utf-8"))

    g = noise["granules"]
    tags = sorted(g)
    pix_noise = [g[t]["pixel_noise_m"] * 100 for t in tags]
    band0712 = [g[t]["band_rms_m"]["0.7-1.2km"] * 100 for t in tags]
    band0412 = [g[t]["band_rms_m"]["0.4-0.7km"] * 100 for t in tags]
    resid_rms = [g[t]["resid_rms_m"] * 100 for t in tags]

    # ---- SNR 预算图
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(tags))
    ax.bar(x - 0.25, resid_rms, 0.22, label="残差总 RMS（含 2.5-10km 海洋背景）")
    ax.bar(x, band0712, 0.22, label="0.7-1.2km 带内背景+噪声 RMS")
    ax.bar(x + 0.25, pix_noise, 0.22, label="像素级小尺度噪声")
    ax.axhspan(SIGNAL_CM[0], SIGNAL_CM[1], color="r", alpha=0.15,
               label=f"预期 ISW 信号 {SIGNAL_CM[0]:.0f}-{SIGNAL_CM[1]:.0f} cm")
    ax.axhline(5, color="r", ls="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("2023", "") for t in tags], rotation=45,
                       fontsize=8)
    ax.set_ylabel("cm")
    ax.set_title("Unsmoothed 250m：信号 vs 噪声/背景预算\n"
                 "（带内背景 0.5-0.7cm < 预期信号，但需沿峰相干平均"
                 "对抗 2.5-10km 海洋背景）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_step5_snr_budget.png", dpi=150)
    plt.close(fig)

    # ---- 汇总数字
    n_det = sum(d["n_events"] for d in det)
    n_null = sum(d.get("n_null_events", 0) for d in det)
    coloc_summary = []
    for p in coloc["pairs"]:
        for sgn in ("+dir", "-dir"):
            t = p["tests"].get(sgn, {})
            if "p_two_perm" in t:
                coloc_summary.append({
                    "day": p["day"], "dt_h": p["dt_h"], "sign": sgn,
                    "n_matched": t["n_matched"],
                    "obs_mean_cm": round(t["obs_mean_m"] * 100, 2),
                    "p_two": t["p_two_perm"]})

    conclusion = {
        "question": "SWOT 不依赖 SAR 标签的内波独立检测在现有数据上是否可行",
        "snr_budget": {
            "expected_signal_cm": list(SIGNAL_CM),
            "unsmoothed_250m": {
                "pixel_noise_cm": [round(min(pix_noise), 2),
                                   round(max(pix_noise), 2)],
                "band_0.7_1.2km_bg_cm": [round(np.nanmin(band0712), 2),
                                         round(np.nanmax(band0712), 2)],
                "resid_total_rms_cm": [round(min(resid_rms), 2),
                                       round(max(resid_rms), 2)],
                "note": "带内(0.4-2.5km)背景+噪声 0.3-0.9cm，低于预期信号 "
                        "2-10cm；账面 SNR>1。但 2.5-10km 海洋背景 RMS 达 "
                        "2.6-5.6cm，必须靠方向匹配滤波+沿峰相干平均抑制。",
            },
            "expert_2km": {
                "note": "2km 格网 Nyquist=4km，ISW 波长 0.4-2km 为亚奈奎斯特，"
                        "单波信号被产品平滑物理性抹除；仅剩波包包络级可检性。",
            },
        },
        "collocation_tests": {
            "note": "事件级覆盖普查(step0)后仅 3 对可检：Δt=1.06h(n=3-4 匹配)、"
                    "18.8h(n=29-43)、23.2h(n=5-25)。全部按 c=0.92m/s±10% "
                    "沿 direction(180°模糊两假设)平移。",
            "results": coloc_summary,
            "verdict": "全部不显著（p=0.32~0.73，两符号、c±10% 敏感性同）。"
                       "9-30 近同时个例中 +12~+25cm 的邻域均值实为刈幅边缘"
                       "大尺度梯度残差，非波包信号（见图 fig_step3b）。",
        },
        "detector_prototype": {
            "method": "12 方向楔形带通(0.4-2.5km) + 沿峰 2km 相干平滑，"
                      "统计量 C×A，相位扰乱零分布 p99.9 阈值，刈幅边缘内缩",
            "n_detections": n_det,
            "n_null_same_threshold": n_null,
            "enrichment": round(n_det / max(n_null, 1), 1),
            "climatology": [d["climatology_check"] for d in det],
            "verdict": "检出数约为相位扰乱零假设的 11 倍（218 vs 19），说明"
                       "检出的多为真实空间相干结构而非仪器噪声；但相干结构≠内波"
                       "（08-27 刈幅检出 103 个，位于内潮活跃的吕宋西侧，内潮"
                       "波束是最可能混淆源；锋面丝带、残差背景坡折亦可产生）。"
                       "与 SAR 事件热区的气候学一致性：2/9 刈幅 p<0.05"
                       "（08-01_right p=0.025、08-20_right p≈0），但经 9 重"
                       "比较校正（α=0.0056）后不显著，且 08-27_left 等反而显著"
                       "偏离热区。不能宣称为内波独立检出。",
        },
        "conclusion": {
            "expert_2km": "不能检：分辨率物理上不够 + 3 对实证检验全部为零。",
            "unsmoothed_250m": "账面信噪比可行（带内背景 0.5-0.7cm << 信号 "
                               "2-10cm），但现有 5 个 granule 与 1474 个 SAR "
                               "事件零重叠，无法事件级验证；检测器原型产出"
                               "候选但无法证真。",
            "what_is_needed": [
                "下载与事件日几何匹配的 Unsmoothed granule：首选 pass "
                "004_243(2023-09-30, Δt=1.06h) 与 574_008(2023-07-06, "
                "Δt=1.86h)——全档案仅有的'小Δt+刈幅覆盖事件群'几何；",
                "或下载 HR Basic 250m 产品（噪声更低、带地球物理修正）；",
                "配对时按 c≈0.9m/s 沿 direction±180° 双假设平移，"
                "用方向匹配滤波检测器（本原型）扫刈幅，以 5km 共定位率验收；",
                "若做独立检测器用于 SAR 可视性偏差标定：需先在上述真值配对上"
                "标定检测率-风速关系，再业务化。",
            ],
            "feasible_overall": "有条件可行：250m 产品信噪比账面支持，"
                                "但需补充几何匹配的 granule 才能完成验证闭环。",
        },
    }
    with open(OUT / "conclusion.json", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=2)
    print(f"→ {OUT}/conclusion.json, fig_step5_snr_budget.png")
    print(json.dumps(conclusion["conclusion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
