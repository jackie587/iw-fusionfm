"""Argo 温盐剖面 → 南海北部 AOI 逐月层结参数（供 KdV/eKdV 两层反演）。

链路：GDAC 全球剖面索引（ar_index_global_prof.txt.gz）→ 按经纬度筛 AOI
→ 逐个下载 nc 剖面文件（幂等：已存在且非空则跳过）→ QC=1 的 P/T/S
→ TEOS-10（gsw）位密 σ0、N² → 逐剖面层结参数 → 逐月气候态汇总 + 2023/2024 单年逐月。

数据源（按可用性排序，--base-url 可覆盖）：
  1. AWS S3 GDAC 镜像 https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub
     （实测 2026-09 可用，索引每日更新）
  2. Ifremer GDAC https://data-argo.ifremer.fr （本机实测 443 不可达）
  3. USGODAE https://usgodae.org/ftp/outgoing/argo
     （索引可下但 2024-08 起损坏且陈旧，dac 树不在该前缀下，仅兜底）

计算口径（与 inversion/physics.py 两层 KdV 输入对齐）：
  - 位密 σ0 = gsw.sigma0(SA, CT)，参考面 0 dbar；密度 ρ = σ0 + 1000；
  - 跃层深度 z_tc：σ0 垂向梯度最大处的深度（同时存 N² 最大处 z_tc_n2）；
    搜索范围限制在 20 m 以深，避开近表层噪声；
  - 混合层深 MLD：σ0 相对 10 m 参考值增加 0.03 kg/m³ 的深度
    （de Boyer Montégut 2004 阈值口径）；
  - 两层化：h1 = z_tc（海面→跃层的上层厚度，混合层在其内）；
    ρ1 = [0, z_tc] 层平均密度；ρ2 = [z_tc, min(z_tc+300 m, 剖面底)] 层平均；
    Δρ = ρ2 − ρ1；g' = g·Δρ/ρ0，ρ0 = 1025 kg/m³，g = 9.81 m/s²；
  - 剖面质量门槛：QC=1 有效层数 ≥ 20、最浅 ≤ 20 dbar、最深 ≥ 200 dbar，
    且 20~200 m 内 σ0 跨距 ≥ 0.5 kg/m³（确保确有跃层）。

产物：
  data/raw/auxiliary/argo/            索引 + nc（幂等缓存）
  results/stratification/argo_profiles_aoi.csv            逐剖面参数
  results/stratification/stratification_monthly.csv       逐月汇总（clim/2023/2024）
  results/stratification/stratification_monthly.json      meta + 逐月记录
  results/figures/argo_profile_map.png                    剖面位置分布
  results/figures/argo_monthly_thermocline.png            逐月跃层深度
  results/figures/argo_monthly_gprime.png                 逐月 g' / Δρ

用法（项目根目录）：
    python code/inversion/argo_stratification.py [--max-download N] [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

import gsw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data/raw/auxiliary/argo"
INDEX_GZ = RAW_DIR / "ar_index_global_prof.txt.gz"
AOI_INDEX = RAW_DIR / "argo_aoi_index.csv"
NC_ROOT = RAW_DIR / "profiles"
OUT_DIR = PROJECT_ROOT / "results/stratification"
FIG_DIR = PROJECT_ROOT / "results/figures"

BASE_URLS = [
    "https://argo-gdac-sandbox.s3.eu-west-3.amazonaws.com/pub",
    "https://data-argo.ifremer.fr",
]
INDEX_KEY = "idx/ar_index_global_prof.txt.gz"  # S3 布局；ifremer 在根目录

G = 9.81
RHO0 = 1025.0
AOI = {"lon_min": 109.0, "lon_max": 116.0, "lat_min": 18.0, "lat_max": 22.5}

MIN_LEVELS = 20
SHALLOW_MAX_DBAR = 20.0
DEEP_MIN_DBAR = 200.0
MIN_SIGMA_SPAN = 0.5      # kg/m³，20~200 m 内
TC_SEARCH_MIN_M = 20.0    # 跃层搜索下限
MLD_DSIGMA = 0.03         # kg/m³，MLD 阈值
LOWER_LAYER_M = 300.0     # 下层平均厚度上限


# ---------------------------------------------------------------- 数据获取

def fetch(url: str, dest: Path, retries: int = 3, timeout: int = 120) -> bool:
    """下载到临时文件再改名（避免半成品）；已存在且非空则跳过。"""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, \
                    open(tmp, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp.rename(dest)
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  下载失败({attempt + 1}/{retries}) {url}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return False


def ensure_index(base_url: str) -> None:
    if INDEX_GZ.exists() and INDEX_GZ.stat().st_size > 0:
        print(f"索引已存在：{INDEX_GZ}")
        return
    key = INDEX_KEY if "s3" in base_url else INDEX_KEY.split("/", 1)[1]
    url = f"{base_url}/{key}"
    print(f"下载全球剖面索引：{url}")
    if not fetch(url, INDEX_GZ, retries=3, timeout=600):
        raise RuntimeError("索引下载失败，可换 --base-url 重试")
    with gzip.open(INDEX_GZ, "rb") as f:
        head = f.read(200)
    if not head.startswith(b"# Title"):
        INDEX_GZ.unlink()
        raise RuntimeError("索引文件校验失败（非 GDAC 索引格式），已删除")


def load_aoi_index() -> pd.DataFrame:
    """全球索引 → AOI 子集（缓存为 argo_aoi_index.csv）。"""
    if AOI_INDEX.exists():
        df = pd.read_csv(AOI_INDEX, parse_dates=["date"])
    else:
        cols = ["file", "date", "latitude", "longitude", "ocean",
                "profiler_type", "institution", "date_update"]
        rows = []
        with gzip.open(INDEX_GZ, "rt") as f:
            for line in f:
                if line.startswith("#") or line.startswith("file,"):
                    continue
                parts = line.rstrip("\n").split(",")
                if len(parts) != len(cols):
                    continue
                try:
                    lat, lon = float(parts[2]), float(parts[3])
                except ValueError:
                    continue
                if (AOI["lat_min"] <= lat <= AOI["lat_max"]
                        and AOI["lon_min"] <= lon <= AOI["lon_max"]):
                    rows.append(parts)
        df = pd.DataFrame(rows, columns=cols)
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S",
                                    errors="coerce")
        df.to_csv(AOI_INDEX, index=False)
    df = df.dropna(subset=["date"])
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    print(f"AOI 内索引剖面 {len(df)} 条，"
          f"时间 {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    return df


def download_profiles(df: pd.DataFrame, base_url: str, workers: int,
                      max_download: int | None) -> None:
    prefix = "dac" if "s3" in base_url else ""
    todo = []
    for rel in df["file"]:
        dest = NC_ROOT / rel
        if not (dest.exists() and dest.stat().st_size > 0):
            todo.append(rel)
    if max_download:
        todo = todo[:max_download]
    print(f"需下载 {len(todo)} 个剖面文件（已有 {len(df) - len(todo)} 个）")
    if not todo:
        return

    def one(rel: str) -> bool:
        dest = NC_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/{prefix + '/' if prefix else ''}{rel}"
        return fetch(url, dest, retries=3, timeout=120)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, rel): rel for rel in todo}
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if (ok + fail) % 200 == 0:
                print(f"  进度 {ok + fail}/{len(todo)}（失败 {fail}）")
    print(f"下载完成：成功 {ok}，失败 {fail}")


# ---------------------------------------------------------------- 层结计算

def _char_qc(arr) -> np.ndarray:
    return np.asarray(arr).astype("U1")


def _get_var(ds, name):
    if name not in ds.variables:
        return None
    v = ds.variables[name][:]
    arr = np.ma.filled(v, np.nan).astype(float)
    return arr


def _pick(ds, base: str):
    """取 {base}_ADJUSTED（含 QC）→ 退回原始 {base}（含 QC）。

    ADJUSTED 可能整体为缺测（如部分 D 文件未调整压力）；QC 变量缺失时
    该候选不可用。返回 (data[n_prof, n_lev], qc 同形) 或 (None, None)。
    """
    for name in (f"{base}_ADJUSTED", base):
        data = _get_var(ds, name)
        qc_name = f"{name}_QC"
        if data is None or data.size == 0 or qc_name not in ds.variables:
            continue
        if not np.isfinite(data).any():
            continue
        return data, _char_qc(ds.variables[qc_name])
    return None, None


def extract_profile(ds: Dataset, iprof: int) -> dict | None:
    """单条剖面 → 层结参数；不合格返回 None。"""
    lat = float(np.ma.filled(ds.variables["LATITUDE"][iprof], np.nan))
    lon = float(np.ma.filled(ds.variables["LONGITUDE"][iprof], np.nan))
    pres, qc_p = _pick(ds, "PRES")
    temp, qc_t = _pick(ds, "TEMP")
    salt, qc_s = _pick(ds, "PSAL")
    if pres is None or temp is None or salt is None:
        return None

    p, t, s = pres[iprof], temp[iprof], salt[iprof]
    good = ((qc_p[iprof] == "1") & (qc_t[iprof] == "1") & (qc_s[iprof] == "1")
            & np.isfinite(p) & np.isfinite(t) & np.isfinite(s))
    if good.sum() < MIN_LEVELS:
        return None
    p, t, s = p[good], t[good], s[good]
    order = np.argsort(p)
    p, t, s = p[order], t[order], s[order]
    keep = np.r_[True, np.diff(p) > 0.5]  # 去掉过密/重复层
    p, t, s = p[keep], t[keep], s[keep]

    if len(p) < MIN_LEVELS or p[0] > SHALLOW_MAX_DBAR or p[-1] < DEEP_MIN_DBAR:
        return None

    sa = gsw.SA_from_SP(s, p, lon, lat)
    ct = gsw.CT_from_t(sa, t, p)
    sigma0 = gsw.sigma0(sa, ct)
    rho = sigma0 + 1000.0
    z = -gsw.z_from_p(p, lat)  # 深度，正值向下

    mid = (z[:-1] + z[1:]) / 2
    mask_span = (z >= 20) & (z <= 200)
    if not mask_span.any() or \
            np.nanmax(sigma0[mask_span]) - np.nanmin(sigma0[mask_span]) \
            < MIN_SIGMA_SPAN:
        return None

    # 跃层：σ0 梯度最大（20 m 以深）
    grad = np.diff(sigma0) / np.diff(z)
    valid = np.isfinite(grad) & (mid >= TC_SEARCH_MIN_M)
    if not valid.any():
        return None
    i_tc = np.where(valid)[0][np.argmax(grad[valid])]
    z_tc = float(mid[i_tc])

    # N² 最大处
    n2, p_mid = gsw.Nsquared(sa, ct, p, lat)
    z_mid = -gsw.z_from_p(p_mid, lat)
    ok2 = np.isfinite(n2) & (n2 > 0) & (z_mid >= TC_SEARCH_MIN_M)
    z_tc_n2 = float(z_mid[ok2][np.argmax(n2[ok2])]) if ok2.any() else np.nan
    n2_max = float(np.max(n2[ok2])) if ok2.any() else np.nan

    # MLD：相对 10 m 参考密度 +0.03 kg/m³
    i_ref = int(np.argmin(np.abs(z - 10.0)))
    d_sig = sigma0 - sigma0[i_ref]
    below = np.where((z > z[i_ref]) & (d_sig >= MLD_DSIGMA))[0]
    mld = float(z[below[0]]) if len(below) else float(z[-1])

    # 两层化
    h1 = z_tc
    up = z <= z_tc
    low = (z > z_tc) & (z <= z_tc + LOWER_LAYER_M)
    if up.sum() < 3 or low.sum() < 3:
        return None
    rho1 = float(np.mean(rho[up]))
    rho2 = float(np.mean(rho[low]))
    drho = rho2 - rho1
    if drho <= 0:
        return None
    gprime = G * drho / RHO0

    return {
        "lat": lat, "lon": lon,
        "n_levels": int(len(p)), "z_max_m": float(z[-1]),
        "mld_m": mld,
        "z_tc_grad_m": z_tc, "z_tc_n2_m": z_tc_n2, "n2_max_s-2": n2_max,
        "rho1": rho1, "rho2": rho2, "delta_rho": drho,
        "h1_m": h1, "gprime": gprime,
    }


def process_all(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    n_missing = n_bad = 0
    for _, row in df.iterrows():
        path = NC_ROOT / row["file"]
        if not (path.exists() and path.stat().st_size > 0):
            n_missing += 1
            continue
        try:
            with Dataset(path) as ds:
                n_prof = ds.dimensions["N_PROF"].size
                for iprof in range(n_prof):
                    rec = extract_profile(ds, iprof)
                    if rec is None:
                        continue
                    rec.update({
                        "file": row["file"], "iprof": iprof,
                        "date": row["date"], "mode": Path(row["file"]).name[0],
                        "institution": row["institution"],
                    })
                    records.append(rec)
        except Exception as e:  # noqa: BLE001
            n_bad += 1
            print(f"  解析失败 {row['file']}: {e}", file=sys.stderr)
    print(f"合格剖面 {len(records)} 条；文件缺失 {n_missing}，解析失败 {n_bad}")
    out = pd.DataFrame(records)
    if not out.empty:
        out["year"] = out["date"].dt.year
        out["month"] = out["date"].dt.month
        out["period"] = np.where(out["year"] == 2023, "2023",
                                 np.where(out["year"] == 2024, "2024", "other"))
    return out


# ---------------------------------------------------------------- 汇总与图

FIELDS = ["z_tc_grad_m", "z_tc_n2_m", "mld_m", "delta_rho", "gprime", "h1_m"]


def monthly_summary(prof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, sub in [("clim", prof),
                        ("2023", prof[prof.year == 2023]),
                        ("2024", prof[prof.year == 2024])]:
        for month in range(1, 13):
            g = sub[sub.month == month]
            row = {"period": period, "month": month, "n": int(len(g))}
            for f in FIELDS:
                row[f"{f}_mean"] = float(g[f].mean()) if len(g) else np.nan
                row[f"{f}_std"] = float(g[f].std()) if len(g) > 1 else np.nan
                row[f"{f}_median"] = float(g[f].median()) if len(g) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def make_figures(prof: pd.DataFrame, mon: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    clim = mon[mon.period == "clim"].set_index("month")

    # 1. 剖面位置分布
    fig, ax = plt.subplots(figsize=(8, 6))
    old = prof[~prof.year.isin([2023, 2024])]
    new = prof[prof.year.isin([2023, 2024])]
    ax.scatter(old.lon, old.lat, s=3, c="0.6", alpha=0.4,
               label=f"other years (n={len(old)})")
    if len(new):
        ax.scatter(new.lon, new.lat, s=5, c="tab:red", alpha=0.7,
                   label=f"2023-2024 (n={len(new)})")
    ax.add_patch(plt.Rectangle(
        (AOI["lon_min"], AOI["lat_min"]),
        AOI["lon_max"] - AOI["lon_min"], AOI["lat_max"] - AOI["lat_min"],
        fill=False, ec="k", lw=1.2))
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"Argo profiles in AOI ({len(prof)} total, "
                 f"{prof.date.min():%Y-%m} ~ {prof.date.max():%Y-%m})")
    ax.legend(); ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(); fig.savefig(FIG_DIR / "argo_profile_map.png", dpi=150)
    plt.close(fig)

    # 2. 逐月跃层深度
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(clim.index, clim["z_tc_grad_m_mean"],
                yerr=clim["z_tc_grad_m_std"], fmt="o-", capsize=3,
                label="climatology (mean±std)")
    ax.plot(clim.index, clim["z_tc_n2_m_mean"], "s--", ms=4,
            label="climatology, $N^2$-max depth")
    for yr, c in [("2023", "tab:green"), ("2024", "tab:orange")]:
        sub = mon[(mon.period == yr) & (mon.n > 0)].set_index("month")
        if len(sub):
            ax.plot(sub.index, sub["z_tc_grad_m_mean"], "o", ms=5, color=c,
                    alpha=0.8, label=f"{yr} (n=" +
                    ",".join(str(int(n)) for n in sub["n"]) + ")")
    ax2 = ax.twinx()
    ax2.bar(clim.index, clim["n"], width=0.5, color="0.85", zorder=0)
    ax2.set_ylabel("sample count"); ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 13)); ax.set_xlabel("Month")
    ax.set_ylabel("Thermocline depth (m)")
    ax.set_title("Monthly thermocline depth, northern SCS AOI")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "argo_monthly_thermocline.png", dpi=150)
    plt.close(fig)

    # 3. 逐月 g' 与 Δρ
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, fld, lab in [(axes[0], "gprime", "reduced gravity g' (m/s²)"),
                         (axes[1], "delta_rho", "Δρ (kg/m³)")]:
        ax.errorbar(clim.index, clim[f"{fld}_mean"], yerr=clim[f"{fld}_std"],
                    fmt="o-", capsize=3, label="climatology")
        for yr, c in [("2023", "tab:green"), ("2024", "tab:orange")]:
            sub = mon[(mon.period == yr) & (mon.n > 0)].set_index("month")
            if len(sub):
                ax.plot(sub.index, sub[f"{fld}_mean"], "o", ms=5, color=c,
                        alpha=0.8, label=yr)
        ax.set_xticks(range(1, 13)); ax.set_xlabel("Month")
        ax.set_ylabel(lab); ax.legend(fontsize=8)
    axes[0].set_title("Monthly two-layer stratification, northern SCS AOI")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "argo_monthly_gprime.png", dpi=150)
    plt.close(fig)
    print(f"图已写入 {FIG_DIR}")


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=None, help="GDAC 根（默认自动尝试）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-download", type=int, default=None,
                    help="调试用：最多下载 N 个剖面")
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NC_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_url = args.base_url or BASE_URLS[0]
    ensure_index(base_url)
    df = load_aoi_index()
    if not args.skip_download:
        download_profiles(df, base_url, args.workers, args.max_download)

    prof = process_all(df)
    if prof.empty:
        raise RuntimeError("无合格剖面，检查下载与 QC 门槛")
    prof.to_csv(OUT_DIR / "argo_profiles_aoi.csv", index=False)

    mon = monthly_summary(prof)
    mon.to_csv(OUT_DIR / "stratification_monthly.csv", index=False)

    meta = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "aoi": AOI,
        "source": base_url,
        "index_file": str(INDEX_GZ.relative_to(PROJECT_ROOT)),
        "definitions": {
            "sigma0": "gsw.sigma0 (TEOS-10, ref 0 dbar); rho = sigma0+1000",
            "z_tc_grad_m": "depth of max d(sigma0)/dz below 20 m",
            "z_tc_n2_m": "depth of max N2 (gsw.Nsquared) below 20 m",
            "mld_m": "sigma0 +0.03 kg/m3 over 10 m reference",
            "h1_m": "= z_tc_grad_m (sea surface to thermocline)",
            "rho1/rho2": "mean density above / [z_tc, z_tc+300 m] below",
            "gprime": "g*(rho2-rho1)/1025",
            "qc": "only P/T/S with QC=1; prefer ADJUSTED fields",
            "profile_gate": f"n_levels>={MIN_LEVELS}, top<={SHALLOW_MAX_DBAR} dbar, "
                            f"bottom>={DEEP_MIN_DBAR} dbar, "
                            f"sigma span(20-200m)>={MIN_SIGMA_SPAN}",
        },
        "n_profiles_index_aoi": int(len(df)),
        "n_profiles_valid": int(len(prof)),
        "time_span": [str(prof.date.min().date()), str(prof.date.max().date())],
        "monthly": json.loads(mon.to_json(orient="records")),
    }
    with open(OUT_DIR / "stratification_monthly.json", "w",
              encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    make_figures(prof, mon)
    print(f"→ {OUT_DIR}/argo_profiles_aoi.csv, stratification_monthly.{{csv,json}}")


if __name__ == "__main__":
    main()
