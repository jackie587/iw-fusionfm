# -*- coding: utf-8 -*-
"""
Paper A-line step 2: visibility-corrected monthly ISW occurrence.

Idea
----
The hard 2-10 m/s wind window is known to be a soft boundary (D11C had 13 TP
at ~1.4 m/s representative wind; the 2023-10-12 pair at ~9.9 m/s had none).
We therefore fit an empirical soft visibility function

    v(w) = sigmoid((w - a_lo)/s_lo) * sigmoid((a_hi - w)/s_hi)

to the warm-season scenes (months 6-10), where the occurrence rate rho is
assumed constant:  P(has_wave=1 | w) = rho * v(w).  Both edges and rho are
fitted jointly by maximum likelihood; uncertainty by case-bootstrap.

Monthly occurrence rates r_m are computed on CENSUS scenes only (source in
{census, both}; event_db-only scenes are wave-positive by selection and would
bias the rate upward).  The visibility correction divides by the mean
visibility of that month, vbar_m = mean_t v(w(t)), where w(t) is the ERA5
hourly AOI-mean 10 m wind.  Two vbar variants are reported: all hours of the
month, and only hours matching typical S1 acquisition times (10 and 22 UTC).

    rho_hat_m = r_m / vbar_m      (Clopper-Pearson bounds divided by vbar)

Outputs
-------
results/figures/visibility_correction.csv   monthly table
results/figures/visibility_correction.json  params + bootstrap CI + meta

Run from code/ directory:
    python evaluation/visibility_correction.py
"""
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # project root

TABLE_CSV = os.path.join(ROOT, "results", "figures", "scene_visibility_table.csv")
ERA5_DIR = os.path.join(ROOT, "data", "raw", "auxiliary", "era5")
OUT_CSV = os.path.join(ROOT, "results", "figures", "visibility_correction.csv")
OUT_JSON = os.path.join(ROOT, "results", "figures", "visibility_correction.json")

# ERA5 AOI used for the monthly wind climatology: covers the two event
# clusters (west of Hainan ~108E and the NE SCS shelf 110-115E).
AOI = dict(lon=(107.0, 115.0), lat=(18.0, 22.0))
ACQ_HOURS = (10, 22)      # typical S1 acquisition UTC hours in this AOI
WARM_MONTHS = (6, 7, 8, 9, 10)
N_BOOT = 500
RNG = np.random.default_rng(20260909)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def visibility(w, a_lo, s_lo, a_hi, s_hi):
    return sigmoid((w - a_lo) / s_lo) * sigmoid((a_hi - w) / s_hi)


def neg_loglik(theta, w, y):
    rho, a_lo, s_lo, a_hi, s_hi = theta
    p = rho * visibility(w, a_lo, s_lo, a_hi, s_hi)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_visibility(w, y):
    """Fit rho * v(w) on warm-season scenes. Returns (theta, success)."""
    bounds = [(0.3, 1.0), (0.0, 4.0), (0.2, 3.0), (7.0, 16.0), (0.2, 4.0)]
    best = None
    for x0 in ([0.9, 2.0, 1.0, 10.0, 1.0],
               [0.8, 1.5, 0.5, 11.0, 1.5],
               [0.95, 2.5, 1.5, 9.5, 0.6]):
        res = minimize(neg_loglik, x0, args=(w, y), bounds=bounds,
                       method="L-BFGS-B")
        if res.success and (best is None or res.fun < best.fun):
            best = res
    return (best.x if best is not None else None,
            bool(best is not None and best.success))


def era5_monthly_visibility(theta):
    """Monthly mean visibility from ERA5 hourly AOI-mean wind."""
    out = {}
    for fname in sorted(os.listdir(ERA5_DIR)):
        if not (fname.startswith("era5_wind10m_") and fname.endswith(".nc")):
            continue
        yyyymm = fname.replace("era5_wind10m_", "").replace(".nc", "")
        month = int(yyyymm[4:6])
        ds = xr.open_dataset(os.path.join(ERA5_DIR, fname))
        lat = ds["latitude"].values
        lo, hi = (AOI["lat"][0], AOI["lat"][1]) if lat[0] < lat[-1] \
            else (AOI["lat"][1], AOI["lat"][0])
        sub = ds.sel(longitude=slice(*AOI["lon"]), latitude=slice(lo, hi))
        spd = np.hypot(sub["u10"], sub["v10"]).mean(dim=("latitude", "longitude"))
        w = spd.values.astype(float)
        hours = pd.to_datetime(spd["valid_time"].values).hour.values
        v_all = visibility(w, *theta[1:])
        v_acq = v_all[np.isin(hours, ACQ_HOURS)]
        out[month] = dict(
            vbar_all=float(v_all.mean()),
            vbar_acq=float(v_acq.mean()),
            wind_mean=float(w.mean()),
            frac_in_hard_window=float(((w >= 2.0) & (w <= 10.0)).mean()),
            n_hours=int(len(w)),
        )
        ds.close()
    return out


def cp_bounds(x, n, alpha=0.05):
    """Clopper-Pearson two-sided (1-alpha) interval for x/n."""
    lo = 0.0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - alpha / 2, x + 1, n - x)
    return float(lo), float(hi)


def rho_profile(scenes_w, scenes_y, theta, grid=None):
    """ML estimate + profile-likelihood 95% CI for monthly occurrence rho_m.

    Model: y_i ~ Bernoulli(rho * v(w_i)); each scene is one trial whose
    detection efficiency is v(w_i).  With all-zero outcomes the upper bound
    solves prod(1 - rho*v_i) = 0.05, an exact small-sample bound under the
    model (replaces the cruder CP(r)/vbar division).
    """
    if grid is None:
        grid = np.linspace(1e-4, 1.0, 2000)
    v = visibility(np.asarray(scenes_w, float), *theta[1:])
    y = np.asarray(scenes_y, float)
    ll = np.array([np.sum(y * np.log(np.clip(r * v, 1e-12, 1)) +
                          (1 - y) * np.log(np.clip(1 - r * v, 1e-12, 1)))
                   for r in grid])
    i_best = int(np.argmax(ll))
    rho_hat = float(grid[i_best])
    cutoff = ll[i_best] - 3.841 / 2.0      # chi2(1) 95%
    ok = ll >= cutoff
    lo = float(grid[ok].min()) if ok.any() else 0.0
    hi = float(grid[ok].max()) if ok.any() else 1.0
    return rho_hat, lo, hi


def main():
    df = pd.read_csv(TABLE_CSV)
    warm = df[df.month.isin(WARM_MONTHS)].reset_index(drop=True)
    w = warm.wind_ms.to_numpy(float)
    y = warm.has_wave.astype(int).to_numpy()
    print(f"warm-season scenes for v(w) fit: n={len(warm)}, "
          f"wave={int(y.sum())}, no-wave={int((1 - y).sum())}")

    theta, ok = fit_visibility(w, y)
    if not ok:
        raise RuntimeError("visibility fit failed")
    rho, a_lo, s_lo, a_hi, s_hi = theta
    print(f"fit: rho={rho:.3f}, a_lo={a_lo:.2f} (s={s_lo:.2f}), "
          f"a_hi={a_hi:.2f} (s={s_hi:.2f})")

    # bootstrap the fit
    boot = []
    for _ in range(N_BOOT):
        idx = RNG.integers(0, len(w), len(w))
        th, ok_b = fit_visibility(w[idx], y[idx])
        if ok_b:
            boot.append(th)
    boot = np.array(boot)
    boot_ci = np.percentile(boot, [2.5, 50, 97.5], axis=0)
    print(f"bootstrap converged: {len(boot)}/{N_BOOT}")
    for name, ci in zip(["rho", "a_lo", "s_lo", "a_hi", "s_hi"], boot_ci.T):
        print(f"  {name:>5}: {ci[1]:.2f} [{ci[0]:.2f}, {ci[2]:.2f}]")

    vis_month = era5_monthly_visibility(theta)

    # monthly occurrence on census scenes (source census/both), all scenes
    # regardless of hard-window flag; per-scene detection efficiency v(w_i)
    # enters the Bernoulli model directly (rho_profile).
    cen = df[df.source.isin(["census", "both"])]
    rows = []
    for m in range(1, 13):
        sub = cen[cen.month == m]
        n, x = len(sub), int(sub.has_wave.sum())
        if n == 0 or m not in vis_month:
            continue
        r_lo, r_hi = cp_bounds(x, n)
        vb = vis_month[m]
        rho_hat, rho_lo, rho_hi = rho_profile(
            sub.wind_ms.to_numpy(float),
            sub.has_wave.astype(int).to_numpy(), theta)
        rows.append(dict(
            month=m, n_scenes=n, n_wave=x, rate=x / n,
            rate_cp_lo=r_lo, rate_cp_hi=r_hi,
            vbar_acq=vb["vbar_acq"], vbar_all=vb["vbar_all"],
            frac_hours_hard_window=vb["frac_in_hard_window"],
            rho_hat=rho_hat, rho_pl_lo=rho_lo, rho_pl_hi=rho_hi,
        ))
        print(f"month {m:>2}: census {x}/{n} rate={x/n:.2f} "
              f"[{r_lo:.2f},{r_hi:.2f}], vbar_acq={vb['vbar_acq']:.2f} "
              f"-> rho={rho_hat:.2f} [{rho_lo:.2f},{rho_hi:.2f}]")

    # pooled winter (Nov-Mar) bound under the same model
    win = cen[cen.month.isin((11, 12, 1, 2, 3))]
    w_hat, w_lo, w_hi = rho_profile(
        win.wind_ms.to_numpy(float),
        win.has_wave.astype(int).to_numpy(), theta)
    print(f"\nwinter pooled (n={len(win)}, all zero): rho_hat={w_hat:.3f}, "
          f"95% upper={w_hi:.3f}")
    # propagate v-fit bootstrap uncertainty into the winter upper bound
    win_uppers = []
    for th in boot:
        _, _, hi_b = rho_profile(win.wind_ms.to_numpy(float),
                                 win.has_wave.astype(int).to_numpy(), th)
        win_uppers.append(hi_b)
    win_uppers = np.array(win_uppers)
    print(f"winter upper bound with v bootstrap: "
          f"median {np.median(win_uppers):.3f}, "
          f"[{np.percentile(win_uppers, 2.5):.3f}, "
          f"{np.percentile(win_uppers, 97.5):.3f}]")
    winter_pooled = dict(n_scenes=int(len(win)), rho_hat=w_hat,
                         rho_pl_lo=w_lo, rho_pl_hi=w_hi,
                         upper95_boot_median=float(np.median(win_uppers)),
                         upper95_boot_ci95=[float(np.percentile(win_uppers, 2.5)),
                                            float(np.percentile(win_uppers, 97.5))])

    # jackknife sensitivity: the fitted upper edge a_hi is driven mainly by
    # the 2023-10-12 no-wave pair (9.9 m/s); report the winter upper bound
    # when that pair (or all of October) is excluded from the v(w) fit.
    jackknife = {}
    for tag, mask in [
        ("drop_20231012_pair", ~(warm.date == 20231012)),
        ("drop_all_october", warm.month != 10),
    ]:
        th_j, ok_j = fit_visibility(
            warm.wind_ms.to_numpy(float)[np.asarray(mask)],
            warm.has_wave.astype(int).to_numpy()[np.asarray(mask)])
        if ok_j:
            _, _, hi_j = rho_profile(win.wind_ms.to_numpy(float),
                                     win.has_wave.astype(int).to_numpy(), th_j)
            jackknife[tag] = dict(
                params=dict(rho=float(th_j[0]), a_lo=float(th_j[1]),
                            a_hi=float(th_j[3])),
                winter_upper95=float(hi_j))
            print(f"jackknife {tag}: a_hi={th_j[3]:.2f}, "
                  f"winter upper={hi_j:.3f}")
    winter_pooled["jackknife"] = jackknife

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)

    meta = dict(
        model="P(has_wave|w) = rho * sig((w-a_lo)/s_lo) * sig((a_hi-w)/s_hi)",
        fit_scenes="warm-season (Jun-Oct), all sources, QA-corrected has_wave",
        params=dict(rho=float(rho), a_lo=float(a_lo), s_lo=float(s_lo),
                    a_hi=float(a_hi), s_hi=float(s_hi)),
        bootstrap=dict(n_requested=N_BOOT, n_converged=int(len(boot)),
                       ci95={name: [float(c[0]), float(c[2])]
                             for name, c in zip(
                                 ["rho", "a_lo", "s_lo", "a_hi", "s_hi"],
                                 boot_ci.T)}),
        era5_aoi=AOI, acquisition_hours_utc=list(ACQ_HOURS),
        monthly_visibility=vis_month,
        caveats=[
            "r_m uses census/both scenes only (event_db-only scenes are "
            "wave-positive by selection).",
            "rho constant across warm months assumed; June may violate "
            "(stratification not yet established).",
            "vbar evaluated on ERA5 AOI-mean wind; sub-mesoscale lulls are "
            "not resolved by ERA5 (0.25 deg).",
            "rho_hat / CI from Bernoulli profile likelihood with per-scene "
            "efficiency v(w_i); winter pooled upper additionally bootstrapped "
            "over the v(w) fit.",
        ],
    )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, monthly=rows, winter_pooled=winter_pooled),
                  f, ensure_ascii=False, indent=2)
    print(f"\nsaved: {OUT_CSV}\nsaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
