# -*- coding: utf-8 -*-
"""
Paper A-line: scene-level unified visibility table + seasonal visibility figure.

Merges two sources into one scene-level table:
  1. results/scene_eval_poscensus/wind_window_analysis.json  (census scenes)
  2. results/event_database/events.csv                       (event database scenes)
then applies a QA correction: event-db-backed scenes whose sampled events in
qa_judgments_merged_v2db.csv are all FP (TP+UNCERTAIN == 0) are flipped to
no-wave and flagged in the qa_note column.

Outputs:
  results/figures/scene_visibility_table.csv
  results/figures/fig_seasonal_visibility.png
and prints a cross-check table to stdout.

Run from code/ directory:
    python evaluation/plot_seasonal_visibility.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta, fisher_exact

WIND_LO, WIND_HI = 2.0, 10.0  # must match code/inference/wind_filter.py

HERE = os.path.dirname(os.path.abspath(__file__))          # code/evaluation
CODE_DIR = os.path.dirname(HERE)                           # code/
ROOT = os.path.dirname(CODE_DIR)                           # project root

CENSUS_JSON = os.path.join(ROOT, "results", "scene_eval_poscensus", "wind_window_analysis.json")
EVENTS_CSV = os.path.join(ROOT, "results", "event_database", "events.csv")
QA_CSV = os.path.join(ROOT, "results", "event_database", "qa_sheets",
                      "qa_judgments_merged_v2db.csv")
OUT_CSV = os.path.join(ROOT, "results", "figures", "scene_visibility_table.csv")
OUT_PNG = os.path.join(ROOT, "results", "figures", "fig_seasonal_visibility.png")

OVERLAP_CODES = {"178C", "E318", "55E0", "E4CD", "D11C", "7E00"}


def season_of(month):
    if month in (11, 12, 1, 2, 3):
        return "winter"
    if month == 4:
        return "transitional"
    return "summer"


def scene_code(scene_name):
    """Scene tail code = last 4 chars; strip a trailing '_COG' first."""
    name = scene_name
    if name.endswith("_COG"):
        name = name[: -len("_COG")]
    return name[-4:]


def build_table():
    census = pd.DataFrame(json.load(open(CENSUS_JSON, encoding="utf-8")))
    census["wind_ms"] = (census["wind1"] + census["wind2"]) / 2.0
    census["date"] = pd.to_datetime(census["date"], format="%Y%m%d")
    census["month"] = census["date"].dt.month

    ev = pd.read_csv(EVENTS_CSV)
    ev["code"] = ev["scene"].map(scene_code)
    grp = ev.groupby("code").agg(
        n_events=("event_id", "count"),
        wind_ev=("wind_ms", "mean"),
        date=("time_utc", "min"),
        n_in=("wind_flag", lambda s: (s == "in_window").sum()),
        n_out=("wind_flag", lambda s: (s == "out_of_window").sum()),
    )
    grp["date"] = pd.to_datetime(grp["date"], utc=True).dt.tz_localize(None).dt.normalize()
    grp["month"] = grp["date"].dt.month
    mixed = grp[(grp.n_in > 0) & (grp.n_out > 0)]
    if len(mixed):
        print("[warn] scenes with mixed wind_flag, using majority vote:")
        print(mixed.to_string())
    grp["in_window_ev"] = grp.n_in >= grp.n_out

    # QA correction: scenes whose sampled events are all FP lose wave status.
    # Census-only scenes are exempt (their tp is already the reviewed count).
    qa = pd.read_csv(QA_CSV)[["event_id", "judgment"]]
    qa_ev = ev.merge(qa, on="event_id", how="inner")
    qa_grp = qa_ev.groupby("code")["judgment"].agg(
        qa_n="count",
        qa_tp_unc=lambda s: int(s.isin(["TP", "UNCERTAIN"]).sum()),
    )
    qa_all_fp = qa_grp[qa_grp.qa_tp_unc == 0]
    print(f"[qa] {len(qa_ev)} events with QA judgments over "
          f"{qa_ev['code'].nunique()} scenes; all-FP scenes: "
          f"{ {c: int(qa_all_fp.loc[c, 'qa_n']) for c in qa_all_fp.index} }")

    def qa_correction(n_events, source, code):
        """Return (has_wave, qa_note) for an event-db-backed scene."""
        has_wave = n_events > 0
        note = ""
        if source in ("both", "event_db") and code in qa_all_fp.index:
            has_wave = False
            note = f"qa_all_fp (n={int(qa_all_fp.loc[code, 'qa_n'])})"
        return has_wave, note

    rows = []
    for _, r in census.iterrows():
        code = r["code"]
        if code in grp.index:
            g = grp.loc[code]
            has_wave, note = qa_correction(int(g["n_events"]), "both", code)
            rows.append(dict(
                date=r["date"], code=code, month=int(r["month"]),
                season=season_of(int(r["month"])),
                wind_ms=round(float(r["wind_ms"]), 2),
                in_window=bool(r["in_window"]),
                n_events=int(g["n_events"]),  # event-db count wins on overlap
                has_wave=has_wave,
                source="both",
                qa_note=note,
            ))
        else:
            rows.append(dict(
                date=r["date"], code=code, month=int(r["month"]),
                season=season_of(int(r["month"])),
                wind_ms=round(float(r["wind_ms"]), 2),
                in_window=bool(r["in_window"]),
                n_events=int(r["tp"]),
                has_wave=int(r["tp"]) > 0,
                source="census",
                qa_note="",
            ))
    for code, g in grp.iterrows():
        if code in set(census["code"]):
            continue
        has_wave, note = qa_correction(int(g["n_events"]), "event_db", code)
        rows.append(dict(
            date=g["date"], code=code, month=int(g["month"]),
            season=season_of(int(g["month"])),
            wind_ms=round(float(g["wind_ev"]), 2),
            in_window=bool(g["in_window_ev"]),
            n_events=int(g["n_events"]),
            has_wave=has_wave,
            source="event_db",
            qa_note=note,
        ))
    df = pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)
    return df


def statistics(df, census_only):
    """Monthly in-window wave rates + Fisher + Clopper-Pearson."""
    inw = df[df.in_window]
    monthly = []
    for m in range(1, 13):
        sub = inw[inw.month == m]
        monthly.append(dict(
            month=m, n_in=len(sub), n_wave=int(sub.has_wave.sum()),
            n_out=int(((~df.in_window) & (df.month == m)).sum()),
            rate=(sub.has_wave.mean() if len(sub) else np.nan),
        ))
    monthly = pd.DataFrame(monthly)

    win = census_only[(census_only.season == "winter") & census_only.in_window]
    summ = census_only[(census_only.season == "summer") & census_only.in_window]
    tab = [[int(win.has_wave.sum()), int((~win.has_wave).sum())],
           [int(summ.has_wave.sum()), int((~summ.has_wave).sum())]]
    _, p_two = fisher_exact(tab, alternative="two-sided")
    _, p_less = fisher_exact(tab, alternative="less")
    cp_upper = beta.ppf(0.95, tab[0][0] + 1, tab[0][0] + tab[0][1] - tab[0][0])
    return monthly, tab, p_two, p_less, cp_upper


def crosscheck(df, monthly, tab, p_two, p_less, cp_upper):
    print("=" * 72)
    print("CROSS-CHECK TABLE")
    print("=" * 72)
    print(f"total scenes (union): {len(df)}")
    print(f"  by source: {df.source.value_counts().to_dict()}")
    print(f"  wave-positive scenes: {int(df.has_wave.sum())}, "
          f"in-window scenes: {int(df.in_window.sum())}")
    print()
    print(f"{'month':>5} {'season':>12} {'in_window':>9} {'out_window':>10} "
          f"{'wave(in)':>8} {'rate':>6}")
    for _, r in monthly.iterrows():
        m = int(r.month)
        rate = f"{r.rate:.2f}" if r.n_in > 0 else "  --"
        print(f"{m:>5} {season_of(m):>12} {int(r.n_in):>9} {int(r.n_out):>10} "
              f"{int(r.n_wave):>8} {rate:>6}")
    print()
    print("Fisher exact test on CENSUS scenes only (unbiased sample; event_db")
    print("scenes are wave-positive by selection):")
    print(f"  winter in-window: {tab[0][0]}/{sum(tab[0])} with waves")
    print(f"  summer in-window: {tab[1][0]}/{sum(tab[1])} with waves")
    print(f"  p(two-sided) = {p_two:.2e},  p(one-sided, less) = {p_less:.2e}")
    print(f"  winter incidence 95% Clopper-Pearson upper bound "
          f"({tab[0][0]}/{sum(tab[0])}): {cp_upper*100:.1f}%")
    print("=" * 72)


def make_figure(df, monthly, tab, p_two, p_less, cp_upper):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
    })
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 6.4), height_ratios=[1.35, 1.0],
        gridspec_kw=dict(hspace=0.32))

    # ---- panel (a): wind time series -------------------------------------
    t0, t1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-03-31")
    # season background bands
    for a, b, c, lab in [
        (t0, pd.Timestamp("2023-03-31"), "#dce9f5", "winter"),
        (pd.Timestamp("2023-04-01"), pd.Timestamp("2023-04-30"), "#eeeeee", None),
        (pd.Timestamp("2023-05-01"), pd.Timestamp("2023-10-31"), "#fdeede", "summer"),
        (pd.Timestamp("2023-11-01"), t1, "#dce9f5", None),
    ]:
        ax1.axvspan(a, b, color=c, zorder=0)
    # ISW-visible wind window band
    ax1.axhspan(WIND_LO, WIND_HI, color="#bfe3bf", alpha=0.55, zorder=1)
    ax1.axhline(WIND_LO, color="seagreen", lw=0.8, ls="--", zorder=2)
    ax1.axhline(WIND_HI, color="seagreen", lw=0.8, ls="--", zorder=2)
    ax1.text(pd.Timestamp("2023-01-15"), 6.0, "ISW-visible wind window (2-10 m/s)",
             ha="left", va="center", fontsize=8, color="darkgreen", zorder=3,
             bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.2))

    wave = df[df.has_wave]
    nowave = df[~df.has_wave]
    sizes = 40 + 130 * np.log10(wave.n_events.clip(lower=1))
    ax1.scatter(wave.date, wave.wind_ms, s=sizes, c="#1f5fa8",
                edgecolors="k", linewidths=0.5, zorder=5,
                label="wave-positive (size ∝ log$_{10}$ events)")
    ax1.scatter(nowave.date, nowave.wind_ms, s=42, marker="x",
                c="#777777", linewidths=1.2, zorder=4,
                label="no wave detected")
    # season labels
    ax1.text(pd.Timestamp("2023-02-15"), 17.6, "winter", ha="center",
             fontsize=8, color="#3a5a78", style="italic")
    ax1.text(pd.Timestamp("2023-07-31"), 17.6, "summer", ha="center",
             fontsize=8, color="#9a5b1e", style="italic")
    ax1.text(pd.Timestamp("2024-01-15"), 17.6, "winter", ha="center",
             fontsize=8, color="#3a5a78", style="italic")

    ax1.set_xlim(t0, t1)
    ax1.set_ylim(0, 19)
    ax1.set_ylabel("ERA5 wind speed (m/s)")
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax1.set_title("(a) Scene acquisition wind speed vs. ISW-visible wind window",
                  loc="left")

    # ---- panel (b): monthly in-window wave rate ---------------------------
    months = monthly.month.to_numpy()
    rates = monthly.rate.to_numpy(dtype=float)
    colors = ["#3a6fa0" if season_of(int(m)) == "winter" else "#d98c3f"
              for m in months]
    mask = ~np.isnan(rates)
    ax2.bar(months[mask], rates[mask], color=np.array(colors, dtype=object)[mask],
            width=0.62, edgecolor="k", linewidth=0.5, zorder=3)
    for m, r in monthly.iterrows():
        if r.n_in > 0:
            ax2.text(r.month, (r.rate if not np.isnan(r.rate) else 0) + 0.03,
                     f"n={int(r.n_in)}", ha="center", va="bottom", fontsize=8)
        else:
            ax2.text(r.month, 0.02, "n=0", ha="center", va="bottom",
                     fontsize=8, color="#888888", rotation=90)
    ax2.set_xticks(months)
    ax2.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax2.set_ylim(0, 1.32)
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_ylabel("wave-positive fraction\n(in-window scenes)")
    ax2.set_xlabel("month")
    ax2.set_title("(b) Monthly ISW occurrence rate within the visible wind window",
                  loc="left")
    txt = (f"Fisher exact (census scenes, winter {tab[0][0]}/{sum(tab[0])} "
           f"vs summer {tab[1][0]}/{sum(tab[1])} in-window): "
           f"p = {p_less:.1e}\n"
           f"winter incidence 95% Clopper-Pearson upper bound "
           f"({tab[0][0]}/{sum(tab[0])}) = {cp_upper*100:.1f}%")
    ax2.text(0.015, 0.62, txt, transform=ax2.transAxes, ha="left", va="top",
             fontsize=8, bbox=dict(fc="white", ec="#999999", lw=0.6, alpha=0.9))
    for m in (1, 2, 3, 11, 12):
        ax2.axvspan(m - 0.5, m + 0.5, color="#dce9f5", zorder=0)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"figure saved: {OUT_PNG}")


def main():
    df = build_table()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y%m%d")
    out.to_csv(OUT_CSV, index=False)
    print(f"table saved: {OUT_CSV} ({len(df)} scenes)")

    census_only = df[df.source.isin(["census", "both"])]
    monthly, tab, p_two, p_less, cp_upper = statistics(df, census_only)
    crosscheck(df, monthly, tab, p_two, p_less, cp_upper)

    # extra: same Fisher with the full union table, for reference
    win = df[(df.season == "winter") & df.in_window]
    summ = df[(df.season == "summer") & df.in_window]
    tab_full = [[int(win.has_wave.sum()), int((~win.has_wave).sum())],
                [int(summ.has_wave.sum()), int((~summ.has_wave).sum())]]
    _, p_full = fisher_exact(tab_full, alternative="less")
    print(f"[ref] Fisher on FULL union table {tab_full}: p(one-sided) = {p_full:.2e}")
    print(f"[ref] beta.ppf(0.95, 1, 13) = {beta.ppf(0.95, 1, 13)*100:.1f}%  "
          f"(standard CP, x=0/n=13)")
    print(f"[ref] beta.ppf(0.95, 1, 14) = {beta.ppf(0.95, 1, 14)*100:.1f}%  "
          f"(x=0 with n+1 denominator)")

    make_figure(df, monthly, tab, p_two, p_less, cp_upper)


if __name__ == "__main__":
    main()
