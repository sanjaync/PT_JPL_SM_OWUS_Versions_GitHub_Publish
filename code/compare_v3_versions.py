#!/usr/bin/env python3
"""
compare_v3_versions.py
---------------------
Compare metrics across v3.1 (OWUS agent_2), v3.2 (OWUS agent_3_scientific),
and v3.3 (OWUS ensemble_top5) for all 40 OzFlux sites.

Produces:
  1. CSV summary table
  2. KGE comparison bar chart (ALL DAYS) across all sites
  3. KGE comparison by regime (grouped bar)
  4. RMSE comparison bar chart (ALL DAYS)
  5. Scatter: v3.2 vs v3.1 KGE, v3.3 vs v3.1 KGE
  6. Improvement heatmap
"""

import os, re, glob
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
BASE = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm"
VERSIONS = {
    "v3.1": ("output_L6_OWUS_MultiPET_v3.1", "OWUS Agent 2"),
    "v3.2": ("output_L6_OWUS_MultiPET_v3.2", "OWUS Agent 3 Sci"),
    "v3.3": ("output_L6_OWUS_MultiPET_v3.3", "OWUS Ensemble Top5"),
}
OUT_DIR = os.path.join(BASE, "comparison_v3_versions")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["PTJPL_Base", "PTJPL_SM", "SM_OPT", "SM_BF"]
REGIMES = ["ALL DAYS", "DRY DAYS (precip)", "WET DAYS (precip)",
           "VERY DRY (SM regime)", "TRANSITION (SM regime)", "WET (SM regime)"]

# =====================================================================
# Parser
# =====================================================================
def parse_metrics_file(filepath):
    """Parse a v3.x metrics file into dict[regime][model] = {metric: val}."""
    results = {}
    current_regime = None
    with open(filepath) as f:
        for line in f:
            line = line.rstrip()
            # Detect regime header
            m = re.match(r"=== (.+?) \(n=(\d+)\) ===", line)
            if m:
                current_regime = m.group(1)
                results[current_regime] = {}
                continue
            if current_regime and line and not line.startswith("-") and not line.startswith("Model"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 9:
                    model = parts[0].strip()
                    try:
                        results[current_regime][model] = {
                            "RMSE":  float(parts[1]),
                            "Bias":  float(parts[2]),
                            "R2":    float(parts[3]),
                            "NSE":   float(parts[4]),
                            "KGE":   float(parts[5]),
                            "r":     float(parts[6]),
                            "alpha": float(parts[7]),
                            "beta":  float(parts[8]),
                        }
                    except ValueError:
                        pass
    return results


def collect_all():
    """Collect metrics for all versions, all sites, all regimes, all models."""
    rows = []
    for ver, (outdir, label) in VERSIONS.items():
        pattern = os.path.join(BASE, outdir, "**", f"*_{ver}_metrics.txt")
        for fp in sorted(glob.glob(pattern, recursive=True)):
            site = os.path.basename(fp).replace(f"_{ver}_metrics.txt", "")
            parsed = parse_metrics_file(fp)
            for regime, models in parsed.items():
                for model, metrics in models.items():
                    row = {"version": ver, "label": label, "site": site,
                           "regime": regime, "model": model}
                    row.update(metrics)
                    rows.append(row)
    return pd.DataFrame(rows)


# =====================================================================
# Plots
# =====================================================================
def plot_kge_bars_alldays(df, metric="KGE"):
    """Bar chart: per-site KGE for SM_BF across 3 versions (ALL DAYS)."""
    sub = df[(df["regime"] == "ALL DAYS") & (df["model"] == "SM_BF")].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="site", columns="version", values=metric)
    pivot = pivot.sort_values(by="v3.1", ascending=False) if "v3.1" in pivot.columns else pivot

    fig, ax = plt.subplots(figsize=(max(16, len(pivot) * 0.5), 6), constrained_layout=True)
    x = np.arange(len(pivot))
    w = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, ver in enumerate(["v3.1", "v3.2", "v3.3"]):
        if ver in pivot.columns:
            vals = pivot[ver].values
            label_txt = VERSIONS[ver][1]
            ax.bar(x + i * w - w, vals, w, label=f"{ver} ({label_txt})", color=colors[i], alpha=0.85)

    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(f"{metric} (SM_BF — ALL DAYS)")
    ax.set_title(f"SM_BF {metric} by Site — v3.1 vs v3.2 vs v3.3", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.3)
    fig.savefig(os.path.join(OUT_DIR, f"Comparison_{metric}_SM_BF_AllDays.png"), dpi=300)
    plt.close(fig)
    print(f"  [OK] {metric} bar chart")


def plot_all_models_kge(df):
    """Grouped bar: KGE for all 4 models across versions (ALL DAYS, site-mean)."""
    sub = df[df["regime"] == "ALL DAYS"].copy()
    means = sub.groupby(["version", "model"])["KGE"].mean().unstack("model")
    means = means.reindex(columns=MODELS)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(MODELS))
    w = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, ver in enumerate(["v3.1", "v3.2", "v3.3"]):
        if ver in means.index:
            vals = means.loc[ver].values
            ax.bar(x + i * w - w, vals, w, label=f"{ver} ({VERSIONS[ver][1]})", color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)
    ax.set_ylabel("Mean KGE (All Sites)")
    ax.set_title("Mean KGE by Model — ALL DAYS (40 Sites)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.axhline(0, color="gray", lw=0.8)
    fig.savefig(os.path.join(OUT_DIR, "Comparison_MeanKGE_AllModels.png"), dpi=300)
    plt.close(fig)
    print("  [OK] Mean KGE all models")


def plot_regime_comparison(df):
    """KGE for SM_BF across regimes, averaged over sites."""
    sub = df[df["model"] == "SM_BF"].copy()
    means = sub.groupby(["version", "regime"])["KGE"].mean().unstack("regime")
    # Reorder regimes
    regime_order = [r for r in REGIMES if r in means.columns]
    means = means[regime_order]

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    x = np.arange(len(regime_order))
    w = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, ver in enumerate(["v3.1", "v3.2", "v3.3"]):
        if ver in means.index:
            vals = means.loc[ver].values
            ax.bar(x + i * w - w, vals, w, label=f"{ver} ({VERSIONS[ver][1]})", color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([r.replace(" (precip)", "\n(precip)").replace(" (SM regime)", "\n(SM)") for r in regime_order],
                       fontsize=9)
    ax.set_ylabel("Mean KGE (SM_BF)")
    ax.set_title("SM_BF KGE by Regime — Averaged over 40 Sites", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.axhline(0, color="gray", lw=0.8)
    fig.savefig(os.path.join(OUT_DIR, "Comparison_KGE_SM_BF_ByRegime.png"), dpi=300)
    plt.close(fig)
    print("  [OK] Regime KGE comparison")


def plot_scatter_improvement(df):
    """Scatter: v3.2 KGE vs v3.1 KGE, and v3.3 KGE vs v3.1 KGE (SM_BF, ALL DAYS)."""
    sub = df[(df["regime"] == "ALL DAYS") & (df["model"] == "SM_BF")].copy()
    pivot = sub.pivot_table(index="site", columns="version", values="KGE")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)

    for ax, new_ver, color, title in zip(
        axes,
        ["v3.2", "v3.3"],
        ["#ff7f0e", "#2ca02c"],
        ["Agent 3 Sci", "Ensemble Top5"]
    ):
        if "v3.1" not in pivot.columns or new_ver not in pivot.columns:
            continue
        valid = pivot[["v3.1", new_ver]].dropna()
        ax.scatter(valid["v3.1"], valid[new_ver], s=40, alpha=0.7, color=color, edgecolor="k", lw=0.3)

        lo = min(valid["v3.1"].min(), valid[new_ver].min()) - 0.05
        hi = max(valid["v3.1"].max(), valid[new_ver].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"KGE (v3.1 — OWUS Agent 2)")
        ax.set_ylabel(f"KGE ({new_ver} — {title})")
        ax.set_title(f"SM_BF: {new_ver} vs v3.1", fontsize=13, fontweight="bold")
        ax.grid(True, ls="--", alpha=0.3)

        # Count improvements
        improved = (valid[new_ver] > valid["v3.1"]).sum()
        total = len(valid)
        ax.text(0.05, 0.95, f"Improved: {improved}/{total} sites",
                transform=ax.transAxes, fontsize=10, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.8))

        # Label outlier sites
        diff = (valid[new_ver] - valid["v3.1"]).abs()
        top3 = diff.nlargest(3).index
        for site in top3:
            ax.annotate(site, (valid.loc[site, "v3.1"], valid.loc[site, new_ver]),
                        fontsize=6, alpha=0.8, ha="left")

    fig.savefig(os.path.join(OUT_DIR, "Comparison_Scatter_KGE_Improvement.png"), dpi=300)
    plt.close(fig)
    print("  [OK] Scatter improvement")


def plot_delta_heatmap(df):
    """Heatmap: KGE change from v3.1 for SM_OPT and SM_BF across sites."""
    sub = df[(df["regime"] == "ALL DAYS") & (df["model"].isin(["SM_OPT", "SM_BF"]))].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(sub["site"].unique()) * 0.25)),
                             constrained_layout=True)

    for ax, new_ver, title in zip(axes, ["v3.2", "v3.3"], ["Agent 3 Sci", "Ensemble Top5"]):
        rows = []
        for model in ["SM_OPT", "SM_BF"]:
            pivot = sub[sub["model"] == model].pivot_table(index="site", columns="version", values="KGE")
            if "v3.1" in pivot.columns and new_ver in pivot.columns:
                delta = pivot[new_ver] - pivot["v3.1"]
                for site, d in delta.items():
                    rows.append({"site": site, "model": model, "delta_KGE": d})
        if not rows:
            continue

        ddf = pd.DataFrame(rows).pivot_table(index="site", columns="model", values="delta_KGE")
        ddf = ddf.sort_values("SM_BF", ascending=True)
        im = ax.imshow(ddf.values, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.3)
        ax.set_yticks(range(len(ddf)))
        ax.set_yticklabels(ddf.index, fontsize=6)
        ax.set_xticks(range(len(ddf.columns)))
        ax.set_xticklabels(ddf.columns)
        ax.set_title(f"ΔKGE: {new_ver} − v3.1 ({title})", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.6, label="ΔKGE")

    fig.savefig(os.path.join(OUT_DIR, "Comparison_DeltaKGE_Heatmap.png"), dpi=300)
    plt.close(fig)
    print("  [OK] Delta KGE heatmap")


# =====================================================================
# Main
# =====================================================================
def main():
    print("Collecting metrics from all 3 versions...")
    df = collect_all()
    print(f"  Total rows: {len(df)} | Sites: {df['site'].nunique()} | Versions: {df['version'].unique()}")

    # Save summary CSV
    summary = df[df["regime"] == "ALL DAYS"].pivot_table(
        index=["site", "model"],
        columns="version",
        values=["KGE", "RMSE", "R2", "NSE"]
    ).round(3)
    summary.to_csv(os.path.join(OUT_DIR, "Summary_AllDays_Comparison.csv"))
    print("  [OK] Summary CSV saved")

    # Delta summary
    alldays = df[df["regime"] == "ALL DAYS"].copy()
    for metric in ["KGE", "RMSE"]:
        pivot = alldays.pivot_table(index=["site", "model"], columns="version", values=metric)
        for new_v in ["v3.2", "v3.3"]:
            if "v3.1" in pivot.columns and new_v in pivot.columns:
                sign = -1 if metric == "RMSE" else 1  # For RMSE, lower is better
                delta = (pivot[new_v] - pivot["v3.1"]) * sign
                improved = (delta > 0).sum()
                total = delta.notna().sum()
                mean_d = delta.mean()
                print(f"  {metric} {new_v} vs v3.1: {improved}/{total} improved | mean Δ = {mean_d:+.3f}")

    # Generate plots
    print("\nGenerating plots...")
    plot_kge_bars_alldays(df, "KGE")
    plot_kge_bars_alldays(df, "RMSE")
    plot_all_models_kge(df)
    plot_regime_comparison(df)
    plot_scatter_improvement(df)
    plot_delta_heatmap(df)

    print(f"\n[SUCCESS] All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
