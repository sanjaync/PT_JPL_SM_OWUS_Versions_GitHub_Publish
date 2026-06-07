#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_v13_metrics_robust.py
----------------------------
Analyzes OzFlux v13 model structural uncertainty.
- Parses *_v13_metrics.txt files.
- Decodes 1728+ model names using factorial design.
- Generates per-site Top-20 bar charts (PNG/SVG).
- Generates global win/podium counts.
- Generates categorical performance summaries.
- Outputs summary Markdown report.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ==========================
# Settings
# ==========================
ROOT_DIR = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v13_DynamicBeta"
PLOT_DIR = os.path.join(ROOT_DIR, "analysis_plots_v13")
PER_SITE_DIR = os.path.join(PLOT_DIR, "per_site")
GLOBAL_DIR = os.path.join(PLOT_DIR, "global")

TOP_N_PER_SITE = 50  # Memory efficiency: only keep top 50 per site/regime
BAR_PLOT_TOP = 20    # Show Top 20 in per-site bar charts
HEATMAP_TOP = 50     # Show Top 50 in global heatmap

# Force Agg backend for headless execution
plt.switch_backend('agg')

# ==========================
# Helpers
# ==========================
def ensure_dirs():
    for d in [PLOT_DIR, PER_SITE_DIR, GLOBAL_DIR]:
        os.makedirs(d, exist_ok=True)

def save_fig(fig, base_path):
    fig.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(base_path + ".svg", format="svg", bbox_inches="tight")

def decode_model_name(name):
    """
    Decodes strings like 'SM_Base_BF_PM_PTJPL_Lin_Stat_TC'
    Factor order: Proxy, Soil, OWUS, PET, Interception, Shape, Dynamics, Capping
    """
    # Standard models
    if name == "PTJPL_Base":
        return {"Proxy": "Base", "Soil": "Base", "OWUS": "N/A", "PET": "N/A", "Int": "N/A", "Shape": "N/A", "Dyn": "N/A", "Cap": "N/A"}
    if name == "PTJPL_SM":
        return {"Proxy": "SM", "Soil": "SM", "OWUS": "N/A", "PET": "N/A", "Int": "N/A", "Shape": "N/A", "Dyn": "N/A", "Cap": "N/A"}
    
    p = name.split('_')
    if len(p) == 8:
        return {
            "Proxy": p[0],
            "Soil": p[1],
            "OWUS": p[2],
            "PET": p[3],
            "Int": p[4],
            "Shape": p[5],
            "Dyn": p[6],
            "Cap": p[7]
        }
    return {k: "Other" for k in ["Proxy", "Soil", "OWUS", "PET", "Int", "Shape", "Dyn", "Cap"]}

def parse_metrics_file(path):
    """
    Parses v13_metrics.txt into a DataFrame.
    """
    if os.path.getsize(path) == 0:
        return None
    
    site_meta = {}
    regime_data = []
    current_regime = None
    
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Header info
            if line.startswith("Site:"):
                # Site: yanco | L6: Yanco | KG: BSk - Arid, steppe, cold | Veg: Pasture
                parts = [x.strip() for x in line.split('|')]
                for p in parts:
                    if ':' in p:
                        k, v = p.split(':', 1)
                        site_meta[k.strip()] = v.strip()
                continue
                
            # Regime Detection
            if line.startswith("==="):
                # === ALL DAYS (n=4480) ===
                current_regime = line.split('===')[1].split('(')[0].strip()
                continue
                
            # Table Header / Divider skip
            if line.startswith("Model") or line.startswith("---"):
                continue
                
            # Data Lines
            if current_regime and '|' in line:
                # Model | RMSE | Bias | R2 | NSE | KGE | r | a | b
                parts = [x.strip() for x in line.split('|')]
                if len(parts) >= 9:
                    try:
                        row = {
                            "Model": parts[0],
                            "RMSE": float(parts[1]),
                            "Bias": float(parts[2]),
                            "R2": float(parts[3]),
                            "NSE": float(parts[4]),
                            "KGE": float(parts[5]),
                            "r": float(parts[6]),
                            "alpha": float(parts[7]),
                            "beta": float(parts[8]),
                            "Regime": current_regime
                        }
                        regime_data.append(row)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[ERROR] Parsing {path}: {e}")
        return None
        
    if not regime_data:
        return None
        
    df = pd.DataFrame(regime_data)
    df["Site"] = site_meta.get("Site", os.path.basename(os.path.dirname(path)))
    return df

# ==========================
# Main Execution
# ==========================
def main():
    ensure_dirs()
    print(f"[{datetime.now()}] Starting Analysis...")
    
    metric_files = sorted(glob.glob(os.path.join(ROOT_DIR, "*", "*_v13_metrics.txt")))
    print(f"Found {len(metric_files)} potential metric files.")
    
    all_sites_top = []
    
    for fpath in metric_files:
        site_name = os.path.basename(os.path.dirname(fpath))
        print(f"Processing {site_name}...", end=" ", flush=True)
        
        df_site = parse_metrics_file(fpath)
        if df_site is None:
            print("Skipped (Empty/Failed)")
            continue
            
        # Per-Regime Filtering to keep data manageable
        for regime in df_site["Regime"].unique():
            df_reg = df_site[df_site["Regime"] == regime].copy()
            # Sort by KGE descending
            df_reg = df_reg.sort_values("KGE", ascending=False)
            
            # Keep Top N for memory
            top_50 = df_reg.head(TOP_N_PER_SITE)
            all_sites_top.append(top_50)
            
            # Plot Top 20 for 'ALL DAYS' regime
            if regime == "ALL DAYS":
                plot_site_top(site_name, df_reg.head(BAR_PLOT_TOP))
        
        print("Done")

    if not all_sites_top:
        print("No data collected. Exiting.")
        return
        
    df_global = pd.concat(all_sites_top, ignore_index=True)
    
    # Global Plots
    plot_win_counts(df_global)
    plot_categorical_summary(df_global)
    plot_global_heatmap(df_global)
    
    # Generate Report
    generate_report(df_global)
    print(f"[{datetime.now()}] Analysis Complete. Results in {PLOT_DIR}")

# ==========================
# Plotting Functions
# ==========================
def plot_site_top(site, df_top):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_top, x="KGE", y="Model", ax=ax, palette="viridis")
    ax.set_title(f"{site} - Top {BAR_PLOT_TOP} Models (All Days)")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    out_path = os.path.join(PER_SITE_DIR, f"{site}_TopModels")
    save_fig(fig, out_path)
    plt.close(fig)

def plot_win_counts(df):
    """
    Calculates podium (Top 3) win counts per model across sites in 'ALL DAYS' regime.
    """
    df_all = df[df["Regime"] == "ALL DAYS"].copy()
    
    podium_rows = []
    for site in df_all["Site"].unique():
        site_data = df_all[df_all["Site"] == site].sort_values("KGE", ascending=False).head(3)
        for i, (idx, row) in enumerate(site_data.iterrows()):
            podium_rows.append({"Model": row["Model"], "Rank": i+1})
            
    df_podium = pd.DataFrame(podium_rows)
    counts = df_podium.groupby(["Model", "Rank"]).size().unstack(fill_value=0)
    if counts.empty: return
    
    # Sort by Total Top 3 wins
    counts["Total"] = counts.sum(axis=1)
    counts = counts.sort_values("Total", ascending=False).head(20)
    counts = counts.drop(columns="Total")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    counts.plot(kind='barh', stacked=True, ax=ax, color=['#FFD700', '#C0C0C0', '#CD7F32'])
    ax.set_title("Top 20 Models by Podium Frequency (Top 3 Wins)")
    ax.set_xlabel("Number of Sites")
    ax.invert_yaxis()
    ax.legend(["Gold (1st)", "Silver (2nd)", "Bronze (3rd)"])
    
    save_fig(fig, os.path.join(GLOBAL_DIR, "v13_Podium_Wins"))
    plt.close(fig)

def plot_categorical_summary(df):
    """
    Aggregates performance by structural factors.
    """
    df_all = df[df["Regime"] == "ALL DAYS"].copy()
    
    # Decode all models
    decoded = df_all["Model"].apply(decode_model_name)
    df_cat = pd.concat([df_all, pd.DataFrame(list(decoded))], axis=1)
    
    factors = ["Proxy", "Soil", "OWUS", "PET", "Int", "Shape", "Dyn", "Cap"]
    
    # Create a grid of plots
    fig, axes = plt.subplots(4, 2, figsize=(16, 20), constrained_layout=True)
    axes = axes.flatten()
    
    for i, factor in enumerate(factors):
        sns.boxplot(data=df_cat, x=factor, y="KGE", ax=axes[i], palette="Set2")
        axes[i].set_title(f"Impact of {factor} on KGE")
        axes[i].tick_params(axis='x', rotation=45)
        
    save_fig(fig, os.path.join(GLOBAL_DIR, "v13_Categorical_Summary"))
    plt.close(fig)

def plot_global_heatmap(df):
    """
    Heatmap of KGE for the globally Top-N models across all sites.
    """
    df_all = df[df["Regime"] == "ALL DAYS"].copy()
    
    # Find globally best models (highest median KGE)
    top_models = df_all.groupby("Model")["KGE"].median().sort_values(ascending=False).head(HEATMAP_TOP).index
    
    df_heat = df_all[df_all["Model"].isin(top_models)]
    pivot = df_heat.pivot(index="Site", columns="Model", values="KGE")
    
    # Reorder columns by median
    pivot = pivot[top_models]
    
    plt.figure(figsize=(20, 12))
    sns.heatmap(pivot, cmap="RdYlGn", center=0, annot=False, cbar_kws={'label': 'KGE'})
    plt.title(f"Global KGE Heatmap (Top {HEATMAP_TOP} Models)")
    
    save_fig(plt.gcf(), os.path.join(GLOBAL_DIR, "v13_Global_KGE_Heatmap"))
    plt.close()

def generate_report(df):
    """
    Creates a Markdown summary report.
    """
    df_all = df[df["Regime"] == "ALL DAYS"].copy()
    
    # Identify site-wise winners
    winners = df_all.sort_values(["Site", "KGE"], ascending=[True, False]).groupby("Site").head(1)
    
    # Identify "Difficult" sites (Max KGE < 0.3)
    difficult = winners[winners["KGE"] < 0.3]
    
    # Identify Global Winner (highest mean KGE)
    global_model = df_all.groupby("Model")["KGE"].mean().sort_values(ascending=False).head(1)
    
    report_path = os.path.join(ROOT_DIR, "v13_Analysis_Summary.md")
    with open(report_path, 'w') as f:
        f.write(f"# OzFlux v13 Analysis Summary\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"## Global Performance\n")
        f.write(f"- **Top Global Model**: `{global_model.index[0]}` (Mean KGE: {global_model.values[0]:.3f})\n")
        f.write(f"- **Total Sites Processed**: {len(winners)}\n\n")
        
        f.write(f"## Difficult Sites (Physics Warning)\n")
        f.write(f"The following sites show poor performance (KGE < 0.3) across all tested model variants:\n")
        if not difficult.empty:
            for _, row in difficult.iterrows():
                f.write(f"- **{row['Site']}**: Best KGE = {row['KGE']:.3f} (Model: `{row['Model']}`)\n")
        else:
            f.write("- None! All sites show reasonable performance.\n")
            
        f.write(f"\n## Top Model Per Site\n")
        f.write("| Site | Best Model | KGE | r | Bias |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for _, row in winners.iterrows():
            f.write(f"| {row['Site']} | `{row['Model']}` | {row['KGE']:.3f} | {row['r']:.3f} | {row['Bias']:.2f} |\n")
            
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    main()
