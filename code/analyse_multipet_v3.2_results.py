#!/usr/bin/env python3
"""
analyse_multipet_v3.2_results.py
---------------------------------
Analyse PTJPL MultiPET v3.2 results across all OzFlux sites.

Metrics computed per site (and aggregated):
  RMSE, Bias, R², KGE, r (Pearson)

Breakdowns by:
  - Overall (all-data)
  - Season  (SUMMER / AUTUMN / WINTER / SPRING)
  - SM regime (very_dry / transition / wet)

Models compared:
  LE_PTJPL_Base  |  LE_PTJPL_SM  |  LE_SM_OPT  |  LE_SM_BF

Outputs:
  • Console summary tables
  • CSV files per breakdown
  • Summary CSV ranking models
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2"
OUT_DIR  = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/analysis_v3.2_results"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ['LE_PTJPL_Base', 'LE_PTJPL_SM', 'LE_SM_OPT', 'LE_SM_BF']
SEASONS = ['SUMMER', 'AUTUMN', 'WINTER', 'SPRING']
REGIMES = ['very_dry', 'transition', 'wet']

# ─────────────────────────────────────────────
#  METRIC FUNCTIONS
# ─────────────────────────────────────────────
def rmse(obs, sim):
    return np.sqrt(np.mean((sim - obs) ** 2))

def bias(obs, sim):
    return np.mean(sim - obs)

def r_squared(obs, sim):
    if len(obs) < 2 or np.std(obs) == 0:
        return np.nan
    slope, intercept, r_val, p_val, se = stats.linregress(obs, sim)
    return r_val ** 2

def pearson_r(obs, sim):
    if len(obs) < 3 or np.std(obs) == 0:
        return np.nan
    r, _ = stats.pearsonr(obs, sim)
    return r

def nse(obs, sim):
    """Nash-Sutcliffe Efficiency"""
    if len(obs) < 5 or np.std(obs) == 0:
        return np.nan
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - (numerator / denominator)

def kge(obs, sim):
    """Kling–Gupta Efficiency"""
    if len(obs) < 3 or np.std(obs) == 0 or np.std(sim) == 0 or np.mean(obs) == 0:
        return np.nan
    r, _ = stats.pearsonr(obs, sim)
    alpha = np.std(sim) / np.std(obs)
    beta  = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

def compute_metrics(obs, sim):
    """Return dict of metrics given obs and sim arrays (NaN-dropped)."""
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs_c, sim_c = obs[mask], sim[mask]
    n = mask.sum()
    if n < 5:
        return dict(n=n, RMSE=np.nan, Bias=np.nan, R2=np.nan, NSE=np.nan, KGE=np.nan, r=np.nan, alpha=np.nan, beta=np.nan)
    
    r_val = pearson_r(obs_c, sim_c)
    m_obs = np.mean(obs_c)
    s_obs = np.std(obs_c)
    m_sim = np.mean(sim_c)
    s_sim = np.std(sim_c)
    
    alpha = s_sim / s_obs if s_obs != 0 else np.nan
    beta  = m_sim / m_obs if m_obs != 0 else np.nan
    
    return dict(
        n     = n,
        RMSE  = rmse(obs_c, sim_c),
        Bias  = bias(obs_c, sim_c),
        R2    = r_squared(obs_c, sim_c),
        NSE   = nse(obs_c, sim_c),
        KGE   = kge(obs_c, sim_c),
        r     = r_val,
        alpha = alpha,
        beta  = beta
    )

# ─────────────────────────────────────────────
#  LOAD ALL SITE DATA
# ─────────────────────────────────────────────
print("=" * 70)
print("  PTJPL MultiPET v3.2 — Enhanced Cross-site Analysis")
print("=" * 70)

csv_files = sorted(glob.glob(os.path.join(BASE_DIR, "*", "*_results.csv")))
print(f"\n  Found {len(csv_files)} site CSV files.\n")

all_frames = []
site_list  = []

for fpath in csv_files:
    site = os.path.basename(os.path.dirname(fpath))
    try:
        df = pd.read_csv(fpath, parse_dates=['time'])
        df['site'] = site
        # filter extreme LE spikes (e.g. > 1000 W/m2 occasionally found in faulty sensors)
        df.loc[df['LE_Obs'] > 1000, 'LE_Obs'] = np.nan
        df.loc[df['LE_Obs'] < -100, 'LE_Obs'] = np.nan
        
        df = df[df['LE_Obs'].notna()].copy()
        if len(df) < 20:
            print(f"  [SKIP] {site}: too few valid rows ({len(df)})")
            continue
        all_frames.append(df)
        site_list.append(site)
    except Exception as e:
        print(f"  [ERROR] {site}: {e}")

combined_raw = pd.concat(all_frames, ignore_index=True)

# ─────────────────────────────────────────────
#  OUTLIER SITE DETECTION (EXCLUSION)
# ─────────────────────────────────────────────
# Calculate per-site KGE to identify extremely broken sites
site_metrics_list = []
for site in site_list:
    df_s = combined_raw[combined_raw['site'] == site]
    m = compute_metrics(df_s['LE_Obs'], df_s['LE_PTJPL_Base']) # Using Base as reference for data quality
    site_metrics_list.append({'site': site, 'KGE_Base': m['KGE']})

site_stats = pd.DataFrame(site_metrics_list)
# Remove sites with extremely bad KGE (e.g. KGE < -3 is usually a sign of misaligned data or bad forcing)
# Based on user request to remove 'extreme outliers'
OUTLIER_THRESHOLD = -5
outlier_sites = site_stats[site_stats['KGE_Base'] < OUTLIER_THRESHOLD]['site'].tolist()

if outlier_sites:
    print(f"\n  [EXCLUDING] Outlier sites (Base KGE < {OUTLIER_THRESHOLD}):")
    for s in outlier_sites:
        print(f"    - {s}")
    combined = combined_raw[~combined_raw['site'].isin(outlier_sites)].copy()
    site_list = [s for s in site_list if s not in outlier_sites]
else:
    combined = combined_raw.copy()

print(f"\n  Final Clean Dataset:")
print(f"    - Sites retained: {len(site_list)}")
print(f"    - Total daily records: {len(combined):,}")

# ─────────────────────────────────────────────
#  1. OVERALL ALL-SITE METRICS
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("  1. OVERALL ALL-SITE METRICS (pooled)")
print("─" * 70)
overall_rows = []
for model in MODELS:
    m = compute_metrics(combined['LE_Obs'], combined[model])
    m['model'] = model
    overall_rows.append(m)
overall = pd.DataFrame(overall_rows).set_index('model')
print(overall[['n', 'RMSE', 'Bias', 'R2', 'NSE', 'KGE', 'r']].round(4).to_string())
overall.round(4).to_csv(os.path.join(OUT_DIR, "metrics_overall_enhanced.csv"))

# ─────────────────────────────────────────────
#  2. PER-SITE METRICS
# ─────────────────────────────────────────────
site_rows = []
for site in site_list:
    df_s = combined[combined['site'] == site]
    for model in MODELS:
        m = compute_metrics(df_s['LE_Obs'], df_s[model])
        m['site']  = site
        m['model'] = model
        site_rows.append(m)

site_df = pd.DataFrame(site_rows)
site_df.to_csv(os.path.join(OUT_DIR, "metrics_per_site_enhanced.csv"), index=False)

# ─────────────────────────────────────────────
#  3. BREAKDOWNS (Season, Regime, Precip)
# ─────────────────────────────────────────────

def group_analysis(df, group_col, group_vals, filename, label):
    print("\n" + "─" * 70)
    print(f"  {label} BREAKDOWN")
    print("─" * 70)
    rows = []
    for val in group_vals:
        df_sub = df[df[group_col] == val]
        if len(df_sub) < 10: continue
        for model in MODELS:
            m = compute_metrics(df_sub['LE_Obs'], df_sub[model])
            m[group_col] = val
            m['model']   = model
            rows.append(m)
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(OUT_DIR, filename), index=False)
    
    # Print KGE and RMSE summaries
    for mtr in ['KGE', 'RMSE']:
        piv = res_df.pivot(index=group_col, columns='model', values=mtr).round(3)
        print(f"\n  {mtr} by {group_col}:\n{piv.to_string()}")

group_analysis(combined, 'Season', SEASONS, "metrics_by_season_enhanced.csv", "3. SEASONAL")
group_analysis(combined, 'regime_sm', REGIMES, "metrics_by_sm_regime_enhanced.csv", "4. SM REGIME")

# Add Precip-based Dry/Wet breakdown (detailed in site metrics.txt)
# dry_day=1 means no/low rainfall, wet_day=1 means rainfall
print("\n" + "─" * 70)
print("  5. PRECIP-BASED BREAKDOWN (Dry vs Wet Days)")
print("─" * 70)
precip_rows = []
for ptype, pcol in [('Precip_Dry', 'dry_day'), ('Precip_Wet', 'wet_day')]:
    df_p = combined[combined[pcol] == 1]
    for model in MODELS:
        m = compute_metrics(df_p['LE_Obs'], df_p[model])
        m['type']  = ptype
        m['model'] = model
        precip_rows.append(m)
precip_df = pd.DataFrame(precip_rows)
precip_df.to_csv(os.path.join(OUT_DIR, "metrics_by_precip_day.csv"), index=False)
piv_kge = precip_df.pivot(index='type', columns='model', values='KGE').round(3)
print(f"\n  KGE by Precip Day Type:\n{piv_kge.to_string()}")

# ─────────────────────────────────────────────
#  7. CONSOLIDATED SUMMARY TABLE
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("  7. CONSOLIDATED SUMMARY (Pooled Results)")
print("=" * 70)

# Collect all primary slices into one dataframe for a master report
master_rows = []

# Slice 1: All Days
for model in MODELS:
    m = compute_metrics(combined['LE_Obs'], combined[model])
    m['category'] = 'ALL_DAYS'
    m['model']    = model
    master_rows.append(m)

# Slice 2: Precip-based
for ptype, pcol in [('PRECIP_DRY', 'dry_day'), ('PRECIP_WET', 'wet_day')]:
    df_p = combined[combined[pcol] == 1]
    for model in MODELS:
        m = compute_metrics(df_p['LE_Obs'], df_p[model])
        m['category'] = ptype
        m['model']    = model
        master_rows.append(m)

# Slice 3: SM Regimes
for regime in REGIMES:
    df_r = combined[combined['regime_sm'] == regime]
    for model in MODELS:
        m = compute_metrics(df_r['LE_Obs'], df_r[model])
        m['category'] = f"SM_{regime.upper()}"
        m['model']    = model
        master_rows.append(m)

master_df = pd.DataFrame(master_rows)
master_df.to_csv(os.path.join(OUT_DIR, "master_pooled_metrics_summary.csv"), index=False)

# Print a focused summary of KGE across all primary categories
print("\n  Summary Table: KGE across key categories")
print("-" * 50)
summary_kge = master_df.pivot(index='category', columns='model', values='KGE').round(3)
# Reorder categories for logical flow
cat_order = ['ALL_DAYS', 'PRECIP_DRY', 'PRECIP_WET', 'SM_VERY_DRY', 'SM_TRANSITION', 'SM_WET']
summary_kge = summary_kge.reindex([c for c in cat_order if c in summary_kge.index])
print(summary_kge.to_string())

print("\n  Summary Table: RMSE across key categories")
print("-" * 50)
summary_rmse = master_df.pivot(index='category', columns='model', values='RMSE').round(2)
summary_rmse = summary_rmse.reindex([c for c in cat_order if c in summary_rmse.index])
print(summary_rmse.to_string())

# ─────────────────────────────────────────────
#  8. PLOTS
# ─────────────────────────────────────────────
print("\n" + "─" * 70)
print("  8. GENERATING PLOTS …")
print("─" * 70)

COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
MODEL_COLORS = dict(zip(MODELS, COLORS))

# ── 7a. Boxplot: per-site RMSE by model ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Per-site Model Performance (MultiPET v3.2)', fontsize=14, fontweight='bold')

for ax, metric in zip(axes, ['RMSE', 'KGE']):
    data_list = [site_df[site_df['model'] == m][metric].dropna().values for m in MODELS]
    bp = ax.boxplot(data_list, patch_artist=True, labels=MODELS,
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f'{metric} − per site', fontsize=12)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fpath = os.path.join(OUT_DIR, "plot_boxplot_persite_metrics.png")
plt.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fpath}")

# ── 7b. Bar chart: Seasonal RMSE ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
fig.suptitle('Seasonal & Regime Metrics (pooled all sites)', fontsize=14, fontweight='bold')

for ax, metric in zip(axes[:2], ['RMSE', 'KGE']):
    piv = season_df.pivot(index='Season', columns='model', values=metric)
    piv = piv.reindex(SEASONS)
    x = np.arange(len(SEASONS))
    w = 0.2
    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        if model in piv.columns:
            ax.bar(x + i*w, piv[model], width=w, label=model, color=color, alpha=0.8)
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(SEASONS)
    ax.set_title(f'Seasonal {metric}')
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

for ax, metric in zip(axes[2:], ['RMSE', 'KGE']):
    piv = regime_df.pivot(index='regime', columns='model', values=metric)
    piv = piv.reindex(REGIMES)
    x = np.arange(len(REGIMES))
    for i, (model, color) in enumerate(MODEL_COLORS.items()):
        if model in piv.columns:
            ax.bar(x + i*w, piv[model], width=w, label=model, color=color, alpha=0.8)
    ax.set_xticks(x + 1.5*w)
    ax.set_xticklabels(REGIMES)
    ax.set_title(f'SM Regime {metric}')
    ax.set_ylabel(metric)
    ax.legend(fontsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
fpath = os.path.join(OUT_DIR, "plot_seasonal_regime_metrics.png")
plt.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fpath}")

# ── 7c. Heatmap: per-site KGE ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(site_list)//3)))
fig.suptitle('Per-site KGE and RMSE Heatmaps', fontsize=14, fontweight='bold')

for ax, metric, cmap, center in zip(axes, ['KGE', 'RMSE'], ['RdYlGn', 'RdYlGn_r'], [0.5, None]):
    piv = site_df.pivot(index='site', columns='model', values=metric)
    piv = piv[MODELS]
    sns.heatmap(piv, ax=ax, cmap=cmap, annot=True, fmt='.2f',
                linewidths=0.5, linecolor='white',
                center=center,
                cbar_kws={'label': metric})
    ax.set_title(f'{metric} by Site and Model')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

plt.tight_layout()
fpath = os.path.join(OUT_DIR, "plot_heatmap_persite.png")
plt.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fpath}")

# ── 7d. Scatter: obs vs sim (all sites, each model) ─────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
fig.suptitle('Observed vs Simulated LE — All Sites Pooled (v3.2)', fontsize=14, fontweight='bold')

obs_all = combined['LE_Obs'].values.astype(float)
lim = [0, np.nanpercentile(obs_all, 99)]

for ax, model, color in zip(axes, MODELS, COLORS):
    if model not in combined.columns:
        continue
    sim_all = combined[model].values.astype(float)
    mask = np.isfinite(obs_all) & np.isfinite(sim_all)
    o, s = obs_all[mask], sim_all[mask]
    ax.hexbin(o, s, gridsize=60, cmap='Blues', mincnt=1)
    ax.plot(lim, lim, 'r--', linewidth=1.5, label='1:1 line')
    m = compute_metrics(o, s)
    ax.set_title(f'{model}\nRMSE={m["RMSE"]:.1f}  KGE={m["KGE"]:.3f}  r={m["r"]:.3f}', fontsize=11)
    ax.set_xlabel('Observed LE (W/m²)')
    ax.set_ylabel('Simulated LE (W/m²)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.legend(fontsize=9)
    ax.grid(linestyle='--', alpha=0.4)

plt.tight_layout()
fpath = os.path.join(OUT_DIR, "plot_scatter_obs_vs_sim.png")
plt.savefig(fpath, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {fpath}")

# ─────────────────────────────────────────────
#  DONE
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"  Analysis complete. Outputs in:\n  {OUT_DIR}")
print("=" * 70)
