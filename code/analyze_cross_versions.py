import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Directories
dir_v1 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_MultiPET_v3.2.1"
dir_v2 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_SinglePET_v12.3_DynamicBeta"
dir_v3 = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/output_L6_OWUS_SinglePET_v13.3_DynamicBeta"

out_dir = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/cross_version_analysis_results"
os.makedirs(out_dir, exist_ok=True)

# Columns to use for each version
models_v1 = {"LE_PTJPL_SM": "v1 (PTJPL_SM)", "LE_SM_OPT": "v1 (SM_OPT)"}
models_v2 = {"LE_Base_BF_PT_PTJPL_Lin_Stat_TC": "v2 (Base_BF Lin Stat)"}
models_v3 = {
    "SM_Base_BF_PT_PTJPL_Lin_Stat_TC": "v3 (SM_Base Lin Stat)",
    "VODC_Base_BF_PT_PTJPL_Lin_Stat_TC": "v3 (VODC_Base Lin Stat)",
    "VODL_Base_BF_PT_PTJPL_Lin_Stat_TC": "v3 (VODL_Base Lin Stat)"
}

def calculate_stats(obs, model):
    df = pd.DataFrame({"o": obs, "m": model}).dropna()
    if len(df) < 10:
        return {"RMSE": np.nan, "Bias": np.nan, "r": np.nan}
    o, m = df["o"].astype(float), df["m"].astype(float)
    rmse = np.sqrt(((m - o) ** 2).mean())
    bias = (m - o).mean()
    r = np.corrcoef(o, m)[0, 1] if o.std() > 0 and m.std() > 0 else np.nan
    return {"RMSE": rmse, "Bias": bias, "r": r}

def get_csv(d, site):
    files = glob.glob(os.path.join(d, site, "*_results.csv"))
    return files[0] if files else None

# Specific 17 sites requested by the user
target_sites = [
    "adelaideriver", "alpinepeatland", "capetribulation", "dalypasture",
    "emerald", "foggdam", "gatumpasture", "greatwesternwoodlands",
    "litchfield", "reddirtmelonfarm", "ridgefield", "silverplains",
    "sturtplains", "wallabycreek", "whroo", "yarramundiirrigated", "yanco"
]

# Intersect valid directories from Target Sites only
valid_sites = []
for s in target_sites:
    v1_exists = os.path.exists(os.path.join(dir_v1, s))
    v2_exists = os.path.exists(os.path.join(dir_v2, s))
    v3_exists = os.path.exists(os.path.join(dir_v3, s))
    if v1_exists and v2_exists and v3_exists:
        valid_sites.append(s)
        
print(f"Found {len(valid_sites)} of the 17 requested sites with data in all 3 versions.")

all_metrics = []

for site in valid_sites:
    f1 = get_csv(dir_v1, site)
    f2 = get_csv(dir_v2, site)
    f3 = get_csv(dir_v3, site)
    
    if not (f1 and f2 and f3):
        continue

    df1 = pd.read_csv(f1, parse_dates=["time"]).set_index("time")
    df2 = pd.read_csv(f2, parse_dates=["time"]).set_index("time")
    df3 = pd.read_csv(f3, parse_dates=["time"]).set_index("time")
    
    # Merge datasets
    df_merged = df1[["LE_Obs", "LE_PTJPL_Base"] + list(models_v1.keys())].join(
        df2[list(models_v2.keys())], how="inner", rsuffix='_v2'
    ).join(
        df3[list(models_v3.keys())], how="inner", rsuffix='_v3'
    )
    
    # Calculate Metrics
    obs = df_merged["LE_Obs"]
    models_to_eval = {"LE_PTJPL_Base": "v1/2/3 Base"}
    models_to_eval.update(models_v1)
    models_to_eval.update(models_v2)
    models_to_eval.update(models_v3)
    
    for m_col, m_label in models_to_eval.items():
        if m_col in df_merged:
            stats = calculate_stats(obs, df_merged[m_col])
            stats["Site"] = site
            stats["Model"] = m_label
            stats["Column"] = m_col
            all_metrics.append(stats)
            
    # Timeseries Plot (Resampled to 10-day averages to look cleaner)
    df_plot = df_merged.resample('10D').mean()
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_plot.index, df_plot["LE_Obs"], 'ko-', label='Observed (LE_Obs)', markersize=3, linewidth=1.5, alpha=0.8)
    plt.plot(df_plot.index, df_plot["LE_PTJPL_Base"], 'r-', label='Base (LE_PTJPL_Base)', linewidth=1)
    
    if "LE_PTJPL_SM" in df_plot:
        plt.plot(df_plot.index, df_plot["LE_PTJPL_SM"], 'b--', label='v1 (LE_PTJPL_SM)', linewidth=1)
    if "LE_Base_BF_PT_PTJPL_Lin_Stat_TC" in df_plot:
        plt.plot(df_plot.index, df_plot["LE_Base_BF_PT_PTJPL_Lin_Stat_TC"], 'g--', label='v2 (LE_Base_BF_PT_PTJPL_Lin_Stat_TC)', linewidth=1)
    if "SM_Base_BF_PT_PTJPL_Lin_Stat_TC" in df_plot:
        plt.plot(df_plot.index, df_plot["SM_Base_BF_PT_PTJPL_Lin_Stat_TC"], 'm--', label='v3 (SM_Base_BF_PT_PTJPL_Lin_Stat_TC)', linewidth=1)
    if "VODC_Base_BF_PT_PTJPL_Lin_Stat_TC" in df_plot:
        plt.plot(df_plot.index, df_plot["VODC_Base_BF_PT_PTJPL_Lin_Stat_TC"], 'c:', label='v3 (VODC_Base_BF_PT_PTJPL_Lin_Stat_TC)', linewidth=1)
        
    plt.title(f"{site.upper()} - Latent Heat Flux Comparison Across Versions (10-Day Avg)")
    plt.ylabel("Latent Heat Flux (W/m2)")
    plt.xlabel("Date")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{site}_timeseries.png"), dpi=150)
    plt.close()
    
df_metrics = pd.DataFrame(all_metrics)
df_metrics = df_metrics[["Site", "Model", "Column", "RMSE", "Bias", "r"]]
df_metrics.to_csv(os.path.join(out_dir, "cross_version_metrics_summary.csv"), index=False)
print(f"Processing complete! Outputs saved to {out_dir}")
