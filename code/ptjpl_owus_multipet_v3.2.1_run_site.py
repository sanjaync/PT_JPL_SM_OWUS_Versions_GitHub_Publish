#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ptjpl_owus_multipet_v3.2.1_run_site.py
-------------------------------------------------------------------
Features (v3.2.1 - Refined):
1. PAW Normalization for s_arr (s_star/s_wilt compatible).
2. Dry-day metrics evaluation (Precip == 0 and lag Precip == 0).
3. NO Multi-PET sources. No FAO, PM, ENS, or internal PT variations.
4. **Model Variants**: Produce only the following 4 models for the output and metrics:
   1. `PTJPL_Base`: Original PT-JPL.
   2. `PTJPL_SM`: Original PT-JPL-SM.
   3. `PTJPL_SM_OPT`: OWUS OPT configuration using SM soil/interception, with **no PET capping**.
   4. `PTJPL_SM_BF`: OWUS BF configuration using SM soil/interception, with **no PET capping**.
5. Swapped raw volumetric soil moisture observation (SM_arr) with normalized Plant Available Water (s_arr) in the weighting framework.
"""

import os
import sys
import glob
import re
import types
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETUP OFFLINE ENVIRONMENT
# =====================================================================
os.environ["PTJPLSM_OFFLINE"] = "1"
os.environ["PTJPL_OFFLINE"]   = "1"
os.environ["NO_DOWNLOAD"]     = "1"

try:
    import h5netcdf
    print("[INFO] h5netcdf module loaded successfully.")
except ImportError:
    print("[WARN] h5netcdf is not installed.")

try:
    from PTJPLSM.process_PTJPLSM_table import process_PTJPLSM_table
    from PTJPL.process_PTJPL_table import process_PTJPL_table
except ImportError:
    try:
        from PTJPLSM import PTJPLSM  # type: ignore
    except ImportError:
        print("[ERROR] Could not import PTJPLSM. Check your PYTHONPATH.")
        sys.exit(1)

# =====================================================================
# 2. CONFIGURATION & FILE PATHS
# =====================================================================
UNIFIED_CSV = (
    "/fs04/scratch2/et97/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "OzFluxStations_NDVI_Soil_MetaUnified_withCanopy_LAI_FPAR_2003_2024_DAILY_FIXED.csv"
)
L6_DATA_DIR = "/fs04/scratch2/et97/oldscratch/Ozflux_data_full/L6"
OWUS_BF_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia_agent_3_scientific/output_corrected/combined/results_bf__ozflux_1.csv"
OWUS_OPT_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia_agent_3_scientific/output_corrected/combined/results_opt__ozflux_1.csv"

# Output directory for version 3.2.1
OUTPUT_DIR = (
    "/fs04/scratch2/et97/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "output_L6_OWUS_MultiPET_v3.2.1"
)

# Constants
SIGMA = 5.670374419e-8
TOPT_FALLBACK = 30.0
LV = 2.45e6
RHO_W = 1000
FE_VALID_MIN = -100.0
FE_VALID_MAX = 1000.0

SITE_MAPPING = {
    "gatumpasture": "AU-Gat", "otway": "AU-Otw", "silverplains": "AU-SiP", "emerald": "AU-Eme",
    "alpinepeatland": "AU-Alp", "riggscreek": "AU-Rig", "yarramundiirrigated": "AU-YarI",
    "yarramundicontrol": "AU-YarC", "yanco": "AU-Ync", "dalypasture": "AU-DaP", "foggdam": "AU-Fog",
    "sturtplains": "AU-Stp", "wallabycreek": "AU-Wal", "collie": "AU-Col", "boyagin": "AU-Boy",
    "adelaideriver": "AU-Adr", "dalyuncleared": "AU-DaS", "litchfield": "AU-Lit",
    "greatwesternwoodlands": "AU-Gre", "ridgefield": "AU-Rgf", "capetribulation": "AU-Ctr",
    "cowbay": "AU-Cow", "whroo": "AU-Whr", "tumbarumba": "AU-Tum", "loxton": "AU-Lox",
    "warra": "AU-War", "gingin": "AU-Gin", "reddirtmelonfarm": "AU-RDF", "howardsprings": "AU-How",
    "cumberlandplain": "AU-Cum", "robsoncreek": "AU-Rob", "alicespringsmulga": "AU-ASM",
    "titreeeast": "AU-TTE", "digbyplantation": "AU-Dig", "wombatstateforest": "AU-Wom",
}

# =====================================================================
# 3. UTILITY & MATHEMATICAL FUNCTIONS
# =====================================================================
def norm_site_string(s):
    if not s: return ""
    return re.sub(r"[\s_]+", "", str(s).strip().lower())

def strip_trailing_digits(s):
    return re.sub(r"\d+$", "", s)

def get_au_season(month):
    if month in [12, 1, 2]: return "SUMMER"
    if month in [3, 4, 5]: return "AUTUMN"
    if month in [6, 7, 8]: return "WINTER"
    return "SPRING"

def find_nc_files(site_name, search_dir):
    target_base = strip_trailing_digits(norm_site_string(site_name))
    files = glob.glob(os.path.join(search_dir, "*_L6.nc"))
    return sorted([f for f in files
                   if strip_trailing_digits(
                       norm_site_string(
                           os.path.basename(f).replace("_L6.nc", "").replace(".nc", "")
                       )) == target_base])

def calculate_stats(obs, model):
    df = pd.DataFrame({"o": obs, "m": model}).dropna()
    nans = {k: np.nan for k in ["RMSE", "Bias", "R2", "ubRMSE", "MAE", "NSE", "KGE", "r", "alpha", "beta"]}
    if len(df) < 10:
        return nans
    o, m = df["o"].astype(float), df["m"].astype(float)

    rmse   = np.sqrt(((m - o) ** 2).mean())
    bias   = (m - o).mean()
    mae    = np.abs(m - o).mean()
    ubrmse = np.sqrt(max(rmse**2 - bias**2, 0.0))
    
    std_o, std_m   = o.std(), m.std()
    mean_o, mean_m = o.mean(), m.mean()

    r  = np.corrcoef(o, m)[0, 1] if std_o > 0 and std_m > 0 else np.nan
    r2 = (r ** 2) if np.isfinite(r) else np.nan
    denom_nse = ((o - mean_o) ** 2).sum()
    nse = 1 - (((m - o) ** 2).sum() / denom_nse) if denom_nse > 0 else np.nan

    alpha = (std_m / std_o) if std_o > 0 else np.nan
    beta  = (mean_m / mean_o) if mean_o != 0 else np.nan
    kge = (1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
           if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta)
           else np.nan)

    return {"RMSE": rmse, "Bias": bias, "R2": r2, "ubRMSE": ubrmse, "MAE": mae,
            "NSE": nse, "KGE": kge, "r": r, "alpha": alpha, "beta": beta}

def derive_topt(df):
    gpp = df.get("GPP_SOLO", df.get("GPP"))
    ta  = df.get("Ta")
    if gpp is None or ta is None:
        return TOPT_FALLBACK
    df_t = pd.DataFrame({"Ta": ta, "GPP": gpp}).dropna()
    df_t = df_t[(df_t["Ta"] > -5) & (df_t["Ta"] < 50) & (df_t["GPP"] > 0)]
    if df_t.empty:
        return TOPT_FALLBACK
    gpp_by_bin = df_t.groupby(df_t["Ta"].round(0))["GPP"].mean()
    return float(gpp_by_bin.idxmax()) if not gpp_by_bin.empty else TOPT_FALLBACK

# =====================================================================
# 4. CORE PROCESSING LOOP
# =====================================================================
def process_single_nc(site_name, nc_path, df_owus_bf, df_owus_opt, meta_all):
    site_suffix = os.path.basename(nc_path).replace("_L6.nc", "")
    site_label  = f"{site_name}::{site_suffix}"

    site_dir     = os.path.join(OUTPUT_DIR, norm_site_string(site_suffix))
    os.makedirs(site_dir, exist_ok=True)
    out_csv      = os.path.join(site_dir, f"{site_suffix}_v3.2.1_results.csv")
    metrics_txt  = os.path.join(site_dir, f"{site_suffix}_v3.2.1_metrics.txt")
    plot_path    = os.path.join(site_dir, f"{site_suffix}_v3.2.1_timeseries.png")

    # Always overwrite for version 3.2.1 refinements
    print(f"\n=== PROCESSING: {site_label} ===")

    ds = None
    for eng in ["h5netcdf", None, "scipy"]:
        try:
            ds = xr.open_dataset(nc_path, engine=eng)
            break
        except Exception:
            continue
    if ds is None:
        print(f"  [ERROR] Failed to open {nc_path}")
        return

    lat_val = float(ds.attrs.get("latitude",
                    ds.get("latitude", np.array(0.0)).values.flat[0]))
    lon_val = float(ds.attrs.get("longitude",
                    ds.get("longitude", np.array(0.0)).values.flat[0]))
    ds = ds.squeeze(drop=True)

    # -- OWUS parameter lookup --
    def get_owus(df_owus, label=""):
        if df_owus.empty:
            return 0.65, 0.20, 0.95
        c_name = strip_trailing_digits(norm_site_string(site_suffix))
        t_id   = SITE_MAPPING.get(c_name, site_suffix)
        m = df_owus[df_owus["siteID"].str.lower() == t_id.lower()]
        if not m.empty:
            b = m.iloc[0]
            print(f"  [INFO] OWUS {label} match: {b['siteID']} (s*={b['s_star_median']:.3f})")
            return (float(b["s_star_median"]), float(b["s_wilt_median"]), float(b["beta_ww_median"]))
        # Spatial fallback
        dist = np.sqrt((df_owus["lon"] - lon_val)**2 + (df_owus["lat"] - lat_val)**2)
        b = df_owus.loc[dist.idxmin()]
        print(f"  [INFO] OWUS {label} spatial fallback: {b['siteID']}")
        return (float(b["s_star_median"]), float(b["s_wilt_median"]), float(b["beta_ww_median"]))

    bf_s_star, bf_s_wilt, bf_beta_ww   = get_owus(df_owus_bf, "BF")
    opt_s_star, opt_s_wilt, opt_beta_ww = get_owus(df_owus_opt, "OPT")

    # -- Extract data --
    vars_needed = ["Fsd", "Fsu", "Fld", "Flu", "Fn", "Ta", "RH", "Sws", "Fe", "Precip", "GPP_SOLO", "GPP"]
    df_raw = ds[[v for v in vars_needed if v in ds]].to_dataframe()
    if "time" in df_raw.columns:
        df_raw = df_raw.set_index("time")

    # Half-hourly precip for masking
    p_series_hh = (ds["Precip"].to_series() if "Precip" in ds else pd.Series(0.0, index=df_raw.index))
    if "time" not in p_series_hh.index.names: p_series_hh.index = df_raw.index

    df_d = df_raw.resample("1D").mean(numeric_only=True)
    if "Fe" in df_d:
        bad = (df_d["Fe"] < FE_VALID_MIN) | (df_d["Fe"] > FE_VALID_MAX)
        df_d.loc[bad, "Fe"] = np.nan

    df_d["Precip"] = p_series_hh.resample("1D").sum()
    df_d["lat"], df_d["lon"] = lat_val, lon_val

    # Derived variables
    if {"Fsd", "Fsu"}.issubset(df_d.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            alb_raw = df_d["Fsu"] / df_d["Fsd"]
        alb_clean = alb_raw.replace([np.inf, -np.inf], np.nan).dropna()
        alb_clean = alb_clean[(alb_clean > 0) & (alb_clean < 1)]
        amin = float(alb_clean.quantile(0.01)) if len(alb_clean) > 10 else 0.05
        amax = float(alb_clean.quantile(0.99)) if len(alb_clean) > 10 else 0.80
        df_d["albedo"] = alb_raw.clip(amin, amax)

    if "Flu" in df_d:
        df_d["ST_C"] = ((df_d["Flu"] / (0.98 * SIGMA)) ** 0.25) - 273.15

    if "Fn" in df_d:
        df_d["Rn"] = df_d["Fn"]
    elif {"Fsd", "Fsu"}.issubset(df_d.columns):
        df_d["Rn"] = (df_d["Fsd"] - df_d["Fsu"] + (df_d["Fld"] - df_d["Flu"] if {"Fld", "Flu"}.issubset(df_d.columns) else 0))
    else:
        df_d["Rn"] = np.nan

    if "RH" in df_d and df_d["RH"].max() > 1.5:
        df_d["RH"] /= 100.0

    # -- Metadata alignment --
    s_norm = norm_site_string(site_name)
    s_base = strip_trailing_digits(s_norm)
    meta_all_curr = meta_all.copy()
    m_site = meta_all_curr[meta_all_curr["site_norm"] == s_norm].copy()
    if m_site.empty: m_site = meta_all_curr[meta_all_curr["site_base"] == s_base].copy()
    if m_site.empty: m_site = meta_all_curr[meta_all_curr["site_norm"].str.contains(s_base)]
    if m_site.empty:
        print(f"  [ERROR] No metadata for {site_name}")
        return

    m_site["date"] = pd.to_datetime(m_site["date"])
    m_align = (m_site.set_index("date").sort_index().reindex(df_d.index).interpolate(limit_direction="both"))

    for c in ["ndvi", "lai", "fpar", "canopy_height", "fc_000_005", "fc_0_30cm", "wp_000_005", "wp_0_30cm"]:
        if c in m_align: m_align[c] = pd.to_numeric(m_align[c], errors="coerce")

    df_d["NDVI"]  = m_align.get("ndvi", np.nan).clip(0, 1)
    df_d["LAI"]   = m_align.get("lai", np.nan)
    df_d["FPAR"]  = m_align.get("fpar", np.nan).clip(0, 1)
    df_d["canopy_height_meters"] = m_align.get("canopy_height", 0.5)
    df_d["field_capacity"] = m_align.get("fc_000_005", m_align.get("fc_0_30cm", 0.30))
    df_d["wilting_point"] = m_align.get("wp_000_005", m_align.get("wp_0_30cm", 0.10))

    if "Sws" in df_d:
        df_d["SM"] = df_d["Sws"].clip(lower=df_d["wilting_point"], upper=df_d["field_capacity"])
    else:
        df_d["SM"] = np.nan

    df_d["fAPARmax"] = (df_d["FPAR"].quantile(0.95) if "FPAR" in df_d and df_d["FPAR"].notna().any() else (df_d["NDVI"].quantile(0.95) if "NDVI" in df_d else 0.7))
    df_d["Topt_C"] = derive_topt(df_d)

    # Emissivity
    ndvi_s = df_d["NDVI"].dropna()
    if not ndvi_s.empty:
        n_min, n_max = float(ndvi_s.quantile(0.05)), float(ndvi_s.quantile(0.95))
        if n_max <= n_min: n_min, n_max = 0.10, 0.85
        fvc = ((df_d["NDVI"] - n_min) / (n_max - n_min)).clip(0, 1) ** 2
        df_d["emissivity"] = 0.97 * (1 - fvc) + 0.99 * fvc
    else:
        df_d["emissivity"] = 0.98

    # Prepare model input
    input_df = df_d.copy()
    input_df["time_UTC"] = pd.to_datetime(input_df.index, utc=True)
    if "Ta" in input_df: input_df.rename(columns={"Ta": "Ta_C"}, inplace=True)

    req = ["NDVI", "ST_C", "albedo", "Ta_C", "RH", "SM", "Rn", "fAPARmax", "lat", "lon"]
    missing = [c for c in req if c not in input_df]
    if missing:
        print(f"  [ERROR] Missing columns: {missing}")
        return

    df_run = input_df[input_df[req].notna().all(axis=1)].copy()
    if df_run.empty:
        print("  [ERROR] No valid rows after filtering")
        return

    # Dry-day and wet-day masks
    precip_daily = df_d.reindex(df_run.index)["Precip"]
    dry_mask = ((precip_daily == 0) & (precip_daily.shift(1) == 0)).fillna(False)
    wet_mask = ((precip_daily > 0) | (precip_daily.shift(1) > 0)).fillna(False)
    n_dry  = dry_mask.sum()
    n_wet  = wet_mask.sum()

    # Mock modules
    m_gedi = types.ModuleType("gedi_canopy_height")
    m_gedi.GEDI_DOWNLOAD_DIRECTORY = "/tmp/mock/"
    m_gedi.load_canopy_height = (lambda **k: np.full(len(df_run), df_run["canopy_height_meters"].iloc[0]))
    sys.modules["gedi_canopy_height"] = m_gedi

    m_scw = types.ModuleType("soil_capacity_wilting")
    m_scw.DEFAULT_DOWNLOAD_DIRECTORY = "/tmp/mock/"
    m_scw.load_field_capacity = (lambda **k: np.full(len(df_run), df_run["field_capacity"].iloc[0]))
    m_scw.load_wilting_point = (lambda **k: np.full(len(df_run), df_run["wilting_point"].iloc[0]))
    sys.modules["soil_capacity_wilting"] = m_scw

    # ---- RUN BASE MODELS ----
    print(f"  [INFO] Running PTJPL-SM & PTJPL (Base)…")
    res_sm   = process_PTJPLSM_table(df_run)
    res_base = process_PTJPL_table(df_run)

    # Extract components
    col_can  = next((c for c in ["LE_canopy", "LE_canopy_Wm2"] if c in res_base.columns), "LE_canopy")
    sm_soil_col = next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_sm.columns), "LE_soil")
    sm_int_col  = next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_sm.columns), "LE_interception")

    LE_can_orig  = res_base[col_can].values
    LE_soil_sm = res_sm[sm_soil_col].values if sm_soil_col in res_sm.columns else np.zeros(len(df_run))
    LE_int_sm  = res_sm[sm_int_col].values  if sm_int_col in res_sm.columns else np.zeros(len(df_run))

    # PAW Normalization
    wp, fc, SM_arr = df_run["wilting_point"].values, df_run["field_capacity"].values, df_run["SM"].values
    den = np.maximum(fc - wp, 1e-6)
    s_arr = np.clip((SM_arr - wp) / den, 0.0, 1.0)
    fM, RH_arr = np.clip(df_run["FPAR"].values / df_run["fAPARmax"].values, 0.0, 1.0), df_run["RH"].values
    
    # Swapped SM_arr (raw observation) with s_arr (Plant Available Water)
    weight = RH_arr ** (4.0 * (1.0 - s_arr) * (1.0 - RH_arr))

    # ---- CORE MODELS (No Single/Multi-PET variances, No Capping) ----
    # 1. Base (PT-JPL)
    # 2. SM (PT-JPL-SM)
    # 3. OPT (SM Soil/Int + OWUS OPT Canopy)
    # 4. BF (SM Soil/Int + OWUS BF Canopy)

    def calc_le_owus_raw(s_star, s_wilt, beta_ww):
        f_OWUS = np.zeros_like(s_arr)
        f_OWUS = np.where(s_arr > s_star, beta_ww, f_OWUS)
        mask_lim = (s_arr > s_wilt) & (s_arr <= s_star)
        d = max(s_star - s_wilt, 1e-6)
        f_OWUS = np.where(mask_lim, beta_ww * (s_arr - s_wilt) / d, f_OWUS)
        f_TRM = (1.0 - weight) * fM + weight * f_OWUS
        LE_can = np.divide(LE_can_orig * f_TRM, fM, out=np.zeros_like(LE_can_orig), where=(fM > 0))
        total = LE_soil_sm + LE_int_sm + LE_can
        return LE_can, LE_soil_sm, LE_int_sm, total

    opt_tc, opt_sc, opt_cc, le_opt = calc_le_owus_raw(opt_s_star, opt_s_wilt, opt_beta_ww)
    bf_tc, bf_sc, bf_cc, le_bf  = calc_le_owus_raw(bf_s_star, bf_s_wilt, bf_beta_ww)

    # ---- SM regime classification ----
    regime_sm = np.full(s_arr.shape, "nan", dtype=object)
    regime_sm_code = np.full(s_arr.shape, -1, dtype=int)
    m_valid = np.isfinite(s_arr)
    m_vdry = m_valid & (s_arr <= opt_s_wilt)
    m_trans = m_valid & (s_arr > opt_s_wilt) & (s_arr <= opt_s_star)
    m_wet  = m_valid & (s_arr > opt_s_star)
    regime_sm[m_vdry] = "very_dry"
    regime_sm[m_trans] = "transition"
    regime_sm[m_wet] = "wet"
    regime_sm_code[m_vdry] = 0
    regime_sm_code[m_trans] = 1
    regime_sm_code[m_wet] = 2

    sm_vdry_mask = (regime_sm_code == 0)
    sm_trans_mask = (regime_sm_code == 1)
    sm_wet_mask = (regime_sm_code == 2)

    # ---- Assemble output ----
    output = pd.DataFrame(index=df_run.index)
    output["LE_Obs"]        = df_d.reindex(df_run.index)["Fe"]
    output["LE_PTJPL_Base"] = res_base["LE"] if "LE" in res_base.columns else res_base["LE_Wm2"]
    output["LE_PTJPL_SM"]   = res_sm["LE"]
    output["LE_SM_OPT"]     = le_opt
    output["LE_SM_OPT_TC"]  = opt_tc
    output["LE_SM_OPT_SC"]  = opt_sc
    output["LE_SM_OPT_CC"]  = opt_cc
    output["LE_SM_BF"]      = le_bf
    output["LE_SM_BF_TC"]   = bf_tc
    output["LE_SM_BF_SC"]   = bf_sc
    output["LE_SM_BF_CC"]   = bf_cc
    output["Precip"]        = precip_daily
    output["dry_day"]       = dry_mask.astype(int)
    output["wet_day"]       = wet_mask.astype(int)
    output["SM_PAW"]        = s_arr
    output["SM_regime_threshold_set"] = "OPT"
    output["SM_s_wilt_reg"] = float(opt_s_wilt)
    output["SM_s_star_reg"] = float(opt_s_star)
    output["regime_sm"]      = regime_sm
    output["regime_sm_code"] = regime_sm_code
    output["Year"]   = output.index.year
    output["Season"] = output.index.month.map(get_au_season)

    output.to_csv(out_csv, float_format="%.4f")

    # ---- METRICS ----
    models = ["LE_PTJPL_Base", "LE_PTJPL_SM", "LE_SM_OPT", "LE_SM_BF"]
    hdr = (f"{'Model':<15s} | RMSE | Bias |   R2 |  NSE |  KGE |    r |    a |    b")
    sep = "-" * 80

    def write_block(f, label, mask, models_list):
        if mask is not None and not isinstance(mask, slice):
            o_data = output["LE_Obs"][mask]
            n = int(mask.sum())
        else:
            o_data = output["LE_Obs"]
            n = len(output)

        f.write(f"\n=== {label} (n={n}) ===\n{hdr}\n{sep}\n")
        if n < 5:
            f.write("  [NOT ENOUGH DATA]\n")
            return

        for m in models_list:
            if mask is not None and not isinstance(mask, slice):
                st = calculate_stats(o_data, output[m][mask])
            else:
                st = calculate_stats(o_data, output[m])
            f.write(f"{m.replace('LE_',''):<15s} | {st['RMSE']:4.1f} | {st['Bias']:4.1f}"
                    f" | {st['R2']:4.2f} | {st['NSE']:4.2f} | {st['KGE']:4.2f}"
                    f" | {st['r']:.2f} | {st['alpha']:.2f}"
                    f" | {st['beta']:.2f}\n")

    with open(metrics_txt, "w") as f:
        f.write(f"Site: {site_name} | L6: {site_suffix}\n")
        f.write(f"Total rows: {len(output)} | Dry(precip): {n_dry} | Wet(precip): {n_wet}\n")
        f.write(f"SM regime threshold set: OPT | s_wilt={opt_s_wilt:.3f} | s_star={opt_s_star:.3f}\n")
        f.write(f"Total models: {len(models)}\n")

        write_block(f, "ALL DAYS", None, models)
        write_block(f, "DRY DAYS (precip)", dry_mask, models)
        write_block(f, "WET DAYS (precip)", wet_mask, models)
        write_block(f, "VERY DRY (SM regime)", sm_vdry_mask, models)
        write_block(f, "TRANSITION (SM regime)", sm_trans_mask, models)
        write_block(f, "WET (SM regime)", sm_wet_mask, models)

    # ---- PLOT ----
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        obs = output["LE_Obs"]
        if obs.notna().any(): ax.plot(obs.index, obs, "k-", lw=1.2, label="Obs")
        ax.plot(output.index, output["LE_PTJPL_Base"], lw=1, alpha=0.5, label="PTJPL_Base")
        ax.plot(output.index, output["LE_PTJPL_SM"],   lw=1, alpha=0.5, label="PTJPL_SM")
        ax.plot(output.index, output["LE_SM_OPT"],     lw=1.2, label="SM_OPT")
        ax.plot(output.index, output["LE_SM_BF"],      lw=1.2, label="SM_BF")
        ax.set_title(f"{site_suffix} (v3.2.1) - 4 Core Models (No PET Suffixes)")
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    except Exception as e: print(f"  [WARN] Plot failed: {e}")

def process_site(site_name):
    print(f"=== INITIALIZING SITE: {site_name} ===")
    cols = ["siteID", "lat", "lon", "s_star_median", "s_wilt_median", "beta_ww_median"]
    df_bf  = (pd.read_csv(OWUS_BF_CSV, usecols=cols).dropna() if os.path.exists(OWUS_BF_CSV) else pd.DataFrame())
    df_opt = (pd.read_csv(OWUS_OPT_CSV, usecols=cols).dropna() if os.path.exists(OWUS_OPT_CSV) else pd.DataFrame())
    meta_all = pd.read_csv(UNIFIED_CSV, low_memory=False)
    meta_all.columns = [c.strip().lower() for c in meta_all.columns]
    meta_all["site_norm"] = meta_all["site"].apply(norm_site_string)
    meta_all["site_base"] = meta_all["site_norm"].apply(strip_trailing_digits)
    nc_files = find_nc_files(site_name, L6_DATA_DIR)
    if not nc_files: return
    os.makedirs(OUTPUT_DIR, exist_ok=True); [process_single_nc(site_name, nc, df_bf, df_opt, meta_all) for nc in nc_files]

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    process_site(sys.argv[1])
