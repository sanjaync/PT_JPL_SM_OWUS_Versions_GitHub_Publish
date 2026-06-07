#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ptjpl_owus_multipet_v3.2.1.1_run_site.py
-------------------------------------------------------------------
Features (v3.2.1.1 - Analytical OWUS):
1. PAW Normalization for s_arr (s_star/s_wilt replaced by Pi groups).
2. Dry-day metrics evaluation (Precip == 0 and lag Precip == 0).
3. **Model Variants**: PTJPL_Base, PTJPL_SM, PTJPL_SM_OPT, PTJPL_SM_BF.
4. **Analytical OWUS**: Implementation of Equation (4) for f_ww and 
   Equation (5) for sigmoidal beta(s) inversion using brentq.
"""

import os
import sys
import glob
import re
import types
import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import brentq

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
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "OzFluxStations_NDVI_Soil_MetaUnified_withCanopy_LAI_FPAR_2003_2024_DAILY_FIXED.csv"
)
L6_DATA_DIR = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6"
OWUS_BF_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia_agent_3_scientific/output_corrected/combined/results_bf__ozflux_1.csv"
OWUS_OPT_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia_agent_3_scientific/output_corrected/combined/results_opt__ozflux_1.csv"

# Output directory for version 3.2.1.1
OUTPUT_DIR = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "output_L6_OWUS_MultiPET_v3.2.1.1"
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

# ---- NEW ANALYTICAL OWUS FUNCTIONS (VECTORIZED v3.2.1.1) ----

def calculate_f_ww(pi_R, pi_F):
    """
    Implements Equation (4): Determining Maximum Transpiration Plateau.
    f_ww reflects degree of transpiration downregulation in well-watered conditions.
    """
    term1 = 1.0 + (pi_F / 2.0)
    # Ensure the value inside sqrt is non-negative to avoid imaginary results
    sqrt_val = np.sqrt(np.maximum(0.0, (pi_F / 2.0 + 1.0)**2 - 2.0 * pi_F * pi_R))
    f_ww = 1.0 - (1.0 / (2.0 * pi_R)) * (term1 - sqrt_val)
    return np.clip(f_ww, 0.0, 1.0)

def owus_s_of_beta(beta, pi_R, pi_F, pi_T, pi_S, b):
    """
    Equation (5): Maps beta to soil saturation s.
    """
    # Handle scalar or array beta
    beta = np.atleast_1d(beta)
    s_out = np.zeros_like(beta)
    
    mask = beta > 1e-7
    if not np.any(mask): return s_out if len(s_out) > 1 else s_out[0]
    
    b_m = beta[mask]
    denom_inner = 1.0 - (1.0 - b_m) * pi_R
    # Avoid div by zero in extreme trait combinations
    denom_inner = np.where(np.abs(denom_inner) < 1e-9, 1e-9, denom_inner)
    
    bracket = 2.0 * (1.0 - b_m) - (b_m * pi_F) / denom_inner
    # sqrt term accounts for transport limitations
    sqrt_val = np.sqrt(np.maximum(0, 1.0 + (4.0 * b_m * pi_S**2 / pi_T) * bracket))
    
    # Power is -1/b where b is the soil pore size parameter
    s_val = (pi_T / (2.0 * b_m * pi_S) * (sqrt_val - 1.0))**(-1.0 / b)
    s_out[mask] = np.clip(s_val, 0.0, 1.0)
    
    return s_out if len(s_out) > 1 else s_out[0]

def _get_beta_single_set(proxy_arr, pi_params, b_soil):
    pR, pF, pT, pS = pi_params
    f_ww = calculate_f_ww(pR, pF)
    beta_grid = np.linspace(1e-7, float(f_ww), 1000)
    s_grid = owus_s_of_beta(beta_grid, pR, pF, pT, pS, b_soil)
    beta_values = np.interp(proxy_arr, s_grid, beta_grid, left=0.0, right=float(f_ww))
    return beta_values, float(f_ww)

def get_beta_vectorized(s_arr, pi_params):
    """
    Solves for beta across the entire time series using interpolation
    of the analytical sigmoidal curve. Handles both static and dynamic Pi parameters.
    """
    pR, pF, pT, pS, b_soil = pi_params
    if any(isinstance(x, np.ndarray) for x in [pR, pF, pT, pS]):
        # Dynamic case: parameters vary by regime (time step)
        N = len(s_arr)
        pR_arr = np.full(N, pR) if not isinstance(pR, np.ndarray) else pR
        pF_arr = np.full(N, pF) if not isinstance(pF, np.ndarray) else pF
        pT_arr = np.full(N, pT) if not isinstance(pT, np.ndarray) else pT
        pS_arr = np.full(N, pS) if not isinstance(pS, np.ndarray) else pS
        
        params_stack = np.stack([pR_arr, pF_arr, pT_arr, pS_arr]).T
        unique_pis, inverse_indices = np.unique(params_stack, axis=0, return_inverse=True)
        beta_out = np.zeros_like(s_arr)
        f_ww_out = np.zeros_like(s_arr)
        for i, upi in enumerate(unique_pis):
            mask = (inverse_indices == i)
            b_val, f_val = _get_beta_single_set(s_arr[mask], upi, b_soil)
            beta_out[mask] = b_val
            f_ww_out[mask] = f_val
        return beta_out, f_ww_out
    else:
        return _get_beta_single_set(s_arr, [pR, pF, pT, pS], b_soil)

# =====================================================================
# 4. CORE PROCESSING LOOP
# =====================================================================
def process_single_nc(site_name, nc_path, df_owus_bf, df_owus_opt, meta_all):
    site_suffix = os.path.basename(nc_path).replace("_L6.nc", "")
    site_label  = f"{site_name}::{site_suffix}"

    site_dir     = os.path.join(OUTPUT_DIR, norm_site_string(site_suffix))
    os.makedirs(site_dir, exist_ok=True)
    out_csv      = os.path.join(site_dir, f"{site_suffix}_v3.2.1.1_results.csv")
    metrics_txt  = os.path.join(site_dir, f"{site_suffix}_v3.2.1.1_metrics.txt")
    plot_path    = os.path.join(site_dir, f"{site_suffix}_v3.2.1.1_timeseries.png")

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

    # -- UPDATED: OWUS Pi group lookup --
    def get_pi_groups(df_owus, label=""):
        if df_owus.empty:
            # Baseline Temperate Forest: pi_R, pi_F, pi_T, pi_S, b
            return 0.5, 1.0, 1e5, 200.0, 5.39
            
        c_name = strip_trailing_digits(norm_site_string(site_suffix))
        t_id   = SITE_MAPPING.get(c_name, site_suffix)
        m = df_owus[df_owus["siteID"].str.lower() == t_id.lower()]
        
        if m.empty:
            dist = np.sqrt((df_owus["lon"] - lon_val)**2 + (df_owus["lat"] - lat_val)**2)
            m = df_owus.loc[[dist.idxmin()]]
            
        b_row = m.iloc[0]
        print(f"  [INFO] OWUS {label} Pi-groups matched for {b_row['siteID']}")
        return (float(b_row["pi_R_median"]), float(b_row["pi_F_median"]), 
                float(b_row["pi_T_median"]), float(b_row["pi_S_median"]), float(b_row["b"]))

    pi_bf  = get_pi_groups(df_owus_bf, "BF")
    pi_opt = get_pi_groups(df_owus_opt, "OPT")

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
    
    # Weighting framework
    weight = RH_arr ** (4.0 * (1.0 - s_arr) * (1.0 - RH_arr))

    # ---- NEW ANALYTICAL OWUS CALCULATION ----
    def calc_le_owus_full(pi_set):
        pR, pF, pT, pS, b_soil = pi_set
        
        # Vectorized beta calculation (replaces the brentq loop)
        beta_full, f_ww = get_beta_vectorized(s_arr, pi_set)
        
        # Integrate with weighting framework
        f_TRM = (1.0 - weight) * fM + weight * beta_full
        le_can = np.divide(LE_can_orig * f_TRM, fM, out=np.zeros_like(LE_can_orig), where=(fM > 0))
        return LE_soil_sm + LE_int_sm + le_can

    le_opt = calc_le_owus_full(pi_opt)
    le_bf  = calc_le_owus_full(pi_bf)

    # ---- SM regime classification (Using OPT Pi groups to derive thresholds for backward compatibility) ----
    # Transition thresholds in sigmoidal curve are not sharp, so we use s at beta=0.05*f_ww and 0.95*f_ww as proxies.
    # Note: owus_s_of_beta returns scalar if input is scalar
    f_ww_opt = calculate_f_ww(pi_opt[0], pi_opt[1])
    s_wilt_prox = float(owus_s_of_beta(0.05 * f_ww_opt, *pi_opt))
    s_star_prox = float(owus_s_of_beta(0.95 * f_ww_opt, *pi_opt))

    regime_sm = np.full(s_arr.shape, "nan", dtype=object)
    regime_sm_code = np.full(s_arr.shape, -1, dtype=int)
    m_valid = np.isfinite(s_arr)
    m_vdry = m_valid & (s_arr <= s_wilt_prox)
    m_trans = m_valid & (s_arr > s_wilt_prox) & (s_arr <= s_star_prox)
    m_wet  = m_valid & (s_arr > s_star_prox)
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
    output["LE_SM_BF"]      = le_bf
    output["Precip"]        = precip_daily
    output["dry_day"]       = dry_mask.astype(int)
    output["wet_day"]       = wet_mask.astype(int)
    output["SM_PAW"]        = s_arr
    output["SM_regime_threshold_set"] = "Analytical_OPT"
    output["SM_s_wilt_reg"] = float(s_wilt_prox)
    output["SM_s_star_reg"] = float(s_star_prox)
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
        f.write(f"SM regime (proxies): s_wilt={s_wilt_prox:.3f} | s_star={s_star_prox:.3f}\n")
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
        ax.plot(output.index, output["LE_SM_OPT"],     lw=1.2, label="SM_OPT (Analytical)")
        ax.plot(output.index, output["LE_SM_BF"],      lw=1.2, label="SM_BF (Analytical)")
        ax.set_title(f"{site_suffix} (v3.2.1.1) - Full Analytical OWUS")
        ax.legend(loc="best", fontsize=9)
        plt.tight_layout()
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)
    except Exception as e: print(f"  [WARN] Plot failed: {e}")

def process_site(site_name):
    print(f"=== INITIALIZING SITE: {site_name} ===")
    # Note: Column 'b' in CSV is soil pore size parameter
    # Pi columns expected: pi_R_median, pi_F_median, pi_T_median, pi_S_median
    cols = ["siteID", "lat", "lon", "pi_R_median", "pi_F_median", "pi_T_median", "pi_S_median", "b"]
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
