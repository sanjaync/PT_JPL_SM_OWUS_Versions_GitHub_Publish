#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ptjpl_owus_single_pet_v12.3.1.1_run_site.py
-------------------------------------------------------------------
Version 12.3.1.1: SM-Only Grid Search + Analytical OWUS (Single PET)
- Replaces empirical shape functions (Exp/Sig) with Equation (5) Analytical Inversion.
- Incorporates Pi groups (pi_R, pi_F, pi_T, pi_S, b) for mechanistic SPAC representation.
- Removed multiple PET inputs; relies exclusively on base PT-JPL PET.
- Includes strict numerical stability floors and physical threshold hierarchy.
- Grid search: 24 variations (1 soil x 2 OWUS x 2 int x 2 dynamic x 3 caps).
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
os.environ["PTJPL_OFFLINE"] = "1"
os.environ["NO_DOWNLOAD"] = "1"

try:
    from PTJPLSM.process_PTJPLSM_table import process_PTJPLSM_table
    from PTJPL.process_PTJPL_table import process_PTJPL_table
except ImportError:
    try:
        from PTJPLSM import PTJPLSM  # noqa: F401
    except ImportError:
        print("[ERROR] Could not import PTJPLSM.")
        sys.exit(1)

# =====================================================================
# 2. CONFIGURATION
# =====================================================================
UNIFIED_CSV = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "OzFluxStations_NDVI_Soil_MetaUnified_withCanopy_LAI_FPAR_"
    "2003_2024_DAILY_FIXED.csv"
)
L6_DATA_DIR = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/L6"
OWUS_BF_CSV = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/"
    "OWUS_australia_agent_3_scientific/output_corrected/combined/"
    "results_bf__ozflux_1.csv"
)
OWUS_OPT_CSV = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/"
    "OWUS_australia_agent_3_scientific/output_corrected/combined/"
    "results_opt__ozflux_1.csv"
)

META_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia/sanjay data creation/ozflux_metadata.csv"
KG_CSV = "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/OWUS_australia/sanjay data creation/ozflux_Köppen_climate_classification.csv"

OUTPUT_DIR = (
    "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/ptjpl_ptjplsm/"
    "output_L6_OWUS_SinglePET_v12.3.1.1_DynamicBeta"
)

SIGMA = 5.670374419e-8
TOPT_FALLBACK = 30.0
LV = 2.45e6
RHO_W = 1000
FE_VALID_MIN = -100.0
FE_VALID_MAX = 1000.0

GASH_S, GASH_P, GASH_PT, GASH_ER = 1.0, 0.2, 0.02, 0.15
USE_FPAR_FOR_COVER = True
LAI_K_Beer = 0.5
COVER_MIN, COVER_MAX = 0.05, 0.98

SITE_MAPPING = {
    "gatumpasture": "AU-Gat", "otway": "AU-Otw", "silverplains": "AU-SiP", 
    "emerald": "AU-Eme", "alpinepeatland": "AU-Alp", "riggscreek": "AU-Rig",
    "yarramundiirrigated": "AU-YarI", "yarramundicontrol": "AU-YarC",
    "yanco": "AU-Ync", "dalypasture": "AU-DaP", "foggdam": "AU-Fog", 
    "sturtplains": "AU-Stp", "wallabycreek": "AU-Wal", "collie": "AU-Col",
    "boyagin": "AU-Boy", "adelaideriver": "AU-Adr", "dalyuncleared": "AU-DaS", 
    "litchfield": "AU-Lit", "greatwesternwoodlands": "AU-Gre", "ridgefield": "AU-Rgf",
    "capetribulation": "AU-Ctr", "cowbay": "AU-Cow", "whroo": "AU-Whr", 
    "tumbarumba": "AU-Tum", "loxton": "AU-Lox", "warra": "AU-War",
    "gingin": "AU-Gin", "reddirtmelonfarm": "AU-RDF", "howardsprings": "AU-How", 
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
    target = strip_trailing_digits(norm_site_string(site_name))
    files = glob.glob(os.path.join(search_dir, "*_L6.nc"))
    return sorted([f for f in files if strip_trailing_digits(norm_site_string(os.path.basename(f).replace("_L6.nc", "").replace(".nc", ""))) == target])

def calculate_stats(obs, model):
    df = pd.DataFrame({"o": obs, "m": model}).dropna()
    nans = {k: np.nan for k in ["RMSE", "Bias", "R2", "ubRMSE", "MAE", "NSE", "KGE", "r", "alpha", "beta"]}
    if len(df) < 10: return nans
    o, m = df["o"].astype(float), df["m"].astype(float)
    rmse = np.sqrt(((m - o) ** 2).mean())
    bias = (m - o).mean()
    mae = np.abs(m - o).mean()
    ubrmse = np.sqrt(np.maximum(rmse**2 - bias**2, 0))
    std_o, std_m = o.std(), m.std()
    mean_o, mean_m = o.mean(), m.mean()
    r = np.corrcoef(o, m)[0, 1] if std_o > 0 and std_m > 0 else np.nan
    r2 = (r ** 2) if np.isfinite(r) else np.nan
    denom_nse = ((o - mean_o) ** 2).sum()
    nse = (1 - (((m - o) ** 2).sum() / denom_nse) if denom_nse > 0 else np.nan)
    alpha = (std_m / std_o) if std_o > 0 else np.nan
    beta = (mean_m / mean_o) if mean_o != 0 else np.nan
    kge = (1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2) if np.isfinite(r) and np.isfinite(alpha) and np.isfinite(beta) else np.nan)
    return {"RMSE": rmse, "Bias": bias, "R2": r2, "ubRMSE": ubrmse, "MAE": mae, "NSE": nse, "KGE": kge, "r": r, "alpha": alpha, "beta": beta}

def derive_topt(df):
    gpp = df.get("GPP_SOLO", df.get("GPP"))
    ta = df.get("Ta")
    if gpp is None or ta is None: return TOPT_FALLBACK
    df_t = pd.DataFrame({"Ta": ta, "GPP": gpp}).dropna()
    df_t = df_t[(df_t["Ta"] > -5) & (df_t["Ta"] < 50) & (df_t["GPP"] > 0)]
    if df_t.empty: return TOPT_FALLBACK
    gpp_bin = df_t.groupby(df_t["Ta"].round(0))["GPP"].mean()
    return float(gpp_bin.idxmax()) if not gpp_bin.empty else TOPT_FALLBACK

def canopy_cover_proxy(daily_df):
    if USE_FPAR_FOR_COVER and "FPAR" in daily_df.columns and daily_df["FPAR"].notna().any():
        c = daily_df["FPAR"].astype(float)
    elif "LAI" in daily_df.columns and daily_df["LAI"].notna().any():
        lai = daily_df["LAI"].astype(float).clip(lower=0.0)
        c = 1.0 - np.exp(-LAI_K_Beer * lai)
        c = pd.Series(c, index=daily_df.index)
    else:
        c = pd.Series(np.nan, index=daily_df.index)
    c = c.clip(COVER_MIN, COVER_MAX).fillna(c.median() if c.notna().any() else 0.5)
    return c

def compute_gash_daily_dynamic(p_hh, c_daily, S=GASH_S, p0=GASH_P, pt0=GASH_PT, ER0=GASH_ER):
    s = p_hh.dropna()
    daily_idx = p_hh.resample("1D").sum().index
    zeros = pd.Series(0.0, index=daily_idx)
    if s.empty: return zeros
    rain_t = s[s > 0].index
    if len(rain_t) == 0: return zeros
    ev_ids = (rain_t.to_series().diff() >= pd.Timedelta(hours=6)).cumsum()
    events = s[s > 0].groupby(ev_ids).sum()
    ev_day = s[s > 0].groupby(ev_ids).apply(lambda x: x.index.min()).dt.floor("D")
    c_event = ev_day.map(c_daily.reindex(daily_idx).fillna(method="ffill").fillna(method="bfill")).astype(float).values
    p_event, pt_event, ER_event = np.clip(1.0 - c_event, 0.0, 0.999), np.clip(pt0 * c_event, 0.0, 0.2), np.clip(ER0 * c_event, 1e-6, 0.95)
    denom_log = 1.0 - p_event - pt_event
    PG_prime = np.full(len(events), S, dtype=float)
    ok = (denom_log > 1e-6) & (ER_event < denom_log)
    PG_prime[ok] = -(S / ER_event[ok]) * np.log(1.0 - ER_event[ok] / denom_log[ok])
    R_ev = events.values.astype(float)
    I_ev = np.where(R_ev < PG_prime, (1.0 - p_event - pt_event) * R_ev, (1.0 - p_event - pt_event) * PG_prime + ER_event * (R_ev - PG_prime))
    df_rain = pd.DataFrame({"R": s[s > 0], "eid": ev_ids})
    frac = (df_rain["R"] / df_rain.groupby("eid")["R"].transform("sum").replace(0, np.nan)).astype(float)
    I_hh = pd.Series(0.0, index=p_hh.index)
    I_hh.update((frac * df_rain["eid"].map(pd.Series(I_ev, index=events.index))).dropna())
    return I_hh.resample("1D").sum().reindex(daily_idx, fill_value=0.0)

def simple_kmeans(X, n_clusters=2, max_iter=100):
    np.random.seed(42)
    idx = np.random.choice(len(X), n_clusters, replace=False)
    centers = X[idx]
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([X[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centers[j] for j in range(n_clusters)])
        if np.allclose(centers, new_centers): break
        centers = new_centers
    return labels

# ---- ANALYTICAL OWUS FUNCTIONS (v12.3.1.1) ----
def calculate_f_ww(pi_R, pi_F):
    term1 = 1.0 + (pi_F / 2.0)
    sqrt_val = np.sqrt(np.maximum(0.0, (pi_F / 2.0 + 1.0)**2 - 2.0 * pi_F * pi_R))
    f_ww = 1.0 - (1.0 / (2.0 * pi_R)) * (term1 - sqrt_val)
    return np.clip(f_ww, 0.0, 1.0)

def owus_s_of_beta(beta, pi_R, pi_F, pi_T, pi_S, b):
    beta = np.atleast_1d(beta)
    s_out = np.zeros_like(beta)
    mask = beta > 1e-7
    if not np.any(mask): return s_out if len(s_out) > 1 else s_out[0]
    b_m = beta[mask]
    denom_inner = 1.0 - (1.0 - b_m) * pi_R
    denom_inner = np.where(np.abs(denom_inner) < 1e-9, 1e-9, denom_inner)
    bracket = 2.0 * (1.0 - b_m) - (b_m * pi_F) / denom_inner
    sqrt_val = np.sqrt(np.maximum(0, 1.0 + (4.0 * b_m * pi_S**2 / pi_T) * bracket))
    s_val = (pi_T / (2.0 * b_m * pi_S) * (sqrt_val - 1.0))**(-1.0 / b)
    s_out[mask] = np.clip(s_val, 0.0, 1.0)
    return s_out if len(s_out) > 1 else s_out[0]

def _get_beta_single_set(proxy_arr, pi_params, b_soil):
    pR, pF, pT, pS = pi_params
    f_ww = calculate_f_ww(pR, pF)
    beta_grid = np.linspace(1e-7, float(f_ww), 1000)
    s_grid = owus_s_of_beta(beta_grid, pR, pF, pT, pS, b_soil)
    # s(beta) is monotonic increasing; interp expects xp to be increasing
    beta_values = np.interp(proxy_arr, s_grid, beta_grid, left=0.0, right=float(f_ww))
    return beta_values, float(f_ww)

def get_beta_vectorized(proxy_arr, pi_params):
    pR, pF, pT, pS, b_soil = pi_params
    # Broadcast scalar params if any are numpy arrays (dynamic regime case)
    if any(isinstance(x, np.ndarray) for x in [pR, pF, pT, pS]):
        N = len(proxy_arr)
        pR_arr = np.full(N, pR) if not isinstance(pR, np.ndarray) else pR
        pF_arr = np.full(N, pF) if not isinstance(pF, np.ndarray) else pF
        pT_arr = np.full(N, pT) if not isinstance(pT, np.ndarray) else pT
        pS_arr = np.full(N, pS) if not isinstance(pS, np.ndarray) else pS
        
        params_stack = np.stack([pR_arr, pF_arr, pT_arr, pS_arr]).T
        unique_pis, inverse_indices = np.unique(params_stack, axis=0, return_inverse=True)
        beta_out = np.zeros_like(proxy_arr)
        f_ww_out = np.zeros_like(proxy_arr)
        for i, upi in enumerate(unique_pis):
            mask = (inverse_indices == i)
            b_val, f_val = _get_beta_single_set(proxy_arr[mask], upi, b_soil)
            beta_out[mask] = b_val
            f_ww_out[mask] = f_val
        return beta_out, f_ww_out
    else:
        # Static case
        return _get_beta_single_set(proxy_arr, [pR, pF, pT, pS], b_soil)

# =====================================================================
# 4. CORE PROCESSING
# =====================================================================
def process_single_nc(site_name, nc_path, df_owus_bf, df_owus_opt, meta_all, df_science_meta):
    site_suffix = os.path.basename(nc_path).replace("_L6.nc", "")
    site_label = f"{site_name}::{site_suffix}"
    site_dir = os.path.join(OUTPUT_DIR, norm_site_string(site_suffix))
    os.makedirs(site_dir, exist_ok=True)
    
    out_csv = os.path.join(site_dir, f"{site_suffix}_v12.3.1.1_results.csv")
    metrics_txt = os.path.join(site_dir, f"{site_suffix}_v12.3.1.1_metrics.txt")
    manifest_csv = os.path.join(site_dir, f"{site_suffix}_v12.3.1.1_manifest.csv")

    print(f"\n=== PROCESSING v12.3.1.1: {site_label} ===")

    sn_low = norm_site_string(site_suffix)
    sc_row = df_science_meta[df_science_meta["site_match"] == sn_low]
    if sc_row.empty:
        sc_row = df_science_meta[df_science_meta["site_match"] == strip_trailing_digits(sn_low)]
        
    kg_label = sc_row["KG_Label"].iloc[0] if not sc_row.empty else "Unknown"
    kg_code = sc_row["KG_Code"].iloc[0] if not sc_row.empty else -1
    veg_type = sc_row["Vegetation Type"].iloc[0] if not sc_row.empty else "Unknown"
    h_meta = sc_row["Canopy_Height_m"].iloc[0] if not sc_row.empty else 0.5
    
    print(f"  [INFO] Metadata: KG={kg_label}, Veg={veg_type}, H={h_meta}m")

    ds = None
    for eng in ["h5netcdf", None, "scipy"]:
        try: ds = xr.open_dataset(nc_path, engine=eng); break
        except Exception: continue
    if ds is None:
        print(f"  [ERROR] Cannot open {nc_path}"); return

    lat_val = float(ds.attrs.get("latitude", ds.get("latitude", np.array(0.0)).values.flat[0]))
    lon_val = float(ds.attrs.get("longitude", ds.get("longitude", np.array(0.0)).values.flat[0]))
    ds = ds.squeeze(drop=True)

    def get_pi_groups(df_owus, label=""):
        if df_owus.empty: return 0.5, 1.0, 1e5, 200.0, 5.39
        cn = strip_trailing_digits(norm_site_string(site_suffix))
        tid = SITE_MAPPING.get(cn, site_suffix)
        m = df_owus[df_owus["siteID"].str.lower() == tid.lower()]
        if m.empty:
            dist = np.sqrt((df_owus["lon"] - lon_val)**2 + (df_owus["lat"] - lat_val)**2)
            m = df_owus.loc[[dist.idxmin()]]
        b_row = m.iloc[0]
        return (float(b_row["pi_R_median"]), float(b_row["pi_F_median"]), 
                float(b_row["pi_T_median"]), float(b_row["pi_S_median"]), float(b_row.get("b", 5.39)))

    pi_bf = get_pi_groups(df_owus_bf, "BF")
    pi_opt = get_pi_groups(df_owus_opt, "OPT")

    vn = ["Fsd", "Fsu", "Fld", "Flu", "Fn", "Fg", "Ta", "RH", "Sws", "Fe", "Precip", "GPP_SOLO", "GPP"]
    df_raw = ds[[v for v in vn if v in ds]].to_dataframe()
    if "time" in df_raw.columns: df_raw = df_raw.set_index("time")
    p_hh = (ds["Precip"].to_series() if "Precip" in ds else pd.Series(0.0, index=df_raw.index))
    if "time" not in p_hh.index.names: p_hh.index = df_raw.index

    df_d = df_raw.resample("1D").mean(numeric_only=True)
    if "Fe" in df_d:
        df_d.loc[(df_d["Fe"] < FE_VALID_MIN) | (df_d["Fe"] > FE_VALID_MAX), "Fe"] = np.nan
    df_d["Precip"] = p_hh.resample("1D").sum()
    df_d["lat"], df_d["lon"] = lat_val, lon_val

    if {"Fsd", "Fsu"}.issubset(df_d.columns):
        alb_c = (df_d["Fsu"] / df_d["Fsd"]).replace([np.inf, -np.inf], np.nan).dropna()
        alb_c = alb_c[(alb_c > 0) & (alb_c < 1)]
        df_d["albedo"] = (df_d["Fsu"] / df_d["Fsd"]).clip(float(alb_c.quantile(0.01)) if len(alb_c) > 10 else 0.05, float(alb_c.quantile(0.99)) if len(alb_c) > 10 else 0.80)
    if "Flu" in df_d: df_d["ST_C"] = ((df_d["Flu"] / (0.98 * SIGMA)) ** 0.25) - 273.15
    if "Fn" in df_d: df_d["Rn"] = df_d["Fn"]
    elif {"Fsd", "Fsu"}.issubset(df_d.columns): df_d["Rn"] = (df_d["Fsd"] - df_d["Fsu"] + (df_d["Fld"] - df_d["Flu"] if {"Fld", "Flu"}.issubset(df_d) else 0))
    else: df_d["Rn"] = np.nan
    if "RH" in df_d and df_d["RH"].max() > 1.5: df_d["RH"] /= 100.0

    sn, sb = norm_site_string(site_name), strip_trailing_digits(norm_site_string(site_name))
    ms = meta_all[meta_all["site_norm"] == sn].copy()
    if ms.empty: ms = meta_all[meta_all["site_base"] == sb].copy()
    if ms.empty: ms = meta_all[meta_all["site_norm"].str.contains(sb)]
    
    if not ms.empty:
        ms = ms.copy()
        ms["date"] = pd.to_datetime(ms["date"])
        ma = ms.groupby("date").mean(numeric_only=True).sort_index().reindex(df_d.index).interpolate(limit_direction="both")
        df_d["NDVI"] = ma.get("ndvi", np.nan).clip(0, 1)
        df_d["LAI"] = ma.get("lai", np.nan)
        df_d["FPAR"] = ma.get("fpar", np.nan).clip(0, 1)
        df_d["canopy_height_meters"] = h_meta if h_meta > 0.5 else ma.get("canopy_height", 0.5)
        df_d["field_capacity"] = ma.get("fc_000_005" if "fc_000_005" in ma.columns and ma["fc_000_005"].notna().any() else "fc_0_30cm", 0.30)
        df_d["wilting_point"] = ma.get("wp_000_005" if "wp_000_005" in ma.columns and ma["wp_000_005"].notna().any() else "wp_0_30cm", 0.10)
    else:
        df_d["NDVI"], df_d["canopy_height_meters"], df_d["field_capacity"], df_d["wilting_point"] = 0.5, h_meta, 0.3, 0.1

    if "Sws" in df_d: df_d["SM"] = df_d["Sws"].clip(lower=df_d["wilting_point"], upper=df_d["field_capacity"])
    else: df_d["SM"] = np.nan

    df_d["fAPARmax"] = df_d["FPAR"].quantile(0.95) if "FPAR" in df_d and df_d["FPAR"].notna().any() else (df_d["NDVI"].quantile(0.95) if "NDVI" in df_d else 0.7)
    df_d["Topt_C"] = derive_topt(df_d)

    ns_nd = df_d["NDVI"].dropna()
    if not ns_nd.empty:
        nmin, nmax = (float(ns_nd.quantile(0.05)), float(ns_nd.quantile(0.95))) if len(ns_nd) > 20 else (0.10, 0.85)
        df_d["emissivity"] = 0.97 * (1 - ((df_d["NDVI"] - nmin) / (nmax - nmin)).clip(0, 1) ** 2) + 0.99 * ((df_d["NDVI"] - nmin) / (nmax - nmin)).clip(0, 1) ** 2
    else: df_d["emissivity"] = 0.98

    inp = df_d.copy()
    if "Ta" in inp: inp.rename(columns={"Ta": "Ta_C"}, inplace=True)
    req = ["NDVI", "ST_C", "albedo", "Ta_C", "RH", "SM", "Rn", "fAPARmax", "lat", "lon"]
    for r in req:
        if r not in inp.columns: inp[r] = np.nan
    df_run = inp[inp[req].notna().all(axis=1)].copy()
    if df_run.empty: print("  [ERROR] No valid rows"); return

    prec = df_d.reindex(df_run.index)["Precip"]
    dry_mask = ((prec == 0) & (prec.shift(1) == 0)).fillna(False)
    wet_mask = ((prec > 0) | (prec.shift(1) > 0)).fillna(False)

    mg, mw = types.ModuleType("gedi_canopy_height"), types.ModuleType("soil_capacity_wilting")
    mg.load_canopy_height = lambda **k: np.full(len(df_run), df_run["canopy_height_meters"].iloc[0])
    mw.load_field_capacity = lambda **k: np.full(len(df_run), df_run["field_capacity"].iloc[0])
    mw.load_wilting_point = lambda **k: np.full(len(df_run), df_run["wilting_point"].iloc[0])
    sys.modules["gedi_canopy_height"], sys.modules["soil_capacity_wilting"] = mg, mw

    res_sm, res_base = process_PTJPLSM_table(df_run), process_PTJPL_table(df_run)

    N = len(df_run)
    LE_can_orig = res_base[next((c for c in ["LE_canopy", "LE_canopy_Wm2"] if c in res_base.columns), None)].values
    LE_soil_sm = res_sm[next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_sm.columns), None)].values if next((c for c in ["LE_soil", "LE_soil_Wm2"] if c in res_sm.columns), None) else np.zeros(N)
    LE_int_sm = res_sm[next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_sm.columns), None)].values if next((c for c in ["LE_interception", "LE_interception_Wm2"] if c in res_sm.columns), None) else np.zeros(N)
    LE_tot_base = res_base[next((c for c in ["LE", "LE_Wm2", "evapotranspiration"] if c in res_base.columns), None)].values if next((c for c in ["LE", "LE_Wm2", "evapotranspiration"] if c in res_base.columns), None) else np.zeros(N)

    cf = LV / 86400.0
    Ei_mm = compute_gash_daily_dynamic(p_hh, canopy_cover_proxy(df_d.reindex(df_run.index)).reindex(p_hh.resample("1D").sum().index)).reindex(df_run.index).fillna(0).values
    LE_int_gash = Ei_mm * cf
    pet_pt = res_base["PET"].values if "PET" in res_base.columns else np.full(N, 1000.0)

    wp, fc = df_run["wilting_point"].values, df_run["field_capacity"].values
    s_arr = np.clip((df_run["SM"].values - wp) / np.maximum(fc - wp, 1e-6), 0.0, 1.0)
    
    cluster_feat = pd.DataFrame(index=df_run.index)
    cluster_feat['LAI_roll'] = df_run['LAI'].rolling(window=30, min_periods=1).mean()
    cluster_feat['PET_roll'] = pd.Series(pet_pt, index=df_run.index).rolling(window=30, min_periods=1).mean()
    cluster_feat = cluster_feat.fillna(cluster_feat.median()).values
    X_scaled = (cluster_feat - np.mean(cluster_feat, axis=0)) / (np.std(cluster_feat, axis=0) + 1e-6)
    regimes = simple_kmeans(X_scaled, n_clusters=2)
    
    dyn_mult_ss = np.where(regimes == 0, 0.9, 1.1)
    dyn_mult_sw = np.where(regimes == 0, 0.9, 1.1)
    dyn_mult_bw = np.where(regimes == 0, 1.05, 0.95)

    fM = np.clip(df_run["FPAR"].values / df_run["fAPARmax"].values, 0.0, 1.0)
    weight = df_run["RH"].values ** (4.0 * (1.0 - s_arr) * (1.0 - df_run["RH"].values))

    owus_results = {}; manifest_records = []

    def calc_le_owus_analytical(pi_set, soil_comp, int_comp, pet_limit):
        # Apply the analytical inversion
        beta_full, f_ww = get_beta_vectorized(s_arr, pi_set)
        f_TRM = (1.0 - weight) * fM + weight * beta_full
        fM_safe = np.maximum(fM, 0.05)
        LE_can = np.divide(LE_can_orig * f_TRM, fM_safe, out=np.zeros_like(LE_can_orig), where=(fM_safe > 0))
        PET_rem = np.maximum(pet_limit - int_comp, 0.0)
        LE_sc_raw = soil_comp + LE_can
        LE_tc = int_comp + np.clip(LE_sc_raw, 0, PET_rem)
        LE_cc = int_comp + soil_comp + np.maximum(LE_can - np.maximum(LE_sc_raw - PET_rem, 0.0), 0.0)
        LE_sc = int_comp + soil_comp + LE_can * np.clip(np.where(LE_sc_raw > PET_rem, np.divide(PET_rem - soil_comp, LE_can, out=np.ones_like(LE_can), where=LE_can > 0), 1.0), 0, 1)
        return LE_tc, LE_cc, LE_sc

    soil_opts = {"SM": (LE_soil_sm, LE_int_sm)}
    owus_opts = {"BF": pi_bf, "OPT": pi_opt}
    int_opts = {"PTJPL": None, "GASH": LE_int_gash}
    dyn_opts = {"Stat": False, "Dyn": True}

    for s_name, (soil_v, int_ptjpl) in soil_opts.items():
        for o_name, pi_base in owus_opts.items():
            for i_name, i_val in int_opts.items():
                for d_name, is_dyn in dyn_opts.items():
                    act_int = int_ptjpl if i_name == "PTJPL" else i_val
                    if is_dyn:
                        # Dynamic parameter shifting applied to Pi groups (approximate shifts)
                        pi_curr = (pi_base[0] * dyn_mult_ss, pi_base[1], pi_base[2], pi_base[3], pi_base[4])
                    else:
                        pi_curr = pi_base
                    
                    bid = f"LE_{s_name}_{o_name}_PT_{i_name}_Analytical_{d_name}"
                    tc, cc, sc = calc_le_owus_analytical(pi_curr, soil_v, act_int, pet_pt)
                    owus_results[f"{bid}_TC"], owus_results[f"{bid}_CC"], owus_results[f"{bid}_SC"] = tc, cc, sc
                    for cap in ["TC", "CC", "SC"]:
                        manifest_records.append({
                            "Model_ID": f"{bid}_{cap}", "Soil": s_name, "OWUS": o_name, 
                            "PET": "PT", "Cap": cap, "Int": i_name, "Shape": "Analytical", "Dynamic": d_name,
                            "KG_Label": kg_label, "Veg": veg_type
                        })

    output = pd.DataFrame(index=df_run.index)
    output["LE_Obs"] = df_d.reindex(df_run.index)["Fe"]
    output["LE_PTJPL_Base"] = LE_tot_base
    output["LE_PTJPL_SM"] = res_sm["LE"]
    for k, v in owus_results.items(): output[k] = v
    output["Precip"], output["dry_day"], output["wet_day"] = prec, dry_mask.astype(int), wet_mask.astype(int)
    output["SM_PAW"] = s_arr
    output["canopy_cover_c"] = canopy_cover_proxy(df_d.reindex(df_run.index)).values
    output["KG_Label"], output["Vegetation_Type"] = kg_label, veg_type
    output["Year"], output["Season"] = output.index.year, output.index.month.map(get_au_season)
    
    # SM regime proxies (from OPT Analytical)
    f_ww_opt = calculate_f_ww(pi_opt[0], pi_opt[1])
    s_wilt_prox = float(owus_s_of_beta(0.05 * f_ww_opt, *pi_opt))
    s_star_prox = float(owus_s_of_beta(0.95 * f_ww_opt, *pi_opt))
    regime_sm_code = np.full(s_arr.shape, -1, dtype=int)
    regime_sm_code[s_arr <= s_wilt_prox] = 0
    regime_sm_code[(s_arr > s_wilt_prox) & (s_arr <= s_star_prox)] = 1
    regime_sm_code[s_arr > s_star_prox] = 2
    output["regime_sm_code"] = regime_sm_code

    output.to_csv(out_csv, float_format="%.4f")
    pd.DataFrame(manifest_records).to_csv(manifest_csv, index=False)

    all_model_cols = ["LE_PTJPL_Base", "LE_PTJPL_SM"] + list(owus_results.keys())
    hdr = f"{'Model':<60s} | RMSE | Bias |   R2 |  NSE |  KGE |    r |    a |    b"
    sep = "-" * 120

    def write_block(f, label, mask, models_list):
        if mask is not None:
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
            if mask is not None:
                st = calculate_stats(o_data, output[m][mask])
            else:
                st = calculate_stats(o_data, output[m])
            f.write(f"{m.replace('LE_',''):<60s} | {st['RMSE']:4.1f} | {st['Bias']:4.1f}"
                    f" | {st['R2']:4.2f} | {st['NSE']:4.2f} | {st['KGE']:4.2f}"
                    f" | {st['r']:.2f} | {st['alpha']:.2f}"
                    f" | {st['beta']:.2f}\n")

    with open(metrics_txt, "w") as f:
        f.write(f"Site: {site_name} | v12.3.1.1 Analytical | KG: {kg_label}\n")
        f.write(f"SM regime (proxies): s_wilt={s_wilt_prox:.3f} | s_star={s_star_prox:.3f}\n")
        write_block(f, "ALL DAYS", None, all_model_cols)

def process_site(site_name):
    df_bf = pd.read_csv(OWUS_BF_CSV) if os.path.exists(OWUS_BF_CSV) else pd.DataFrame()
    df_opt = pd.read_csv(OWUS_OPT_CSV) if os.path.exists(OWUS_OPT_CSV) else pd.DataFrame()
    meta_all = pd.read_csv(UNIFIED_CSV, low_memory=False)
    meta_all.columns = [c.strip().lower() for c in meta_all.columns]
    meta_all["site_norm"] = meta_all["site"].apply(norm_site_string)
    meta_all["site_base"] = meta_all["site_norm"].apply(strip_trailing_digits)
    df_sci_meta = pd.read_csv(META_CSV)
    df_kg = pd.read_csv(KG_CSV)
    df_sci_meta = pd.merge(df_sci_meta, df_kg[["original_site", "KG_Code", "KG_Label"]], on="original_site", how="left")
    df_sci_meta["site_match"] = df_sci_meta["original_site"].apply(norm_site_string)
    for nc in find_nc_files(site_name, L6_DATA_DIR):
        process_single_nc(site_name, nc, df_bf, df_opt, meta_all, df_sci_meta)

if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(1)
    process_site(sys.argv[1])
